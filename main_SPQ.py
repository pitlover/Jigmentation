from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import kornia.augmentation as Kg
from dataset.data import ContrastiveSegDataset, get_transform


def get_args_parser():
    parser = argparse.ArgumentParser('SPQ', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--if_download', default=False, type=bool, help="""Whether to download the dataset or not.""")
    parser.add_argument('--data_dir', default="./data", type=str, help="""Path of the dataset to be installed.""")
    parser.add_argument('--batch_size', default=256, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--input_size', default=32, type=int, help="""Input image size, default is set to CIFAR10.""")

    parser.add_argument('--N_books', default=8, type=int, help="""The number of the codebooks.""")
    parser.add_argument('--N_words', default=16, type=int,
                        help="""The number of the codewords. It should be a power of two.""")
    parser.add_argument('--L_word', default=16, type=int, help="""Dimensionality of the codeword.""")
    parser.add_argument('--soft_quantization_scale', default=5.0, type=float,
                        help="""Soft-quantization scaling parameter.""")
    parser.add_argument('--contrastive_temperature', default=0.5, type=float,
                        help="""Contrastive learning Temperature scaling parameter.""")

    parser.add_argument('--num_cls', default="10", type=int,
                        help="""The number of classes in the dataset for evaluation, default is set to CIFAR10""")
    parser.add_argument('--eval_epoch', default=100, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--Top_N', default=1000, type=int,
                        help="""Top N number of images to be retrieved for evaluation.""")

    return parser


class CQCLoss(T.nn.Module):

    def __init__(self, device, batch_size, tau_cqc):
        super(CQCLoss, self).__init__()
        self.batch_size = batch_size
        self.tau_cqc = tau_cqc
        self.device = device
        self.COSSIM = T.nn.CosineSimilarity(dim=-1)
        self.CE = T.nn.CrossEntropyLoss(reduction="sum")
        self.get_corr_mask = self._get_correlated_mask().type(T.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = T.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(T.bool)
        return mask.to(self.device)

    def forward(self, Xa, Xb, Za, Zb):
        XaZb = T.cat([Xa, Zb], dim=0)
        XbZa = T.cat([Xb, Za], dim=0)

        Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))

        Rab = T.diag(Cossim_ab, self.batch_size)
        Lab = T.diag(Cossim_ab, -self.batch_size)
        Pos_ab = T.cat([Rab, Lab]).view(2 * self.batch_size, 1)
        Neg_ab = Cossim_ab[self.get_corr_mask].view(2 * self.batch_size, -1)

        Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))

        Rba = T.diag(Cossim_ba, self.batch_size)
        Lba = T.diag(Cossim_ba, -self.batch_size)
        Pos_ba = T.cat([Rba, Lba]).view(2 * self.batch_size, 1)
        Neg_ba = Cossim_ba[self.get_corr_mask].view(2 * self.batch_size, -1)

        logits_ab = T.cat((Pos_ab, Neg_ab), dim=1)
        logits_ab /= self.tau_cqc

        logits_ba = T.cat((Pos_ba, Neg_ba), dim=1)
        logits_ba /= self.tau_cqc

        labels = T.zeros(2 * self.batch_size).to(self.device).long()

        loss = self.CE(logits_ab, labels) + self.CE(logits_ba, labels)
        return loss / (2 * self.batch_size)


def train_SPQ(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = T.device('cuda')

    sz = args.input_size  # 32
    data_dir = args.data_dir
    batch_size = args.batch_size

    N_books = args.N_books  # number of codebook : 8
    N_words = args.N_words  # number of codevector(words) : 16 -> pow of 2
    L_word = args.L_word  # code book dimension : 16
    tau_q = args.soft_quantization_scale
    tau_cqc = args.contrastive_temperature

    N_bits = int(N_books * np.sqrt(N_words))

    # Define the data augmentation following the setup of SimCLR
    Augmentation = nn.Sequential(
        Kg.RandomResizedCrop(size=(sz, sz)),
        Kg.RandomHorizontalFlip(p=0.5),
        Kg.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        Kg.RandomGrayscale(p=0.2),
        Kg.RandomGaussianBlur((int(0.1 * sz), int(0.1 * sz)), (0.1, 2.0), p=0.5))

    opt = {"data_type": "cocostuff27",
           "data_path": "../Datasets/cocostuff",
           "loader_crop_type": "center",
           "crop_type": "five",
           "crop_ratio": 0.5,
           "res": 224,
           "num_neighbors": 7}

    trainset = ContrastiveSegDataset(
        pytorch_data_dir=opt["data_path"],
        dataset_name=opt["data_type"],
        crop_type=opt["crop_type"],
        model_type="dino",
        image_set="train",
        transform=get_transform(opt["res"], False, opt["loader_crop_type"]),
        target_transform=get_transform(opt["res"], True, opt["loader_crop_type"]),
        cfg=opt,
        num_neighbors=opt["num_neighbors"],
        mask=True,
        pos_images=False,
        pos_labels=False
    )
    trainloader = T.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                          num_workers=args.num_workers)

    class Quantization_Head(nn.Module):
        def __init__(self, N_words, N_books, L_word, tau_q):
            super(Quantization_Head, self).__init__()
            self.fc = nn.Linear(512, N_books * L_word)
            nn.init.xavier_uniform_(self.fc.weight)

            # Codebooks
            self.C = T.nn.Parameter(Variable((T.randn(N_words, N_books * L_word)).type(T.float32), requires_grad=True))
            nn.init.xavier_uniform_(self.C)

            self.N_books = N_books
            self.L_word = L_word
            self.tau_q = tau_q

        def forward(self, input):
            X = self.fc(input)
            Z = Soft_Quantization(X, self.C, self.N_books, self.tau_q)
            return X, Z

    Q = Quantization_Head(N_words, N_books, L_word, tau_q)
    net = nn.Sequential(ResNet_Baseline(BasicBlock, [2, 2, 2, 2]), Q)

    net.cuda(device)

    criterion = CQCLoss(device, batch_size, tau_cqc)

    optimizer = T.optim.Adam(net.parameters(), lr=3e-4, weight_decay=10e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0, last_epoch=-1)

    MAX_mAP = 0.0
    mAP = 0.0

    for epoch in range(5000):  # loop over the dataset multiple times

        print('Epoch: %d, Learning rate: %.4f' % (epoch, scheduler.get_last_lr()[0]))
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs = data['img'].to(device, non_blocking=True)
            Ia = Augmentation(inputs)
            Ib = Augmentation(inputs)

            optimizer.zero_grad()

            Xa, Za = net(Ia)
            Xb, Zb = net(Ib)

            loss = criterion(Xa, Xb, Za, Zb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # print every 10 mini-batches
                print('[%3d] loss: %.4f, mAP: %.4f, MAX mAP: %.4f' %
                      (i + 1, running_loss / 10, mAP, MAX_mAP))
                running_loss = 0.0

        if epoch >= 10:
            scheduler.step()


def Soft_Quantization(X, C, N_books, tau_q):
    L_word = int(C.size()[1] / N_books)
    x = T.split(X, L_word, dim=1)
    c = T.split(C, L_word, dim=1)
    for i in range(N_books):
        soft_c = F.softmax(squared_distances(x[i], c[i]) * (-tau_q), dim=-1)
        if i == 0:
            Z = soft_c @ c[i]
        else:
            Z = T.cat((Z, soft_c @ c[i]), dim=1)
    return Z


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return T.sum(diff * diff, -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Baseline(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_Baseline, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_out = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_out(out))
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPQ', parents=[get_args_parser()])
    args = parser.parse_args()
    train_SPQ(args)
