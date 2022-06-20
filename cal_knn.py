'''
@article{hamilton2022unsupervised,
  title={Unsupervised Semantic Segmentation by Distilling Feature Correspondences},
  author={Hamilton, Mark and Zhang, Zhoutong and Hariharan, Bharath and Snavely, Noah and Freeman, William T},
  journal={arXiv preprint arXiv:2203.08414},
  year={2022}
}
'''

import argparse
from dataset.data import ContrastiveSegDataset, get_transform
import os
from os.path import join
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from build import build_model
from utils.common_utils import parse
from torch.utils.data import DataLoader


def get_feats(model, loader) -> torch.Tensor:
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


def main(opt: dict) -> None:
    data_dir = join(opt["output_dir"], "data")
    log_dir = join(opt["output_dir"], "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join("./", "nns"), exist_ok=True)

    image_sets = ["val", "train"]
    dataset_name = opt["dataset"]["data_type"]
    # dataset_names = ["cocostuff27", "cityscapes", "potsdam"]
    crop_types = ["five", None]
    model_type = opt["pretrained"]["model_type"]

    res = 224
    n_batches = 16  # temp

    if opt["pretrained"]["name"] == "dino":
        no_ap_model = build_model(opt["pretrained"], model_type, data_dir)
    else:
        cut_model = build_model(opt["pretrained"], model_type, data_dir)
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1])
    par_model = torch.nn.DataParallel(no_ap_model.cuda())

    for crop_type in crop_types:  # five, None
        for image_set in image_sets:  # val, train
            feature_cache_file = join("nns", "nns_{}_{}_{}_{}_{}.npz".format(
                model_type, dataset_name, image_set, crop_type, res))

            if not os.path.exists(feature_cache_file):
                print("{} not found, computing".format(feature_cache_file))
                dataset = ContrastiveSegDataset(  # TODO need fix
                    pytorch_data_dir=opt["dataset"]["data_path"],
                    dataset_name=dataset_name,
                    crop_type=crop_type,
                    image_set=image_set,
                    transform=get_transform(res, False, opt["dataset"]["crop_type"]),
                    target_transform=get_transform(res, True, opt["dataset"]["crop_type"]),
                    cfg=opt,
                )

                loader = DataLoader(dataset, batch_size=opt["dataloader"]["batch_size"], shuffle=False,
                                    num_workers=opt["dataloader"]["num_workers"], pin_memory=False)

                with torch.no_grad():
                    normed_feats = get_feats(par_model, loader)
                    all_nns = []
                    step = normed_feats.shape[0] // n_batches
                    print(normed_feats.shape)
                    for i in tqdm(range(0, normed_feats.shape[0], step)):
                        torch.cuda.empty_cache()
                        batch_feats = normed_feats[i:i + step, :]
                        pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                        # np.matmul(batch_feats, np.transpose(normed_feats,(1,0))))
                        all_nns.append(torch.topk(pairwise_sims, 30)[1])  # 상위 30 개의 Tensor's index
                        del pairwise_sims
                    nearest_neighbors = torch.cat(all_nns, dim=0)

                    np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                    print("Saved NNs", model_type, dataset_name, image_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="Path to option JSON file.")
    parser_args = parser.parse_args()
    parser_opt = parse(parser_args.opt)
    main(parser_opt)
