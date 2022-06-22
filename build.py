from typing import Optional
import torch
from os.path import join
from torchvision import models
import wget
import os
import torch.nn as nn
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from model import DinoFeaturizer, LambdaLayer

def build_model(opt: dict, data_dir: str):
    # opt = opt["model"]  //   opt["pretrained"]
    model_type = opt["name"].lower()

    if model_type == "dino":
        model = nn.Sequential(
            DinoFeaturizer(20, opt),  # dim doesent matter
            LambdaLayer(lambda p: p[0])
        )

    elif model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download",
                          model_file)
        model_weights = torch.load(model_file)
        # model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
        #                          'model' in name}
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    if model_type != "dino":
        model.eval()
        model.cuda()

    else:
        bn_momentum = opt.get("bn_momentum", None)
        if bn_momentum is not None:
            for module_name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.momentum = bn_momentum
        bn_eps = opt.get("bn_eps", None)
        if bn_eps is not None:
            for module_name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eps = bn_eps

    return model


def split_params_for_optimizer(model, opt):
    # opt = opt["optimizer"]
    params_small_lr = []
    params_small_lr_no_wd = []
    params_base_lr = []
    params_base_lr_no_wd = []
    for param_name, param_value in model.named_parameters():
        param_value: torch.Tensor
        if not param_value.requires_grad:
            continue
        if "encoder" in param_name:
            if (param_value.ndim > 1) and ("position" not in param_name):
                params_small_lr.append(param_value)
            else:
                params_small_lr_no_wd.append(param_value)
        else:  # decoder
            if (param_value.ndim > 1) and ("position" not in param_name):
                params_base_lr.append(param_value)
            else:
                params_base_lr_no_wd.append(param_value)

    same_lr = opt.get("same_lr", True)
    encoder_weight = 1.0 if same_lr else 0.1
    params_for_optimizer = [
        {"params": params_base_lr},
        {"params": params_base_lr_no_wd, "weight_decay": 0.0},
        # {"params": params_small_lr, "lr": opt["lr"] * encoder_weight, "weight_decay": opt["weight_decay"] * 0.1},
        {"params": params_small_lr, "lr": opt["lr"] * encoder_weight},
        {"params": params_small_lr_no_wd, "lr": opt["lr"] * encoder_weight, "weight_decay": 0.0},
    ]
    return params_for_optimizer


def build_optimizer(params, opt: dict):
    # opt = opt["optimizer"]
    optimizer_type = opt.get("name", "adamw").lower()

    if optimizer_type == "adamw":
        optimizer = AdamW(
            params,
            lr=opt["lr"],
            betas=tuple(opt.get("betas", (0.9, 0.99))),
            weight_decay=opt["weight_decay"],
            eps=opt.get("eps", 1e-8),
        )
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}.")

    return optimizer


def build_scheduler(opt: dict, optimizer, loader, start_epoch):
    # opt = opt BE CAREFUL!
    scheduler_type = opt["scheduler"]['name'].lower()

    if scheduler_type == "onecycle":
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]

        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # noqa
            optimizer,
            # max_lr=opt['optimizer']['lr'],
            max_lr=max_lrs,
            epochs=opt['train']['epoch'] + 1,
            steps_per_epoch=len(loader) // opt["train"]["num_accum"],
            cycle_momentum=opt["scheduler"].get("cycle_momentum", True),
            base_momentum=0.85,
            max_momentum=0.95,
            pct_start=opt["scheduler"]["pct_start"],
            last_epoch=start_epoch - 1,
            div_factor=opt["scheduler"]['div_factor'],
            final_div_factor=opt["scheduler"]['final_div_factor']
        )
    else:
        raise ValueError(f"Unsupported scheduler type {scheduler_type}.")

    return scheduler


def build_dataset(opt: dict, mode: str = "train"):
    # opt = opt["dataset"]
    return DepthDataset( # TODO dataset
        data_path=opt["data_path"],
        data_type=opt["data_type"],
        mode=mode,
        img_size=opt.get("img_size", None),
        height_drop=opt.get("height_drop", (0.0, 0)),
        width_drop=opt.get("width_drop", (0.0, 0)),
        use_right=opt.get("use_right", False),
        drop_edge=opt.get("drop_edge", False),
        clip_depth=opt.get("clip_depth", None),
    )


def build_dataloader(dataset,
                     opt: dict, shuffle: bool = True, pin_memory: bool = True,
                     batch_size: Optional[int] = None) -> DataLoader:
    # opt = opt["dataloader"]

    if batch_size is None:  # override
        batch_size = opt["batch_size"]

    if not dist.is_initialized():
        return DataLoader(
            dataset,
            batch_size=max(batch_size, 1),
            shuffle=shuffle,
            num_workers=opt.get("num_workers", 4),
            pin_memory=pin_memory,
            drop_last=shuffle,
        )
    else:
        assert dist.is_available() and dist.is_initialized()
        ddp_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=shuffle,
        )
        world_size = dist.get_world_size()
        return DataLoader(
            dataset,
            batch_size=max(batch_size // world_size, 1),
            num_workers=(opt.get("num_workers", 4) + world_size - 1) // world_size,
            pin_memory=pin_memory,
            sampler=ddp_sampler,
        )
