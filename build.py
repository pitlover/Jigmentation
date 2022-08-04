from typing import Optional, Dict
import torch
from torch.optim import Adam, AdamW, SGD
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from model.STEGO import (STEGOmodel)
from model.LambdaLayer import LambdaLayer
from model.dino.DinoFeaturizer import DinoFeaturizer
from dataset.data import ContrastiveSegDataset, get_transform
from torchvision import transforms as T
from loss import *
from utils.layer_utils import ClusterLookup
from model.DULLI import DULLI
from model.HOI import HOI
from model.BOB import BOB
from model.VQVAE import VQVAE
from model.JIRANO import JIRANO
from model.HIER import HIER
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_model(opt: dict, n_classes: int = 27, device: torch.device = "cuda"):
    # opt = opt["model"]
    model_type = opt["name"].lower()

    if "stego" in model_type:
        model = STEGOmodel.build(
            opt=opt,
            n_classes=n_classes
        )

    elif "dulli" in model_type:
        model = DULLI.build(
            opt=opt,
            n_classes=n_classes
        )

    elif "hoi" in model_type:
        model = HOI.build(
            opt=opt,
            n_classes=n_classes,
            device=device
        )

    elif "vqvae" in model_type:
        model = VQVAE.build(
            opt=opt,
            n_classes=n_classes,
            device=device
        )

    elif "hier" in model_type:
        model = HIER.build(
            opt=opt,
            n_classes=n_classes,
            device=device
        )

    elif "bob" in model_type:
        model = BOB.build(
            opt=opt,
            n_classes=n_classes,
            device=device
        )

    elif "jirano" in model_type:
        model = JIRANO.build(
            opt=opt,
            n_classes=n_classes,
            device=device
        )

    elif model_type == "dino":
        model = nn.Sequential(
            DinoFeaturizer(20, opt),  # dim doesnt matter
            LambdaLayer(lambda p: p[0])
        )
    else:
        raise ValueError("No model: {} found".format(model_type))

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

    # cluster_probe = ClusterLookup(opt["dim"], n_classes + opt["extra_clusters"])
    # linear_probe = nn.Conv2d(opt["dim"], n_classes, (1, 1))

    return model, model.cluster_probe, model.linear_probe


def build_criterion(n_classes: int, batch_size: int, opt: dict):
    # opt = opt["loss"]
    loss_name = opt["name"].lower()
    if "dulli" in loss_name:
        loss = DulliLoss(cfg=opt)
    elif "hoi" in loss_name:
        loss = HoiLoss(n_classes=n_classes, batch_size=batch_size, cfg=opt)
    elif "vqvae" in loss_name:
        loss = VQVAELoss(n_classes=n_classes, cfg=opt)
    elif "hier" in loss_name:
        loss = HIERLoss(n_classes=n_classes, cfg=opt)
    elif "bob" in loss_name:
        loss = BobLoss(n_classes=n_classes, batch_size=batch_size, cfg=opt)
    elif "stego" in loss_name:
        loss = StegoLoss(n_classes=n_classes, cfg=opt)
    elif "jirano" in loss_name:
        loss = JiranoLoss(n_classes=n_classes, cfg=opt)
    else:
        raise ValueError(f"Unsupported loss type {loss_name}")

    return loss


def split_params_for_optimizer(model, opt):
    # opt = opt["optimizer"]
    params_large_lr = []
    params_large_lr_no_wd = []
    params_base_lr = []
    params_base_lr_no_wd = []
    for param_name, param_value in model.named_parameters():
        param_value: torch.Tensor
        if not param_value.requires_grad:
            continue
        if ("linear_probe" in param_name) or ("cluster_probe" in param_name):
            continue
        if "vq" in param_name:
            if (param_value.ndim > 1) and ("position" not in param_name):
                params_large_lr.append(param_value)
            else:
                params_large_lr_no_wd.append(param_value)
        else:  # decoder
            if (param_value.ndim > 1) and ("position" not in param_name):
                params_base_lr.append(param_value)
            else:
                params_base_lr_no_wd.append(param_value)

    encoder_weight = opt["vq_optim_weight"]
    params_for_optimizer = [
        {"params": params_base_lr},
        {"params": params_base_lr_no_wd, "weight_decay": 0.0},
        # {"params": params_small_lr, "lr": opt["lr"] * encoder_weight, "weight_decay": opt["weight_decay"] * 0.1},
        {"params": params_large_lr, "lr": opt["net"]["lr"] * encoder_weight, "weight_decay": 0.0},
        {"params": params_large_lr_no_wd, "lr": opt["net"]["lr"] * encoder_weight, "weight_decay": 0.0},
    ]
    return params_for_optimizer


def build_optimizer(main_params, cluster_params, linear_params, opt: dict, model_type: str):
    # opt = opt["optimizer"]
    model_type = model_type.lower()

    if model_type in ["dulli", "hoi", "stego", "bob", "jirano", "vqvae", "hier"]:
        net_optimizer_type = opt["net"]["name"].lower()
        if net_optimizer_type == "adam":
            net_optimizer = Adam(main_params, lr=opt["net"]["lr"])
        elif net_optimizer_type == "adamw":
            net_optimizer = AdamW(main_params, lr=opt["net"]["lr"], weight_decay=opt["net"]["weight_decay"])
        elif net_optimizer_type == "sgd":
            net_optimizer = SGD(main_params, lr=opt["net"]["lr"], momentum=0.9, weight_decay=opt["net"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {net_optimizer_type}.")

        cluster_optimizer_type = opt["cluster"]["name"].lower()
        if cluster_optimizer_type == "adam":
            cluster_optimizer = Adam(cluster_params, lr=opt["cluster"]["lr"])
        elif cluster_optimizer_type == "adamw":
            cluster_optimizer = AdamW(cluster_params, lr=opt["cluster"]["lr"],
                                      weight_decay=opt["cluster"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {cluster_optimizer_type}.")

        linear_optimizer_type = opt["cluster"]["name"].lower()
        if linear_optimizer_type == "adam":
            linear_optimizer = Adam(linear_params, lr=opt["linear"]["lr"])
        elif linear_optimizer_type == "adamw":
            linear_optimizer = AdamW(linear_params, lr=opt["linear"]["lr"], weight_decay=opt["linear"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {linear_optimizer_type}.")
    else:
        raise ValueError("No model: {} found".format(model_type))

    return net_optimizer, cluster_optimizer, linear_optimizer


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
    elif scheduler_type == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=len(loader), eta_min=0, last_epoch=-1)
    else:
        raise ValueError(f"Unsupported scheduler type {scheduler_type}.")

    return scheduler


def build_dataset(opt: dict, mode: str = "train", model_type: str = "dino", name: str = "hoi") -> ContrastiveSegDataset:
    # opt = opt["dataset"]
    data_type = opt["data_type"].lower()

    if mode == "train":
        geometric_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=opt["res"], scale=(0.8, 1.0))
        ])
        photometric_transforms = T.Compose([
            T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            T.RandomGrayscale(.2),
            T.RandomApply([T.GaussianBlur((5, 5))])
        ])

        if "hoi" == name:  # cross_loss
            return ContrastiveSegDataset(
                pytorch_data_dir=opt["data_path"],
                dataset_name=opt["data_type"],
                crop_type=opt["crop_type"],
                model_type=model_type,
                image_set=mode,
                transform=get_transform(opt["res"], False, opt["loader_crop_type"]),
                target_transform=get_transform(opt["res"], True, opt["loader_crop_type"]),
                cfg=opt,
                num_neighbors=opt["num_neighbors"],
                mask=True,
                pos_images=False,
                pos_labels=False
            )

        elif name in ["bob", "stego", "jirano", "vqvae"]:
            return ContrastiveSegDataset(
                pytorch_data_dir=opt["data_path"],
                dataset_name=opt["data_type"],
                crop_type=opt["crop_type"],
                model_type=model_type,
                image_set=mode,
                transform=get_transform(opt["res"], False, opt["loader_crop_type"]),
                target_transform=get_transform(opt["res"], True, opt["loader_crop_type"]),
                cfg=opt,
                aug_geometric_transform=geometric_transforms,
                aug_photometric_transform=photometric_transforms,
                num_neighbors=opt["num_neighbors"],
                mask=True,
                pos_images=True,
                pos_labels=True
            )

        elif name in ["hier"]:
            return ContrastiveSegDataset(
                pytorch_data_dir=opt["data_path"],
                dataset_name=opt["data_type"],
                crop_type=opt["crop_type"],
                model_type=model_type,
                image_set=mode,
                transform=get_transform(opt["res"], False, opt["loader_crop_type"]),
                target_transform=get_transform(opt["res"], True, opt["loader_crop_type"]),
                cfg=opt,
                aug_geometric_transform=geometric_transforms,
                aug_photometric_transform=photometric_transforms,
                num_neighbors=opt["num_neighbors"],
                mask=True,
                pos_images=False,
                pos_labels=False
            )
        else:
            raise ValueError("No dataset: {} found".format(name))

    elif mode == "val" or mode == "test":
        if mode == "test":
            loader_crop = "center"
        elif data_type == "voc":
            loader_crop = None
        else:
            loader_crop = "center"

        return ContrastiveSegDataset(
            pytorch_data_dir=opt["data_path"],
            dataset_name=opt["data_type"],
            crop_type=None,
            model_type=model_type,
            image_set="val",
            transform=get_transform(320, False, loader_crop),
            target_transform=get_transform(320, True, loader_crop),
            mask=True,
            cfg=opt,
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
            num_workers=opt["num_workers"],
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
            num_workers=(opt["num_workers"] + world_size - 1) // world_size,
            pin_memory=pin_memory,
            sampler=ddp_sampler,
            prefetch_factor=opt.get("prefetch_factor", 2)
        )
