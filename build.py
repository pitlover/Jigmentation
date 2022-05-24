import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset import SegDataset
from torch.utils.data.dataloader import DataLoader
from loss import *
from model import SeMask

def build_model(opt: dict):
    # opt = opt["model"]
    model_type = opt["name"].lower()

    if model_type == "semask":
        model = SeMask.build(
            opt
        )
    else:
        raise ValueError(f"Unsupported model type {model_type}.")

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

def build_criterion(opt: dict, dataset):
    # opt = opt BE CAREFUL!
    loss_type = opt["model"]['name'].lower()
    opt_loss = opt["loss"]



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
    elif scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.001,
            total_iters=(len(loader) // opt["train"]["num_accum"]) * (opt["train"]["epoch"] + 1),
            last_epoch=start_epoch - 1
        )
    else:
        raise ValueError(f"Unsupported scheduler type {scheduler_type}.")

    return scheduler


def build_dataset(opt: dict, mode: str = "train"):
    # opt = opt["dataset"]
    return SegDataset(
        data_path=opt["data_path"],
        data_type=opt["data_type"],
        mode=mode,
        img_size=opt["input_size"],
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
