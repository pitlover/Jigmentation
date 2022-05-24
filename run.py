from typing import Dict, Tuple
import argparse
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.backends import cudnn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
import wandb

from utils.common_utils import (save_checkpoint, parse, dprint, time_log, compute_param_norm,
                                freeze_bn, zero_grad_bn, RunningAverage, RunningAverageDict, Timer)
# from utils.seg_utils import comppute_error
from utils.dist_utils import all_reduce_dict
from utils.wandb_utils import set_wandb

from build import (build_model, build_criterion, build_dataset, build_dataloader, build_scheduler,
                   split_params_for_optimizer, build_optimizer)


def run(opt: dict, is_test: bool = False, is_debug: bool = False):  # noqa
    is_train = (not is_test)

    # -------------------- Distributed Setup --------------------------#
    if (opt["num_gpus"] == 0) or (not torch.cuda.is_available()):
        raise ValueError("Run requires at least 1 GPU.")

    if (opt["num_gpus"] > 1) and (not dist.is_initialized()):
        assert dist.is_available()
        dist.init_process_group(backend="nccl")  # nccl for NVIDIA GPUs
        world_size = int(dist.get_world_size())
        local_rank = int(dist.get_rank())
        torch.cuda.set_device(local_rank)
        print_fn = partial(dprint, local_rank=local_rank)  # only prints when local_rank == 0
        is_distributed = True
    else:
        world_size = 1
        local_rank = 0
        print_fn = print
        is_distributed = False

    cudnn.benchmark = True

    is_master = (local_rank == 0)
    wandb_save_dir = set_wandb(opt, local_rank, force_mode="disabled" if (is_debug or is_test) else None)
    if not wandb_save_dir:
        wandb_save_dir = opt["output_dir"]

    # ------------------------ DataLoader ------------------------------#
    if is_train:
        train_dataset = build_dataset(opt["dataset"], mode="train")
        train_loader = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)
    else:
        train_loader = None

    val_dataset = build_dataset(opt["dataset"], mode="val")
    val_loader = build_dataloader(val_dataset, opt["dataloader"], shuffle=False,
                                  batch_size=dist.get_world_size())

    test_dataset = build_dataset(opt["dataset"], mode="test")
    test_loader = build_dataloader(test_dataset, opt["dataloader"], shuffle=False,
                                  batch_size=dist.get_world_size())

    data_type = val_dataset.data_type

    # -------------------------- Define -------------------------------#
    model = build_model(opt["model"])  # CPU model
    criterion = build_criterion(opt, val_dataset)  # CPU criterion

    device = torch.device("cuda", local_rank)  # "cuda:0" for single GPU
    model = model.to(device)
    criterion = criterion.to(device)

    # ----------------------- Distributed ----------------------------#
    if is_distributed:
        assert dist.is_available() and dist.is_initialized()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
        criterion = criterion.to(device)
        model_m = model.module  # unwrap DDP
    else:
        model_m = model

    print_fn("Model:")
    print_fn(model_m.decoder)

    # ------------------- Optimizer & Scheduler -----------------------#
    if is_train:
        params_for_optimizer = split_params_for_optimizer(model_m, opt["optimizer"])
        optimizer = build_optimizer(params_for_optimizer, opt["optimizer"])
    else:
        optimizer = None

    # --------------------------- Load --------------------------------#
    if opt['checkpoint']:  # resume case
        checkpoint = torch.load(opt['checkpoint'], map_location=device)
        try:
            model_m.load_state_dict(checkpoint['model'], strict=True)
        except KeyError:
            model_m.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = max(checkpoint.get('epoch', 0), 0)
        current_iter = max(checkpoint.get('iter', 0), 0)
        best_metric = min(checkpoint.get("best", 1e7), 1e7)  # abs rel, lower better
        best_epoch = max(checkpoint.get('best_epoch', 0), 0)
        best_iter = max(checkpoint.get('best_iter', 0), 0)
        best2_metric = min(checkpoint.get("best2", 1e7), 1e7)  # rmse, lower better
        best2_epoch = max(checkpoint.get('best2_epoch', 0), 0)
        best2_iter = max(checkpoint.get('best2_iter', 0), 0)
        print_fn(f"Checkpoint loaded: epoch {start_epoch}, iters {current_iter}, best metric: {best_metric:.6f}")
    else:
        start_epoch, current_iter = 0, 0
        best_metric, best_epoch, best_iter = 1e7, 0, 0
        best2_metric, best2_epoch, best2_iter = 1e7, 0, 0
        if is_test:
            print_fn("Warning: testing but checkpoint is not loaded.")

    # ------------------- Scheduler -----------------------#
    if is_train:
        num_accum = max(opt["train"]["num_accum"], 1)
        scheduler = build_scheduler(opt, optimizer, train_loader, start_epoch)
        if start_epoch != 0:
            scheduler.step(start_epoch + 1)
    else:
        num_accum = 1
        scheduler = None

    timer = Timer()
    # --------------------------- Test --------------------------------#
    if is_test:
        _ = timer.update()
        test_loss, test_metrics = evaluate(
            model, val_loader, criterion, device=device, opt=opt["eval"], data_type=data_type)
        test_time = timer.update()

        s = time_log()
        s += f"[TEST] ---------------------------------------------\n"
        s += f"[TEST] epoch: {start_epoch}, iters: {current_iter}\n"
        s += f"[TEST] loss: {test_loss:.6f}\n"
        for metric_k, metric_v in test_metrics.items():
            s += f"[TEST] {metric_k} : {metric_v:.6f}\n"
        s += f"[TEST] time: {test_time:.3f}"
        print_fn(s)

        print_fn(f"-------- Test Finished --------")
        return

    # --------------------------- Train --------------------------------#
    assert is_train
    max_epoch = opt["train"]["epoch"]
    print_freq = opt["train"]["print_freq"]
    valid_freq = opt["train"]["valid_freq"]
    grad_norm = opt["train"]["grad_norm"]
    freeze_encoder_bn = opt["train"].get("freeze_encoder_bn", False)
    freeze_all_bn = opt["train"].get("freeze_all_bn", -1)  # epoch

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.momentum /= num_accum

    best_valid_metrics = dict(a1=0.0, a2=0.0, a3=0.0, abs_rel=9999.9, sq_rel=9999.9, rmse=9999.9, rmse_log=9999.9,
                              silog=9999.0, log_10=9999.9)
    best2_valid_metrics = dict(a1=0.0, a2=0.0, a3=0.0, abs_rel=9999.9, sq_rel=9999.9, rmse=9999.9, rmse_log=9999.9,
                               silog=9999.0, log_10=9999.9)
    train_stats = RunningAverage()  # track training loss per epoch

    metric1 = "abs_rel" if (not data_type == "ONLINE") else "silog"
    metric2 = "rmse"

    for current_epoch in range(start_epoch, max_epoch):
        print_fn(f"-------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------")

        g_norm = torch.zeros(1, dtype=torch.float32, device=device)  # placeholder
        if is_distributed:
            train_loader.sampler.set_epoch(current_epoch)  # noqa, necessary for DistributedSampler to be shuffled.

        torch.cuda.empty_cache()
        model.train()
        train_stats.reset()
        _ = timer.update()
        for i, data in enumerate(train_loader):
            image: torch.Tensor = data['image'].to(device, non_blocking=True)
            gt_depth: torch.Tensor = data['depth'].to(device, non_blocking=True)
            data_time = timer.update()

            if freeze_encoder_bn:
                freeze_bn(model_m.encoder)
            if 0 < freeze_all_bn <= current_epoch:
                freeze_bn(model)

            batch_size = image.shape[0]
            if i % num_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            model_input = (image, gt_depth)
            model_output = model(image)
            loss, loss_dict = criterion(model_input, model_output)
            forward_time = timer.update()

            loss = loss / num_accum
            loss.backward()

            if i % num_accum == (num_accum - 1):
                if freeze_encoder_bn:
                    zero_grad_bn(model_m.encoder)
                if 0 < freeze_all_bn <= current_epoch:
                    zero_grad_bn(model)

                g_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
                scheduler.step()
                current_iter += 1

            backward_time = timer.update()

            loss_dict = all_reduce_dict(loss_dict, op="mean")
            train_stats.append(loss_dict["loss"])

            if i % print_freq == 0:
                lrs = [int(pg["lr"] * 1e8) / 1e8 for pg in optimizer.param_groups]
                p_norm = compute_param_norm(model.parameters())
                s = time_log()
                s += f"epoch: {current_epoch}, iters: {current_iter} " \
                     f"({i} / {len(train_loader)} -th batch of loader)\n"
                s += f"loss(now/avg): {loss_dict['loss']:.6f}/{train_stats.avg:.6f}\n"
                if len(loss_dict) > 2:  # more than two loss:
                    for loss_k, loss_v in loss_dict.items():
                        if loss_k != "loss":
                            s += f"-- {loss_k}(now): {loss_v:.6f}\n"
                s += f"time(data/fwd/bwd): {data_time:.3f}/{forward_time:.3f}/{backward_time:.3f}\n"
                s += f"LR: {lrs}\n"
                s += f"batch_size x world_size x num_accum: " \
                     f"{batch_size} x {world_size} x {num_accum} = {batch_size * world_size * num_accum}\n"
                s += f"norm(param/grad): {p_norm.item():.3f}/{g_norm.item():.3f}"
                print_fn(s)

                if is_master:
                    wandb.log({
                        "epoch": current_epoch,
                        "iters": current_iter,
                        "train_loss": loss_dict['loss'],
                        "lr": optimizer.param_groups[0]['lr'],
                        "param_norm": p_norm.item(),
                        "grad_norm": g_norm.item(),
                    })

            # --------------------------- Valid --------------------------------#
            if ((i + 1) % valid_freq == 0) or ((i + 1) == len(train_loader)):
                _ = timer.update()
                valid_loss, valid_metrics = evaluate(
                    model, val_loader, criterion, device=device, opt=opt["eval"], data_type=data_type)
                valid_time = timer.update()

                s = time_log()
                s += f"[VAL] -------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------\n"
                s += f"[VAL] epoch: {current_epoch}, iters: {current_iter}\n"
                s += f"[VAL] loss: {valid_loss:.6f}\n"

                # update based on AbsRel
                prev_best_metric = best_metric
                if best_metric >= valid_metrics[metric1]:
                    best_metric = valid_metrics[metric1]
                    best_epoch = current_epoch
                    best_iter = current_iter
                    s += f"[VAL] -------- updated ({metric1})! {prev_best_metric:.6f} -> {best_metric:.6f}\n"
                    if is_master:
                        save_checkpoint(
                            "best", model, optimizer,
                            current_epoch, current_iter, best_metric, wandb_save_dir, model_only=True)
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {best_valid_metrics[metric_k]:.6f} -> {metric_v:.6f}\n"
                    best_valid_metrics.update(valid_metrics)
                else:
                    s += f"[VAL] -------- not updated ({metric1})." \
                         f" (now) {valid_metrics[metric1]:.6f} vs (best) {prev_best_metric:.6f}\n"
                    s += f"[VAL] previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {metric_v:.6f} vs {best_valid_metrics[metric_k]:.6f}\n"

                # update based on RMSE
                prev_best2_metric = best2_metric
                if best2_metric >= valid_metrics[metric2]:
                    best2_metric = valid_metrics[metric2]
                    best2_epoch = current_epoch
                    best2_iter = current_iter
                    s += f"[VAL] -------- updated ({metric2})! {prev_best2_metric:.6f} -> {best2_metric:.6f}\n"
                    if is_master:
                        save_checkpoint(
                            "best2", model, optimizer,
                            current_epoch, current_iter, best2_metric, wandb_save_dir, model_only=True)
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {best2_valid_metrics[metric_k]:.6f} -> {metric_v:.6f}\n"
                    best2_valid_metrics.update(valid_metrics)
                else:
                    s += f"[VAL] -------- not updated ({metric2}). " \
                         f"(now) {valid_metrics[metric2]:.6f} vs (best) {prev_best2_metric:.6f}\n"
                    s += f"[VAL] previous best was at {best2_epoch} epoch, {best2_iter} iters\n"
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {metric_v:.6f} vs {best2_valid_metrics[metric_k]:.6f}\n"

                s += f"[VAL] time: {valid_time:.3f}"
                print_fn(s)

                if is_master:
                    valid_metrics.update({"iters": current_iter, "valid_loss": valid_loss})
                    wandb.log(valid_metrics)

                torch.cuda.empty_cache()
                model.train()
                train_stats.reset()

            _ = timer.update()

        # --------------------------- Save --------------------------------#
        if is_master:
            save_checkpoint("latest", model, optimizer,
                            current_epoch, current_iter, best_metric, wandb_save_dir,
                            best_epoch=best_epoch, best_iter=best_iter, model_only=False)

    # --------------------------- Evaluate with Best --------------------------------#

    # best_checkpoint = torch.load(f"{wandb_save_dir}/best.pth", map_location=device)
    # model_m.load_state_dict(best_checkpoint, strict=True)
    # best_loss, best_metrics = evaluate(
    #     model, val_loader, criterion, device=device, opt=opt["eval"], data_type=data_type)
    #
    # s = time_log()
    # s += f"[BEST] ---------------------------------------------\n"
    # s += f"[BEST] epoch: {best_epoch}, iters: {best_iter}\n"
    # s += f"[BEST] loss: {best_loss:.6f}\n"
    # for metric_k, metric_v in best_metrics.items():
    #     s += f"[BEST] {metric_k} : {metric_v:.6f}\n"
    # print_fn(s)

    if is_master:
        wandb.finish()
    print_fn(f"-------- Train Finished --------")


def evaluate(model: nn.Module,
             eval_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             opt: Dict,
             data_type: str) -> Tuple[float, Dict[str, float]]:  # noqa
    # opt = opt["eval"]
    if not (opt["garg_crop"] != opt["eigen_crop"]):
        raise ValueError("Either garg_crop or eigen_crop should be True")
    eval_mask = None

    min_depth = opt["min_depth_eval"]
    max_depth = opt["max_depth_eval"]
    flip_eval = opt.get("flip_eval", False)

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        eval_stats = RunningAverage()  # loss
        eval_metrics = RunningAverageDict()  # metric

        for i, data in enumerate(eval_loader):
            image = data['image'].to(device, non_blocking=True)
            gt_depth = data['depth'].to(device, non_blocking=True)

            model_output = model(image)
            pred_depth = model_output[0]
            pred_depth.clamp_(min_depth, max_depth)  # inplace
            pred_depth.nan_to_num_(nan=min_depth, posinf=max_depth, neginf=0)  # inplace

            # as a common practice, calculate metric can use the average of left-right
            if flip_eval:
                image_flip = torch.flip(image, [3])
                pred_depth_flip = model(image_flip)[0]
                pred_depth_flip = torch.flip(pred_depth_flip, [3])
                pred_depth_flip.clamp_(min_depth, max_depth)
                pred_depth_flip.nan_to_num_(nan=min_depth, posinf=max_depth, neginf=0)
                pred_depth = 0.5 * (pred_depth + pred_depth_flip)

            model_input = (image, gt_depth)
            loss, loss_dict = criterion(model_input, model_output)
            loss_dict = all_reduce_dict(loss_dict, op="mean")
            eval_stats.append(loss_dict["loss"])

            if eval_mask is None:
                eval_mask = cal_eval_mask(opt, gt_depth, data_type=data_type)  # (h, w)

            # ---- metric ---- #
            pred = F.interpolate(pred_depth, gt_depth.shape[-2:], mode='bilinear', align_corners=True)  # (b, 1, h, w)

            pred = pred.squeeze(1).cpu().numpy()  # (b, h, w)
            gt = gt_depth.squeeze(1).cpu().numpy()  # (b, h, w)
            valid_mask = np.logical_and(gt > min_depth, gt < max_depth)
            valid_mask = np.logical_and(valid_mask, eval_mask)

            batch_size = image.shape[0]
            for j in range(batch_size):
                eval_metrics.update(compute_errors(gt[j][valid_mask[j]], pred[j][valid_mask[j]]))

        eval_metrics = eval_metrics.get_value()  # now this is dict
        eval_metrics = all_reduce_dict(eval_metrics, op="mean")

        return eval_stats.avg, eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="Path to option JSON file.")
    parser.add_argument("--test", action="store_true", help="Test mode, no WandB, highest priority.")
    parser.add_argument("--debug", action="store_true", help="Debug mode, no WandB, second highest priority.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    parser.add_argument("--data_path", type=str, default=None, help="Data path override")

    parser_args = parser.parse_args()
    parser_opt = parse(parser_args.opt)
    if parser_args.checkpoint is not None:
        parser_opt["checkpoint"] = parser_args.checkpoint
    if parser_args.data_path is not None:
        parser_opt["dataset"]["data_path"] = parser_args.data_path

    run(parser_opt, is_test=parser_args.test, is_debug=parser_args.debug)
