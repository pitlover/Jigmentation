from typing import Dict, Tuple
import os
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.backends import cudnn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
import wandb
from tqdm import tqdm
from visualize import visualization
from utils.common_utils import (save_checkpoint, parse, dprint, time_log, compute_param_norm,
                                freeze_bn, zero_grad_bn, RunningAverage, Timer)
from utils.dist_utils import all_reduce_dict
from utils.wandb_utils import set_wandb
from utils.seg_utils import UnsupervisedMetrics, batched_crf, get_metrics
from build import (build_model, build_criterion, build_dataset, build_dataloader, build_optimizer, build_scheduler,
                   split_params_for_optimizer)
from pytorch_lightning.utilities.seed import seed_everything


def run(opt: dict, is_test: bool = False, is_debug: bool = False):  # noqa
    is_train = (not is_test)
    seed_everything(seed=10)

    # -------------------- Folder Setup (Task-Specific) --------------------------#
    prefix = "{}/{}_{}".format(opt["output_dir"], opt["dataset"]["data_type"], opt["wandb"]["name"])
    opt["full_name"] = prefix

    # -------------------- Distributed Setup --------------------------#
    if (opt["num_gpus"] == 0) or (not torch.cuda.is_available()):
        raise ValueError("Run requir es at least 1 GPU.")

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
        wandb_save_dir = os.path.join(opt["output_dir"], opt["wandb"]["name"])
    if is_test:
        wandb_save_dir = "/".join(opt["checkpoint"].split("/")[:-1])

    # ------------------------ DataLoader ------------------------------#
    if is_train:
        train_dataset = build_dataset(opt["dataset"], mode="train", model_type=opt["model"]["name"],
                                      name=opt["model"]["name"].lower())
        train_loader = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)
    else:
        train_loader = None

    val_dataset = build_dataset(opt["dataset"], mode="val", model_type=opt["model"]["pretrained"]["model_type"])
    val_loader = build_dataloader(val_dataset, opt["dataloader"], shuffle=False,
                                  batch_size=world_size)

    test_dataset = build_dataset(opt["dataset"], mode="test")
    test_loader = build_dataloader(test_dataset, opt["dataloader"], shuffle=False,
                                   batch_size=1)

    # -------------------------- Define -------------------------------#
    device = torch.device("cuda", local_rank)  # "cuda:0" for single GPU
    model, cluster_model, linear_model = build_model(opt=opt["model"],
                                                     n_classes=val_dataset.n_classes,
                                                     device=device)  # CPU model

    criterion = build_criterion(n_classes=val_dataset.n_classes,
                                batch_size=opt["dataloader"]["batch_size"],
                                opt=opt["loss"])  # CPU criterion

    model = model.to(device)
    cluster_model = cluster_model.to(device)
    linear_model = linear_model.to(device)
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
        model = model
        model_m = model

    print_fn("Model:")
    print_fn(model_m)

    # ------------------- Optimizer  -----------------------#
    if is_train:
        params_for_optimizer = split_params_for_optimizer(model_m, opt["optimizer"])
        # paramas_for_optimizer = model_m.parameters()
        optimizer, cluster_optimizer, linear_optimizer = build_optimizer(
            main_params=params_for_optimizer,
            cluster_params=cluster_model.parameters(),
            linear_params=linear_model.parameters(),
            opt=opt["optimizer"],
            model_type=opt["wandb"]["name"]
        )

    else:
        optimizer = None

    # --------------------------- Load --------------------------------#
    if opt['checkpoint']:  # resume case
        checkpoint = torch.load(opt['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['net_model_state_dict'], strict=True)
        cluster_model.load_state_dict(checkpoint['cluster_model_state_dict'], strict=True)
        linear_model.load_state_dict(checkpoint['linear_model_state_dict'], strict=True)

        if is_train:
            optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
            cluster_optimizer.load_state_dict(checkpoint['cluster_optimizer_state_dict'])
            linear_optimizer.load_state_dict(checkpoint['linear_optimizer_state_dict'])

        start_epoch = max(checkpoint.get('epoch', 0), 0)
        current_iter = max(checkpoint.get('iter', 0), 0)
        best_metric = max(checkpoint.get("best", 0), 0)  # abs rel, lower better
        best_epoch = max(checkpoint.get('best_epoch', 0), 0)
        best_iter = max(checkpoint.get('best_iter', 0), 0)
        print_fn(f"Checkpoint loaded: epoch {start_epoch}, iters {current_iter}, best metric: {best_metric:.6f}")
    else:
        start_epoch, current_iter = 0, 0
        best_metric, best_epoch, best_iter = 0, 0, 0
        if is_test:
            print_fn("Warning: testing but checkpoint is not loaded.")

    # ------------------- Scheduler -----------------------#
    if is_train:
        num_accum = opt["train"]["num_accum"]
        # scheduler = build_scheduler(opt, optimizer, train_loader, start_epoch)
        # if start_epoch != 0:
        #     scheduler.step(start_epoch + 1)
    else:
        num_accum = 1
        scheduler = None

    timer = Timer()
    # --------------------------- Test --------------------------------#
    if is_test:
        _ = timer.update()
        test_loss, test_metrics = evaluate(  # TODO check flip version
            model, cluster_model, linear_model,
            test_loader, device=device, opt=opt["eval"],
            n_classes=val_dataset.n_classes, data_type=opt["dataset"]["data_type"],
            saved_dir=wandb_save_dir, is_crf=opt["eval"]["is_crf"], current_iter=best_iter,
            out_type=opt["model"]["name"].lower())
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
    freeze_encoder_bn = opt["train"]["freeze_encoder_bn"]
    freeze_all_bn = opt["train"]["freeze_all_bn"]  # epoch

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.momentum /= num_accum

    best_valid_metrics = dict(Cluster_mIoU=0, Cluster_Accuracy=0, Linear_mIoU=0, Linear_Accuracy=0)
    train_stats = RunningAverage()  # track training loss per epoch

    for current_epoch in range(start_epoch, max_epoch):
        print_fn(f"-------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------")

        g_norm = torch.zeros(1, dtype=torch.float32, device=device)  # placeholder
        if is_distributed:
            train_loader.sampler.set_epoch(current_epoch)  # noqa, necessary for DistributedSampler to be shuffled.

        model.train()

        train_stats.reset()
        _ = timer.update()

        for i, data in enumerate(train_loader):
            img: torch.Tensor = data['img'].to(device, non_blocking=True)
            out_type = opt["model"]["name"].lower()

            if opt["loss"]["corr_weight"] > 0.0:
                img_pos: torch.Tensor = data['img_pos'].to(device, non_blocking=True)
            label: torch.Tensor = data['label'].to(device, non_blocking=True)  # (b, h, w)

            data_time = timer.update()

            if freeze_encoder_bn:
                freeze_bn(model_m.model)
            if 0 < freeze_all_bn <= current_epoch:
                freeze_bn(model)

            batch_size = img.shape[0]
            if i % num_accum == 0:
                optimizer.zero_grad(set_to_none=True)
                cluster_optimizer.zero_grad(set_to_none=True)
                linear_optimizer.zero_grad(set_to_none=True)

            model_input = (img, label)
            model_output = model(img, cur_iter=current_iter,
                                 local_rank=local_rank)  # head : (b, 70, 28, 28) quantized : (b, 70, 28, 28)
            #  x, qx, assignment, distance, recon, feat

            # TODO check pos->raw or qutized?
            model_pos_output = None
            if opt["loss"]["corr_weight"] > 0.0:
                if "hoi" in opt["model"]["name"].lower() or "jirano" in opt["model"]["name"].lower():
                    model_pos_output = model(img_pos, cur_iter=current_iter, is_pos=True)
                else:
                    model_pos_output = model(img_pos, cur_iter=current_iter)

            output_type = opt["eval"]["output"].lower()

            if "hoi" in out_type:
                if output_type == "vq":
                    out = model_output[1][0]
                elif output_type == "head":
                    out = model_output[0][0]
                else:
                    raise ValueError(f"Unsupported loss type {output_type}")
            elif out_type in ["bob", "jirano"]:
                if output_type == "vq":
                    out = model_output[1]
                elif output_type == "head":
                    out = model_output[0]
                else:
                    raise ValueError(f"Unsupported loss type {output_type}")
            elif "stego" in out_type:
                out = model_output[1]
            else:
                raise ValueError(f"Unsupported loss type {out_type}")

            detached_code = torch.clone(out.detach())  # (b, 128, 56, 56)
            linear_output = linear_model(detached_code)
            cluster_output = cluster_model(detached_code, None)

            loss, loss_dict, vq_dict, corr_dict = criterion(model_input=model_input,
                                                            model_output=model_output,
                                                            model_pos_output=model_pos_output if model_pos_output is not None else None,
                                                            linear_output=linear_output,
                                                            cluster_output=cluster_output)

            forward_time = timer.update()

            loss = loss / num_accum
            loss.backward()

            if i % num_accum == (num_accum - 1):
                if freeze_encoder_bn:
                    zero_grad_bn(model_m)
                if 0 < freeze_all_bn <= current_epoch:
                    zero_grad_bn(model)

                g_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
                cluster_optimizer.step()
                linear_optimizer.step()
                # scheduler.step()
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
                            if loss_k == "corr":
                                for k, v in corr_dict.items():
                                    s += f"  -- {k}(now): {v:.6f}\n"
                            elif loss_k == "vq":
                                for k, v in vq_dict.items():
                                    s += f"  -- {k}(now): {v:.6f}\n"
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
                        "STEGO_loss": loss_dict["corr"] if opt["loss"].get("corr_weight", 0) > 0 else 0,
                        'vq_loss': loss_dict["vq"] if opt["loss"].get("vq_weight", 0) > 0 else 0,
                        'recon_loss': loss_dict["recon"] if opt["loss"].get("recon_weight", 0) > 0 else 0,
                        'cross_loss': loss_dict["cross"] if opt["loss"].get("cross_weight", 0) > 0 else 0,
                        "net_lr": optimizer.param_groups[0]['lr'],
                        "param_norm": p_norm.item(),
                        "grad_norm": g_norm.item(),
                    })
            # --------------------------- Valid --------------------------------#
            if ((i + 1) % valid_freq == 0) or ((i + 1) == len(train_loader)):
                _ = timer.update()
                valid_loss, valid_metrics = evaluate(
                    model, cluster_model, linear_model, val_loader, device=device,
                    opt=opt["eval"],
                    n_classes=val_dataset.n_classes,
                    out_type=opt["model"]["name"].lower())
                valid_time = timer.update()

                s = time_log()
                s += f"[VAL] -------- [{current_epoch}/{max_epoch} (iters: {current_iter})]--------\n"
                s += f"[VAL] epoch: {current_epoch}, iters: {current_iter}\n"
                s += f"[VAL] loss: {valid_loss:.6f}\n"

                # update based on mIOU
                metric = "Cluster_mIoU"

                prev_best_metric = best_metric

                if best_metric <= valid_metrics[metric]:
                    best_metric = valid_metrics[metric]
                    best_epoch = current_epoch
                    best_iter = current_iter
                    s += f"[VAL] -------- updated ({metric})! {prev_best_metric:.6f} -> {best_metric:.6f}\n"
                    if is_master:
                        save_checkpoint(
                            "best", model, optimizer,
                            cluster_model, cluster_optimizer,
                            linear_model, linear_optimizer,
                            current_epoch, current_iter, best_metric, wandb_save_dir, model_only=True)
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {best_valid_metrics[metric_k]:.6f} -> {metric_v:.6f}\n"
                    best_valid_metrics.update(valid_metrics)
                else:
                    s += f"[VAL] -------- not updated ({metric})." \
                         f" (now) {valid_metrics[metric]:.6f} vs (best) {prev_best_metric:.6f}\n"
                    s += f"[VAL] previous best was at {best_epoch} epoch, {best_iter} iters\n"
                    for metric_k, metric_v in valid_metrics.items():
                        s += f"[VAL] {metric_k} : {metric_v:.6f} vs {best_valid_metrics[metric_k]:.6f}\n"

                s += f"[VAL] time: {valid_time:.3f}"
                print_fn(s)

                if is_master:
                    valid_metrics.update({"iters": current_iter, "valid_loss": valid_loss})
                    wandb.log(valid_metrics)

                model.train()
                linear_model.train()
                cluster_model.train()
                train_stats.reset()

            _ = timer.update()

        # --------------------------- Save --------------------------------#
        if is_master:
            save_checkpoint("latest", model, optimizer, cluster_model, cluster_optimizer, linear_model,
                            linear_optimizer,
                            current_epoch, current_iter, best_metric, wandb_save_dir,
                            best_epoch=best_epoch, best_iter=best_iter, model_only=False)

    # --------------------------- Evaluate with Best --------------------------------#

    best_checkpoint = torch.load(f"{wandb_save_dir}/best.pth", map_location=device)
    model.load_state_dict(best_checkpoint['net_model_state_dict'], strict=True)
    cluster_model.load_state_dict(best_checkpoint['cluster_model_state_dict'], strict=True)
    linear_model.load_state_dict(best_checkpoint['linear_model_state_dict'], strict=True)

    best_loss, best_metrics = evaluate(
        model, cluster_model, linear_model, val_loader, device=device, opt=opt["eval"],
        n_classes=train_dataset.n_classes, is_crf=opt["eval"]["is_crf"], current_iter=best_iter,
        out_type=opt["model"]["name"].lower())

    s = time_log()
    s += f"[BEST] ---------------------------------------------\n"
    s += f"[BEST] ------------------- CRF ---------------------\n"
    s += f"[BEST] epoch: {best_epoch}, iters: {best_iter}\n"
    s += f"[BEST] loss: {best_loss:.6f}\n"
    for metric_k, metric_v in best_metrics.items():
        s += f"[BEST] {metric_k} : {metric_v:.6f}\n"
    print_fn(s)

    # --------------------------- Evaluate with Latest --------------------------------#
    latest_checkpoint = torch.load(f"{wandb_save_dir}/latest.pth", map_location=device)
    model.load_state_dict(latest_checkpoint['net_model_state_dict'], strict=True)
    cluster_model.load_state_dict(latest_checkpoint['cluster_model_state_dict'], strict=True)
    linear_model.load_state_dict(latest_checkpoint['linear_model_state_dict'], strict=True)
    best_loss, best_metrics = evaluate(
        model, cluster_model, linear_model, val_loader, device=device, opt=opt["eval"],
        n_classes=train_dataset.n_classes, current_iter=current_iter, out_type=opt["model"]["name"].lower())

    s = time_log()
    s += f"[LATEST] ---------------------------------------------\n"
    s += f"[LATEST] epoch: {best_epoch}, iters: {best_iter}\n"
    s += f"[LATEST] loss: {best_loss:.6f}\n"
    for metric_k, metric_v in best_metrics.items():
        s += f"[LATEST] {metric_k} : {metric_v:.6f}\n"
    print_fn(s)

    if is_master:
        wandb.finish()
    print_fn(f"-------- Train Finished --------")


def evaluate(model: nn.Module,
             cluster_model: nn.Module,
             linear_model: nn.Module,
             eval_loader: DataLoader,
             device: torch.device,
             opt: Dict,
             n_classes: int,
             is_crf: bool = False,
             data_type: str = "",
             saved_dir: str = "",
             current_iter: int = 0,
             out_type: str = "hoi"
             ) -> Tuple[float, Dict[str, float]]:  # noqa
    # opt = opt["eval"]

    model.eval()
    output_type = opt["output"].lower()

    cluster_metrics = UnsupervisedMetrics(
        "Cluster_", n_classes, opt["extra_clusters"], True)
    linear_metrics = UnsupervisedMetrics(
        "Linear_", n_classes, 0, False)

    with torch.no_grad():
        eval_stats = RunningAverage()  # loss
        from collections import defaultdict
        saved_data = defaultdict(list)  # for visualization

        for i, data in enumerate(tqdm(eval_loader)):
            img: torch.Tensor = data['img'].to(device, non_blocking=True)
            label: torch.Tensor = data['label'].to(device, non_blocking=True)
            img_path: str = data['img_path']

            model_output = model(img, cur_iter=current_iter)  # code, quantized, ass, distance

            if out_type == "stego":
                out = F.interpolate(model_output[1], label.shape[-2:], mode='bilinear', align_corners=False)
            else:
                if output_type == "vq":
                    out = F.interpolate(model_output[1], label.shape[-2:], mode='bilinear', align_corners=False)
                elif output_type == "head":
                    out = F.interpolate(model_output[0], label.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    raise ValueError(f"Unsupported loss type {output_type}")

            if is_crf:
                linear_preds = torch.log_softmax(linear_model(out), dim=1)
                cluster_loss, cluster_preds = cluster_model(out, 2, log_probs=True)

                linear_preds = batched_crf(img, linear_preds).argmax(1).cuda()
                cluster_preds = batched_crf(img, cluster_preds).argmax(1).cuda()

            else:
                linear_preds = linear_model(out).argmax(1)
                cluster_loss, cluster_preds = cluster_model(out, None)
                cluster_preds = cluster_preds.argmax(1)

            linear_metrics.update(linear_preds, label)
            cluster_metrics.update(cluster_preds, label)
            eval_stats.append(cluster_loss)

            if opt["is_visualize"]:
                saved_data["img_path"].append("".join(img_path))
                saved_data["cluster_preds"].append(cluster_preds.cpu().squeeze(0))
                saved_data["linear_preds1"].append(linear_preds.cpu().squeeze(0))
                saved_data["label"].append(label.cpu().squeeze(0))

        eval_metrics = get_metrics(cluster_metrics, linear_metrics)

        if opt["is_visualize"]:
            visualization(saved_dir, data_type, saved_data, cluster_metrics)

        return eval_stats.avg, eval_metrics


def main():
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


if __name__ == "__main__":
    main()
