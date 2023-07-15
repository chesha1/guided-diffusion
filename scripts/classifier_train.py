"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    # 如果args.noised为True，代码将根据指定的调度采样器名称创建一个调度采样器对象
    # 并将其赋值给变量schedule_sampler
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    # 如果存在指定的检查点文件（args.resume_checkpoint）
    # 则将恢复训练步骤数（resume_step）从文件名中解析出来，并加载该检查点文件中的模型参数到当前模型中
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        # 在训练过程中执行前向传播、反向传播和日志记录

        # 从数据加载器（data_loader）中获取一个批次的数据和额外的信息。
        # 从额外的信息中获取标签，并将其移动到适当的设备上（dist_util.dev()）。
        # 将批次数据移动到适当的设备上（dist_util.dev()）
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())

        # Noisy images
        # 如果命令行参数args.noised为True，则执行以下操作：
        if args.noised:
            # 调用schedule_sampler.sample方法
            # 传入批次数据的数量（batch.shape[0]）和设备（dist_util.dev()）作为参数
            # 该方法会生成一组时间步骤（t）和噪声标记（noise），用于后续的噪声采样过程
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            # 将批次数据（batch）和生成的时间步骤（t）传递给扩散模型（diffusion）的q_sample方法，生成噪声图像
            batch = diffusion.q_sample(batch, t)
        else:
            # 将t初始化为全零的张量
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        # 使用split_microbatches函数将批次数据、标签和t划分为子批次，以便适应微批次（microbatch）的训练
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            # 使用模型对子批次数据进行前向传播，得到预测结果（logits）
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            # 创建一个字典（losses），存储损失和准确率等信息
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )

            # 调用log_loss_dict函数，记录损失字典中的信息到日志中
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()

            # 如果损失需要梯度计算，则执行以下操作
            if loss.requires_grad:
                # 如果是子批次的第一个子批次，则调用mp_trainer.zero_grad()
                # 方法将模型参数的梯度置零
                if i == 0:
                    mp_trainer.zero_grad()
                # 执行反向传播，计算梯度
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    # 使用range函数迭代args.iterations - resume_step次
    # 其中args.iterations是总的训练迭代次数，resume_step是之前恢复的训练步骤数
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )

        # 如果命令行参数args.anneal_lr为True，调用set_annealed_lr函数来动态调整学习率
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)

        # 调用forward_backward_log函数执行前向传播、反向传播和日志记录操作，对训练数据进行训练
        forward_backward_log(data)
        # 调用mp_trainer.optimize方法执行优化步骤，更新模型的参数
        mp_trainer.optimize(opt)

        # 如果验证数据集（val_data）不为None
        # 并且当前步骤（step）符合评估间隔（args.eval_interval）的要求，则进行验证
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
