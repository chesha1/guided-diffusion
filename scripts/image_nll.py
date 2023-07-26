"""
Approximate the bits/dimension for an image model.
这段代码用于对图像模型进行比特/维度的近似评估
improved-diffusion 中本文件和 guided-diffusion 中本文件一样
"""

import argparse
import os

import numpy as np
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    # 创建一个空列表all_bpd用于存储每个样本的比特/维度指标
    all_bpd = []

    # 创建一个字典all_metrics，其中包含了三个术语指标的空列表，分别是"vb"、"mse"和"xstart_mse
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}

    # 初始化变量num_complete为0，用于追踪已完成的样本数量
    num_complete = 0
    while num_complete < num_samples:

        # 从数据加载器中获取下一个批次的数据和模型参数
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # 使用扩散对象的calc_bpd_loop方法计算每个批次的比特/维度指标
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # 对每个指标的术语列表进行归约操作，计算每个术语的平均值，并将其添加到all_metrics字典中
        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        # 计算总的比特 / 维度指标并归约
        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)

        # 将总的比特 / 维度指标添加到all_bpd列表中
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    # 如果当前进程的排名为0，则对每个指标的术语列表进行平均，并将结果保存到磁盘上的文件中
    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
