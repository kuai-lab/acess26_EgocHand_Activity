import os
import sys
import torch
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from datasets import collate
from models.gt2d_gibson import TemporalNet
from netscripts import epochpass, reloadmodel, get_dataset_rgb_mp
from libyana.exputils.argutils import save_args
from libyana.modelutils import modelio, freeze
from libyana.randomutils import setseeds
from netscripts.get_dataset_rgb_mp import DataLoaderX

plt.switch_backend("agg")

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def collate_fn(seq, extend_queries=[]):
    return collate.seq_extend_flatten_collate(seq, extend_queries)

def main_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    setseeds.set_all_seeds(args.manual_seed)

    experiment_tag = args.experiment_tag
    exp_id = f"{args.cache_folder}{experiment_tag}/"
    if rank == 0:
        save_args(args, exp_id, "opt")
        board_writer = SummaryWriter(log_dir=exp_id)
    else:
        board_writer = None

    train_dataset, _ = get_dataset_rgb_mp.get_dataset_htt(
        args.train_dataset,
        dataset_folder=args.dataset_folder,
        split=args.train_split,
        no_augm=False,
        scale_jittering=args.scale_jittering,
        center_jittering=args.center_jittering,
        ntokens_pose=args.ntokens_pose,
        ntokens_action=args.ntokens_action,
        spacing=args.spacing,
        is_shifting_window=False,
        split_type="actions"
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    dataset_info = train_dataset.pose_dataset

    model = TemporalNet(
        dataset_info=dataset_info,
        is_single_hand=args.train_dataset != "h2ohands",
        transformer_num_encoder_layers_action=args.enc_action_layers,
        transformer_num_encoder_layers_pose=args.enc_pose_layers,
        transformer_d_model=args.hidden_dim,
        transformer_dropout=args.dropout,
        transformer_nhead=args.nheads,
        transformer_dim_feedforward=args.dim_feedforward,
        transformer_normalize_before=True,
        lambda_action_loss=args.lambda_action_loss,
        lambda_hand_2d=args.lambda_hand_2d,
        lambda_hand_z=args.lambda_hand_z,
        ntokens_pose=args.ntokens_pose,
        ntokens_action=args.ntokens_action,
        trans_factor=args.trans_factor,
        scale_factor=args.scale_factor,
        pose_loss=args.pose_loss
    ).to(rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    freeze.freeze_batchnorm_stats(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)

    epoch = 1
    if args.train_cont:
        epoch = reloadmodel.reload_model(model.module, args.resume_path) + 1
        reloadmodel.reload_optimizer(args.resume_path, optimizer, scheduler)

    for epoch_idx in range(epoch, args.epochs + 1):
        sampler.set_epoch(epoch_idx)
        if rank == 0:
            print(f"***Epoch #{epoch_idx}")

        epochpass.epoch_pass(
            loader,
            model,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_decay_gamma=args.lr_decay_gamma,
            use_multiple_gpu=False,
            tensorboard_writer=board_writer if rank == 0 else None,
            aggregate_sequence=False,
            is_single_hand=args.train_dataset != "h2ohands",
            dataset_action_info=dataset_info.action_to_idx,
            dataset_object_info=dataset_info.object_to_idx,
            ntokens=args.ntokens_action,
            is_demo=False,
            epoch=epoch_idx
        )

        if rank == 0 and epoch_idx % args.snapshot == 0:
            modelio.save_checkpoint(
                {
                    "epoch": epoch_idx,
                    "network": "HTT",
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler,
                },
                is_best=True,
                checkpoint=exp_id,
                snapshot=args.snapshot,
            )

    if board_writer:
        board_writer.close()
    cleanup_ddp()

def run_ddp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_tag', default='gibsonlee')
    parser.add_argument('--dataset_folder', default='../data_MHAV/')
    parser.add_argument('--cache_folder', default='./gibson/ckpts/')
    parser.add_argument("--ntokens_pose", type=int, default=16)
    parser.add_argument("--ntokens_action", type=int, default=128)
    parser.add_argument("--spacing", type=int, default=2)
    parser.add_argument("--train_dataset", choices=["h2ohands", "fhbhands", "mhavhands"], default="mhavhands")
    parser.add_argument("--train_split", default="train", choices=["test", "train", "val"])
    parser.add_argument("--center_idx", default=0, type=int)
    parser.add_argument("--center_jittering", type=float, default=0.1)
    parser.add_argument("--scale_jittering", type=float, default=0)
    parser.add_argument("--train_cont", action="store_true")
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)
    parser.add_argument("--lr_decay_step", type=float, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--trans_factor", type=float, default=100)
    parser.add_argument("--scale_factor", type=float, default=0.0001)
    parser.add_argument("--pose_loss", default="l1", choices=["l2", "l1"])
    parser.add_argument('--enc_pose_layers', default=2, type=int)
    parser.add_argument('--enc_action_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument("--lambda_action_loss", type=float, default=2)
    parser.add_argument("--lambda_hand_2d", type=float, default=1)
    parser.add_argument("--lambda_hand_z", type=float, default=100)
    parser.add_argument("--snapshot", type=int, default=1)
    parser.add_argument("--resume_path", type=str, default="")
    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    run_ddp()