from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import LMDataset, Seq2SeqDataset
from evaluation import evaluate

import wandb


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--seq2seq", default=False, action="store_true")
    parser.add_argument("--model-type",
                        default="transformer",
                        choices=["lstm", "transformer"])
    parser.add_argument("--attention", default=False, type=bool)
    parser.add_argument("--wandb", default=False, action="store_true")
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    return args


def train(args):
    args.save_dir += "_" + args.model_type + ("_lm" if not args.seq2seq else "_seq2seq")
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.attention = False

    if args.wandb:
        wandb.init(project="generation", config=args)

    if args.model_type == "lstm":
        from lstm import LMModel, Seq2SeqModel
    elif args.model_type == "transformer":
        from transformer import LMModel, Seq2SeqModel

    if args.seq2seq:
        train_set = Seq2SeqDataset(device=device)
        valid_set = Seq2SeqDataset(split="valid", device=device)
        model = Seq2SeqModel(args, train_set.dictionary).to(device)
    else:
        train_set = LMDataset(device=device)
        valid_set = LMDataset(split="valid", device=device)
        model = LMModel(args, train_set.dictionary).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        args.num_epoch,
                                                        eta_min=1e-4)
                                                     
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True)

    evaluate(model, valid_set)
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for samples in pbar:

                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" %
                                     (epoch + 1, np.mean(losses),
                                      optimizer.param_groups[0]['lr']))
                global_step += 1
                if args.wandb:
                    wandb.log({"train loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, step=global_step)

        # scheduler.step()
        if epoch % args.save_interval == 0:
            torch.save(
                model,
                args.save_dir + "/{}_{}.pt".format(args.model_type, epoch + 1))
        valid_loss, valid_ppl = evaluate(model, valid_set)
        if args.wandb:
            wandb.log({"valid loss": valid_loss, "valid ppl": valid_ppl}, step=global_step)



if __name__ == "__main__":
    args = get_args()
    train(args)
