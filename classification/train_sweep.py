from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from dataset import CLSDataset
from evaluation import evaluate
from model import Net

curdir = os.path.dirname(__file__)


def get_args(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--max-len", default=512, type=int)
    # parser.add_argument("--hidden-dim", default=512, type=int)
    # parser.add_argument("--embedding-dim", default=512, type=int)
    # parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "models"))
    parser.add_argument("--total-updates", default=50000, type=int)
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumualte before performing a backward/update pass."
    )
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    return args


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="classification", config=args)

    import json
    print(json.dumps(args.__dict__, indent=4))

    train_set = CLSDataset(device=device)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True)

    valid_set = CLSDataset(split="dev", device=device, max_len=args.max_len)
    model = Net(args, max_len=args.max_len).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    global_step = 0
    # evaluate(model, valid_set)
    # if loss did not decrease for 10 steps, stop training
    min_loss = 1e9
    step_not_improved = 0
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for step, samples in enumerate(pbar):
                # optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
                losses.append(loss.item())
                pbar.set_description(
                    "Epoch: %d, Loss: %0.8f, lr: %0.9f, step: %d" %
                    (epoch + 1, np.mean(losses),
                     optimizer.param_groups[0]['lr'], global_step))
                if optimizer.param_groups[0]['lr'] == 0:
                    break
                wandb.log({"train loss": np.mean(losses),
                            "lr": optimizer.param_groups[0]['lr'],
                            }, step=global_step)

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    step_not_improved = 0
                else:
                    step_not_improved += 1
                if step_not_improved > 20:
                    print("Early stopping")
                    break

        if epoch % args.save_interval == 0:
            torch.save(model, args.save_dir + "/ckpt_{}.pt".format(epoch + 1))
        if optimizer.param_groups[0]['lr'] == 0:
            break
        valid_loss, valid_acc = evaluate(model, valid_set)
        wandb.log({"valid loss": valid_loss,
                    "valid acc": valid_acc,
                    }, step=global_step)



if __name__ == "__main__":
    args = get_args()
    train(args)
