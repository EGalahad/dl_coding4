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
    parser.add_argument("--model-name", default="bert-base-chinese", type=str)
    parser.add_argument("--last-layers", default=3, type=int)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "models"))
    parser.add_argument("--total-updates", default=50000, type=int)
    parser.add_argument("--wandb", default=False, action="store_true")
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


def train_embeddings(net: Net):
    for param in net.model.parameters():
        param.requires_grad = False
    for param in net.model.embeddings.parameters():
        param.requires_grad = True

def train_last_n_layers(net: Net, n: int):
    for param in net.model.parameters():
        param.requires_grad = False
    for layer in net.model.encoder.layer[-n:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in net.model.pooler.parameters():
        param.requires_grad = True

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import json
    print(json.dumps(args.__dict__, indent=4))

    if args.wandb:
        wandb.init(project="classification", config=args)

    train_set = CLSDataset(device=device)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True)

    valid_set = CLSDataset(split="dev", device=device, max_len=args.max_len)
    model = Net(args, model_name=args.model_name, max_len=args.max_len).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=1,verbose=True)

    global_step = 0
    # evaluate(model, valid_set)
    for epoch in range(args.num_epoch):
        if epoch < 3:
            train_last_n_layers(model, args.last_layers)
        else:
            if epoch % 2 == 0:
                train_embeddings(model)
            else:
                train_last_n_layers(model, args.last_layers)
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
                if args.wandb:
                    wandb.log({"train loss": np.mean(losses),
                               "lr": optimizer.param_groups[0]['lr'],
                               }, step=global_step)
        if epoch % args.save_interval == 0:
            torch.save(model, args.save_dir + "/ckpt_{}.pt".format(epoch + 1))
        if optimizer.param_groups[0]['lr'] == 0:
            break
        valid_loss, valid_acc = evaluate(model, valid_set)
        if args.wandb:
            wandb.log({"valid loss": valid_loss,
                       "valid acc": valid_acc,
                       }, step=global_step)



if __name__ == "__main__":
    args = get_args()
    train(args)
