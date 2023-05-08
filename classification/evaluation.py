from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import torch

from dataset import CLSDataset
from model import *


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    batch_size = 16
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=dataset.collate_fn)

    losses = []
    hit = 0
    for samples in tqdm(dataloader, desc="validation"):
        targets = samples.pop("targets")

        logits = model.logits(**samples)
        # print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")
        predict_label = logits.argmax(dim=-1)
        hit += (predict_label == targets).sum()
        losses.append(F.cross_entropy(logits, targets).item())
    print("%s: loss: %.3f, acc: %.3f" %
          (dataset.split, np.mean(losses), hit / len(dataset)))


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    try:
        dataset = CLSDataset(split='test', device="cuda")
    except FileNotFoundError:
        dataset = CLSDataset(split="dev", device="cuda")
    try:
        model = torch.load(os.path.join(basedir, "models/cls_best.pt"), map_location='cpu').to('cuda')
    except FileNotFoundError as e:
        print(e)
        exit()
    evaluate(model, dataset)
