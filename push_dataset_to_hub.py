import numpy as np
import torch
import os
import shutil


import sys
from datasets import load_dataset
from datasets import load_metric
from datasets import DatasetDict
import evaluate
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


gao_dataset = "gao_dataset"
yijie_dataset = "yijie_dataset"

# We also define a collate_fn, which will be used to batch examples together. 
# Each batch consists of 2 keys, namely pixel_values and labels.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    dataset = load_dataset("imagefolder", data_dir=gao_dataset)
    
    
    print("dataset:")
    print(dataset)
    print(dataset["train"].features)

    # Preprocessing the data
    splits = dataset["train"].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    print("train_ds[0]:")
    print(train_ds[0])

    ddict = DatasetDict({
    "train": train_ds,   # split1_ds is an instance of `datasets.Dataset`
    "test": val_ds,
    })
    ddict.push_to_hub("epigone707/595Gao")


if __name__=="__main__":
    main()