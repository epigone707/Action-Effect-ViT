import numpy as np
import torch
import os
import shutil


import sys
from datasets import load_dataset

from datasets import load_metric
from datasets import load_dataset_builder
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
from transformers import pipeline



model_checkpoint = "595-finetuned-task-acc580/checkpoint-182" # pre-trained model from which to fine-tune
batch_size = 64 # batch size for training and evaluation

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )
# metric = load_metric("accuracy")
metric = evaluate.load("accuracy")


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch



# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    res = metric.compute(predictions=predictions, references=eval_pred.label_ids)
    y_true = np.array(eval_pred.label_ids)

    top_5 = np.argsort(eval_pred.predictions, axis = 1)[:,-5:]
    top_5_acc = np.mean(np.array([1 if y_true[k] in top_5[k] else 0 for k in range(len(top_5))]))

    res["top-5-accuracy"] = top_5_acc

    top_20 = np.argsort(eval_pred.predictions, axis = 1)[:,-20:]
    top_20_acc = np.mean(np.array([1 if y_true[k] in top_20[k] else 0 for k in range(len(top_20))]))

    res["top-20-accuracy"] = top_20_acc

    return res

# We also define a collate_fn, which will be used to batch examples together. 
# Each batch consists of 2 keys, namely pixel_values and labels.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    # load dataset
    dataset_path = "epigone707/595Gao"
    dataset = load_dataset(dataset_path)
    ds_builder = load_dataset_builder(dataset_path)

    print(f"evaluate {model_checkpoint} on dataset {dataset_path}")
    
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    assert(len(id2label) == 140)

    # Preprocessing the data
    train_ds = dataset['train']
    val_ds = dataset['test']

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    
    # define the model
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
    # print(val_ds[0])
    val_size = len(val_ds)

    from random import sample
    indices = list(range(val_size))
    print(val_ds[0])
    pred = pipe(val_ds[0]["image"])
    print(pred)
    print(label2id)
    N=100
    top_k = 140
    top_1_accuracy = 0
    top_5_accuracy = 0
    top_20_accuracy = 0
    print("N:",N,"top_k:",top_k)
    for i in range(N):
        sampled_idx = sample(indices,140) # pick 140 images (candidate images)
        correct_answer = val_ds[sampled_idx[0]]["label"] # this is the correct class label (action), which is an int
        print("correct_answer:",correct_answer)
        predictions = []
        for idx in sampled_idx:
            true_label = val_ds[idx]["label"] # an integer
            pred = pipe(val_ds[idx]["image"],top_k = top_k)
            for p in pred:
                if label2id[p['label']] == correct_answer:
                    predictions.append((p['score'], true_label))
        sorted_predictions = sorted(predictions, reverse=True,key=lambda x:x[0])
        print("sorted predictions:")
        print(sorted_predictions)
        if sorted_predictions[0][1] == correct_answer:
            top_1_accuracy += 1
        if correct_answer in [p[1] for p in sorted_predictions[:5]]:
            top_5_accuracy += 1
        if correct_answer in [p[1] for p in sorted_predictions[:20]]:
            top_20_accuracy += 1
    # print(correct_answer)

    print("top_1_accuracy:",top_1_accuracy/N)
    print("top_5_accuracy:",top_5_accuracy/N)
    print("top_20_accuracy:",top_20_accuracy/N)


if __name__=="__main__":
    main()