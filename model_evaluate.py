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
    print(res)

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
    # dataset_path = "Steven0633/image3"
    dataset = load_dataset(dataset_path)
    ds_builder = load_dataset_builder(dataset_path)

    print(f"evaluate {model_checkpoint} on dataset {dataset_path}")
    
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    assert(len(id2label) == 140)
    # print(id2label)


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

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"595-evaluate",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=50,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # push_to_hub=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__=="__main__":
    main()