#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import torch
import random

from argparse import ArgumentParser
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

ACCELERATOR = Accelerator()

DEFAULT_MODEL = 'LumiOpen/Poro-34B'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--dry_run", "-d", action="store_true")
    ap.add_argument('--model', default=DEFAULT_MODEL)
    return ap


def prepper(translations):
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"

    new_ds_dict = {
        "samples": []
    }
    for entry in translations["translation"]:
        entry["en"] = template.format(entry["en"])
        new_ds_dict["samples"].append(f"{entry['en']}{entry['fi']}")

    return new_ds_dict


def main(argv):
    args = argparser().parse_args(argv[1:])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")  # With europarl, everything's in "train"
    ds = ds.shuffle(random.seed(5834)).select(range(10000))  # Shuffle dataset and limit sample amount
    ds = ds.train_test_split(test_size=0.2)

    def tokenize(translations):
        for idx, entry in enumerate(translations["samples"]):
            translations["samples"][idx] = tokenizer(
                entry,
                max_length=args.max_length,
                truncation=True
            )

        return translations

    def preprocess(translations):
        prepped_translations = prepper(translations)
        tokenized_translations = tokenize(prepped_translations)
        return tokenized_translations

    # print(data["train"][0])
    # print(data["test"][0])
    # print(f"{type(data_train_tokenized)}: {data_train_tokenized[0]}")
    # print(f"{type(data_test_tokenized)}: {data_test_tokenized[0]}")

    with ACCELERATOR.main_process_first():
        data_train_tokenized = ds["train"].map(
            preprocess,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns("translation")["samples"]
        data_test_tokenized = ds["test"].map(
            preprocess,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns("translation")["samples"]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors='pt',
        mlm=False,
    )

    train_dataloader = DataLoader(
        data_train_tokenized, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

    test_dataloader = DataLoader(
        data_test_tokenized, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

    if not args.dry_run:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
        )

        num_epochs = args.epochs
        lr = 3e-5
        gradient_accumulation_steps = 8

        optimizer = AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs)
        )

        model, train_dataloader, test_dataloader, optimizer, lr_scheduler = ACCELERATOR.prepare(
            model, train_dataloader, test_dataloader, optimizer, lr_scheduler
        )

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            dl_len = len(train_dataloader)
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                ACCELERATOR.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                # capture batch analytics

            model.eval()
            eval_loss = 0
            dl_len = len(test_dataloader)
            for step, batch in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()

            model.save_pretrained(f"trained_model-{epoch}")


if __name__ == '__main__':
    sys.exit(main(sys.argv))
