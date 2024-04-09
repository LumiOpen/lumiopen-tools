#!/usr/bin/env python3

# Basic script to finetune causal LM using HF Trainer.

import sys
import json

from argparse import ArgumentParser

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('model')
    ap.add_argument('train_data')
    ap.add_argument('eval_data')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    data = load_dataset('json', data_files={
        'train': args.train_data,
        'eval': args.eval_data,
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype='auto',
    )

    def tokenize(example):
        return tokenizer(
            example['text'],
            max_length=args.max_length,
            truncation=True,
        )
    data = data.map(tokenize)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors='pt',
        mlm=False,
    )

    train_args = TrainingArguments(
        output_dir='train_output',
        evaluation_strategy='steps',
        save_strategy='no',
        eval_steps=100,
        num_train_epochs=1,
    )

    trainer = Trainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=data['train'],
        eval_dataset=data['eval'],
    )

    result = trainer.evaluate()
    print(f'loss before training: {result["eval_loss"]:.2f}')

    trainer.train()

    result = trainer.evaluate()
    print(f'loss after training: {result["eval_loss"]:.2f}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
