#!/usr/bin/env python3

import os
import sys
import torch
from datasets import load_dataset, load_from_disk, Dataset
from pprint import PrettyPrinter
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

default_model = "LumiOpen/Poro-34B"
pprint = PrettyPrinter(compact=True).pprint
dataset_path = "../data/europarl_en-fi"


def argparser():
    ap = ArgumentParser(
        prog="generate_translations.py",
        description="Script for mass-generating translations with Poro into a dataset."
    )
    ap.add_argument('--dev', action='store_true')
    ap.add_argument('--amount', action='store', required=True)
    return ap


def get_translations(dataset, dev_mode, dataset_length):
    poro_translations = []

    if not dev_mode:
        model = AutoModelForCausalLM.from_pretrained(
            default_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(default_model)
        template = "<|user|>Käännä suomeksi: {} <|assistant|>"

        for i in range(dataset_length):
            prompt = template.format(dataset["translation"][i]["en"])
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**encoded, max_length=512)
            decoded: str = tokenizer.decode(output[0])
            assert decoded.startswith(prompt)  # idea: convert to flag+if block?
            pred: str = decoded[len(prompt):].rstrip('\n')  # cut only to output
            if pred.endswith(tokenizer.eos_token):
                pred = pred[:-len(tokenizer.eos_token)]
            pred = pred.rstrip('\n')
            poro_translations.append(pred)

            print(f"{i+1} / {dataset_length}")
            print(dataset["translation"][i]["en"])
            print(f"{pred}\n")
    else:
        for i in range(dataset_length):
            # because the model's not running just copy whatever's on "fi"
            poro_translations.append(dataset["translation"][i]["fi"])

    # declare dict from dataset and duplicate
    translation_dicts = dataset["translation"]
    for i, translation_dict in enumerate(translation_dicts):
        if i == dataset_length:
            break
        # append poro translations to duplicated dict
        translation_dict["fi_poro"] = poro_translations[i]
    # create new dataset with appended translations
    appended_dataset = Dataset.from_dict({"translation": translation_dicts})
    # save new dataset to new path
    appended_dataset.save_to_disk(dataset_path=f"{dataset_path}-fi_poro")
    print("New dataset length: " + str(len(appended_dataset["translation"])))
    print("Example entry: " + appended_dataset["translation"][0])


def main(argv):
    args = argparser().parse_args(argv[1:])
    dataset_length = int(args.amount)
    dev_mode = bool(args.dev)

    if dev_mode:
        print("Running on development mode, no model will be loaded.")
    if not os.path.exists(dataset_path):
        print("Didn't find already downloaded dataset. Downloading...")
        dataset = load_dataset("Helsinki-NLP/europarl", "en-fi", split=f"train[:{dataset_length}]")
        pprint(dataset)
        dataset.save_to_disk(dataset_path=dataset_path)
    dataset = load_from_disk(dataset_path=dataset_path)

    get_translations(dataset, dev_mode, dataset_length)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
