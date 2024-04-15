#!/usr/bin/env python3

import os
import sys
import torch
from datasets import load_dataset, load_from_disk, Dataset
from pprint import PrettyPrinter
from transformers import AutoTokenizer, AutoModelForCausalLM

NO_MODEL_DEV_MODE = True
DATASET_LENGTH = 100

default_model = "LumiOpen/Poro-34B"
pprint = PrettyPrinter(compact=True).pprint
dataset_path = "../data/europarl_en-fi"


def get_translations(dataset):
    poro_translations = []

    if not NO_MODEL_DEV_MODE:
        model = AutoModelForCausalLM.from_pretrained(
            default_model,
            device_map="auto",
            torch_dtype=torch.bfload16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        template = "<|user|>Käännä suomeksi: {} <|assistant|>"

        for i in range(DATASET_LENGTH):
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
    else:
        for i in range(DATASET_LENGTH):
            # because the model's not running just copy whatever's on "fi"
            poro_translations.append(dataset["translation"][i]["fi"])

    # declare dict from dataset and duplicate
    translation_dicts = dataset["translation"]
    for i, translation_dict in enumerate(translation_dicts):
        # append poro translations to duplicated dict
        translation_dict["fi_poro"] = poro_translations[i]
    # create new dataset with appended translations
    appended_dataset = Dataset.from_dict({"translation": translation_dicts})
    # save new dataset to new path
    appended_dataset.save_to_disk(dataset_path=f"{dataset_path}-fi_poro")
    print(appended_dataset["translation"][0])


def main():
    if NO_MODEL_DEV_MODE:
        print("Running on development mode, no model will be loaded.")
    if not os.path.exists(dataset_path):
        print("Didn't find already downloaded dataset. Downloading...")
        dataset = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train[:100]")
        pprint(dataset)
        dataset.save_to_disk(dataset_path=dataset_path)
    dataset = load_from_disk(dataset_path=dataset_path)

    templates = {
        "translate": "<|user|>Käännä suomeksi: {} <|assistant|>",
        "fix_translation": '<|poor_translation|>{} <|good_translation|>'
    }
    template = templates["translate"]

    get_translations(dataset)


if __name__ == '__main__':
    main()
