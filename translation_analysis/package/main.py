#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import os
import sys
import torch
import logging
from datasets import load_dataset, load_from_disk
from pprint import PrettyPrinter
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import pandas as pd

from t_picker import picker
from t_translate import translate

# Globals
DEFAULT_MODEL = "LumiOpen/Poro-34B"
DATASET_PATH = "../../data/europarl_en-fi"
pprint = PrettyPrinter(compact=True).pprint

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    # Check if downloaded original dataset already exists
    if not os.path.exists(DATASET_PATH):
        logging.info("Predownloaded dataset not found. Starting download.")
        dataset = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")
        dataset.save_to_disk(dataset_path=DATASET_PATH)

    dataset = load_from_disk(dataset_path=DATASET_PATH)
    # Get only the translations to a dictionary
    translation_dicts = dataset["translation"]
    # Sort the dictionary based on the length of the value of "en"
    sorted_list = sorted(translation_dicts, key=lambda d: len(d['en']))
    # Drop every sample with the length < 10
    filtered_list = [d for d in sorted_list if len(d['en']) >= 10]
    # Make a DataFrame out of the dict (the warning shouldn't matter)
    df = pd.DataFrame.from_dict(filtered_list)
    # self-explanatory
    df = df.drop_duplicates()

    bands = 20
    per_band = 5
    sampled_data = picker(dframe=df, bands=bands, per=per_band, thold=0.05)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    logging.info(f"Model {DEFAULT_MODEL} loaded.")

    translated_data = translate(data=sampled_data, tokenizer=tokenizer, model=model)

    # TODO:
    # Use Poro to translate English texts to Finnish // kinda done
    # Calculate losses of already translated text and Poro translated text, compare
    # Draw graph or get values to draw one


if __name__ == "__main__":
    main()
