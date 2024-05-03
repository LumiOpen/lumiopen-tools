#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import os
import sys
import json
import logging
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
import evaluate

from t_picker import picker
from t_translate import translate

# Globals
DATA_PATH = "../../data"
DATASET_NAME = "Helsinki-NLP/europarl"
DATASET_PATH = f"{DATA_PATH}/europarl_en-fi"

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main(per: int, bands: int, thold: float, minlen: int):
    """
    Handler script for Europarl.

    :param per: The amount of samples taken around one single band.
    :param bands: The amount of examination points where samples should be taken, i.e. resolution.
    :param thold: The amount of variation (percentage).
    :param minlen: Minimum string length for samples
    :return: Sampled dataset data
    """

    # Check if downloaded original dataset already exists
    if not os.path.exists(DATASET_PATH):
        logging.info("Predownloaded dataset not found. Starting download.")
        dataset = load_dataset(DATASET_NAME, "en-fi", split="train")
        dataset.save_to_disk(dataset_path=DATASET_PATH)

    dataset = load_from_disk(dataset_path=DATASET_PATH)
    # Get only the translations to a dictionary
    translation_dicts = dataset["translation"]
    # Sort the dictionary based on the length of the value of "en"
    sorted_list = sorted(translation_dicts, key=lambda d: len(d["en"]))
    # Drop every sample with the length < 10
    filtered_list = [d for d in sorted_list if len(d["en"]) >= minlen]
    # Make a DataFrame out of the list (the warning shouldn't matter)
    df = pd.DataFrame.from_dict(filtered_list)
    # self-explanatory
    df = df.drop_duplicates()

    sampled_data = picker(dframe=df, bands=bands, per=per, thold=thold)
    # pprint(sampled_data)

    with open(f"{DATA_PATH}/out/europarl_sampled_entries.json", mode='w') as file:
        json.dump(sampled_data, file, ensure_ascii=False)
    del file

    return sampled_data


    # TODO:
    # Use Poro to translate English texts to Finnish // kinda done
    # Calculate BLEU of already translated text and Poro translated text, compare
        # Use Google GLEU metric for this thing because they claim better performance with sentence-by-sentence stuff
    # Draw graph or get values to draw one


if __name__ == "__main__":
    main()
