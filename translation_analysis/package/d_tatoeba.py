#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import csv
import logging
import sys
import os
import pandas as pd
from t_picker import picker
import json

# Globals
DATA_PATH = "../../data"
DATASET_PATH = f"{DATA_PATH}/tatoeba/tatoeba-test-v2023-09-26.eng-fin.txt"


def main(per: int, bands: int, thold: float, minlen: int):
    if not os.path.isfile(DATASET_PATH):
        logging.error("Tatoeba dataset file could not be found. Program will now exit.")
        sys.exit(1)

    data = []
    with open(DATASET_PATH) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        for line in reader:
            sample = {}
            for i, entry in enumerate(line):
                if i < 2:
                    pass
                elif i == 2:
                    sample["en"] = entry
                elif i == 3:
                    sample["fi"] = entry
            data.append(sample)

    sorted_list = sorted(data, key=lambda d: len(d["en"]))
    # Drop every sample with the length < 10
    filtered_list = [d for d in sorted_list if len(d["en"]) >= minlen]
    # Make a DataFrame out of the list (the warning shouldn't matter)
    df = pd.DataFrame.from_dict(filtered_list)
    # self-explanatory
    df = df.drop_duplicates()

    sampled_data = picker(dframe=df, bands=bands, per=per, thold=thold)

    with open(f"{DATA_PATH}/out/tatoeba_sampled_entries.json", mode='w') as file:
        json.dump(sampled_data, file, ensure_ascii=False)

    return sampled_data


if __name__ == "__main__":
    # Some default values
    main(per=10, bands=10, thold=0.03, minlen=10)
