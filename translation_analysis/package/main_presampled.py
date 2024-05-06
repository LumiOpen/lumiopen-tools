#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi
import argparse
import sys
import logging
from argparse import ArgumentParser
from pprint import PrettyPrinter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from t_translate import translate

DATA_PATH = "../../data"
DEFAULT_MODEL = "LumiOpen/Poro-34B"

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def argparser():
    ap = ArgumentParser()
    ap.add_argument("file", type=argparse.FileType('r'))
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    sampled_data = json.load(args.file)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    logging.info(f"Model {args.model} loaded.")

    translated_data = translate(data=sampled_data, tokenizer=tokenizer, model=model)

    with open(f"{DATA_PATH}/out/translated_entries.json", mode='w') as file:
        json.dump(translated_data, file, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
