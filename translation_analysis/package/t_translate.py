#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

from transformers import AutoTokenizer, PreTrainedTokenizerFast
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def translate(data: list[dict[str, int | list]], tokenizer: PreTrainedTokenizerFast, model):
    """
    Handles translation. Adapted from poro_translated.py.

    :param data: Data formatted in the defined way.
    :param tokenizer: Tokenizer to be used for translation.
    :param model: Model to be used for translation.
    :return: Translated entries within a dict object.
    """
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"

    results = []
    for i, band in enumerate(data):
        band_dict = {
            "band_loc": data[i]["band_loc"],
            "band_no": data[i]["band_no"],
            "median_len": data[i]["median_len"],
            "entries": []
        }
        for j, entry in enumerate(band["entries"]):
            logging.info(f"Translating entry no. {j} in band no. {i}.")
            en_text = entry[0]
            prompt = template.format(en_text)

            encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
            output = model.generate(
                **encoded,
                max_new_tokens=256
            )
            decoded = tokenizer.decode(output[0])

            assert decoded.startswith(prompt)
            pred = decoded[len(prompt):]
            pred = pred.rstrip('\n')
            if pred.endswith(tokenizer.eos_token):
                pred = pred[:-len(tokenizer.eos_token)]
            pred = pred.rstrip('\n')

            band_dict["entries"].append(pred)
        results.append(band_dict)
    return results
