#!/usr/bin/env python3

import sys
import os
import torch

from math import exp
from statistics import mean
from logging import warning
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('text', nargs='+')
    return ap


def metrics(fn, tokenizer, model):
    with open(fn) as f:
        text = f.read()

    loss_fct = CrossEntropyLoss(reduction='sum')

    losses, ppls, cppls = [], [], []
    for ln, line in enumerate(text.splitlines(), start=1):
        encoded = tokenizer(
            line,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors='pt'
        ).to(model.device)

        assert encoded.input_ids.shape[0] == 1
        if encoded.input_ids.shape[1] < 2:
            warning(f'skip line {ln} in {fn}: fewer than 2 tokens: "{line}"')
            continue

        with torch.no_grad():
            output = model(**encoded, labels=encoded.input_ids)

        shift_logits = output.logits[:, :-1, :]
        shift_labels = encoded.input_ids[:, 1:]

        batch_size, seq_length, vocab_size = shift_logits.shape
        assert batch_size == 1
        
        total_loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length)
        )

        # if not torch.isclose(total_loss/seq_length, output.loss):
        #     warning(f'torch loss {float(total_loss/seq_length)} != '
        #             f'model loss {float(output.loss)}')

        loss = float(total_loss) / seq_length
        char_length = len(tokenizer.decode(shift_labels[0]))

        losses.append(loss)
        ppls.append(exp(loss))
        cppls.append(exp(total_loss/char_length))

    return (mean(losses), mean(ppls), mean(cppls))


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    losses, ppls, cppls = [], [], []
    for fn in args.text:
        bn = os.path.basename(fn)
        loss, ppl, cppl = metrics(fn, tokenizer, model)
        print(f'{bn} mean loss: {loss:.2f}')
        print(f'{bn} mean ppl : {ppl:.2f}')
        print(f'{bn} mean cppl: {cppl:.2f}')
        losses.append(loss)
        ppls.append(ppl)
        cppls.append(cppl)
        
    print(f'overall mean loss: {mean(losses):.2f}')
    print(f'overall mean ppl : {mean(ppls):.2f}')
    print(f'overall mean cppl: {mean(cppls):.2f}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
