#!/usr/bin/env python3

# Fix poorly translated texts

import sys

import torch

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = 'LumiOpen/Poro-34B'

TEMPLATES = {
    "default": '<|prompt|>Seuraava syöteteksti on huonosti käännettyä suomen kieltä. Tehtäväsi on muokata teksti '
               'niin, että se on kieliopillisesti oikeellinen, ja kuulostaa luonnolliselta.<|input|>Tämä on hieno '
               'kysymys, johon ei ole yhtä oikeaa vastausta. Patentin vahvuus tulee kyvystä panna se täytäntöön. Jos '
               'patentinhaltija ei jostain syystä (kuten rahoituksen puutteesta) voi panna sitä täytäntöön, '
               'patentti on käytännössä hampaaton. Mutta kilpailijasi eivät todennäköisesti tiedä sitä. He voivat '
               'siksi saada heidät luopumaan loukkaamisesta pelkästään patentin olemassaololla ja olettamuksella, '
               'että haastat oikeuteen. Tällainen jäähdyttävä vaikutus kilpailuun voi olla sinulle arvokasta. '
               'Lisäksi, jos rikkomuksia tapahtuu, saatat pystyä hankkimaan lisenssisopimuksen ilman oikeudenkäyntiä. '
               'Tämä voi olla erittäin tuottoisa liiketoimintamalli ja voi siten oikeuttaa patentoinnin kustannukset. '
               '<|output|>Tämä on hyvä kysymys, johon ei ole yhtä oikeaa vastausta. Patentin vahvuus perustuu siihen, '
               'että se voidaan panna täytäntöön. Jos patentin omistaja ei pysty panemaan sitä täytäntöön mistä '
               'tahansa syystä (esimerkiksi rahoituksen puutteen vuoksi), patentti on käytännössä hampaaton. Mutta '
               'kilpailijasi eivät todennäköisesti voi tietää sitä. Tämän vuoksi jo patentin olemassaolo voi estää '
               'heitä loukkaamasta sitä, koska he olettavat että haastat heidät oikeuteen. Tällainen kilpailua '
               'hillitsevä vaikutus voi olla sinulle arvokas. Lisäksi, jos patenttia on loukattu, saatat saada aikaan '
               'lisenssisopimuksen ilman oikeudenkäyntiä. Tämä voi olla erittäin tuottoisa liiketoimintamalli, '
               'joten patentoinnin kustannukset saattavat olla perusteltuja.<|input|>'
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('file', nargs='?')
    return ap


def translate(stream, tokenizer, model, args):
    template = TEMPLATES["default"]

    for line in stream:
        line = line.rstrip('\n')
        prompt = template.format(line)

        encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = model.generate(
            **encoded,
            max_length=256,
            do_sample=True,
            repetition_penalty=2.0
        )
        decoded = tokenizer.decode(output[0])

        assert decoded.startswith(prompt)
        pred = decoded[len(prompt):]
        pred = pred.rstrip('\n')
        if pred.endswith(tokenizer.eos_token):
            pred = pred[:-len(tokenizer.eos_token)]
        pred = pred.rstrip('\n')

        print(pred)


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    print('model loaded.', file=sys.stderr)

    if args.file is None:
        translate(sys.stdin, tokenizer, model, args)
    else:
        with open(args.file) as f:
            translate(f, tokenizer, model, args)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
