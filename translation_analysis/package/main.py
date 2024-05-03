#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

# Main script

import sys
import logging
from argparse import ArgumentParser
from pprint import PrettyPrinter

from t_picker import picker
from t_translate import translate
import europarl
import elrcnorden

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def argparser():
    ap = ArgumentParser()
    data_choices = ["europarl", "elrcnord"]
    ap.add_argument("-d", "--dataset", choices=data_choices, required=True)
    ap.add_argument("-p", "--per", default=10)
    ap.add_argument("-b", "--bands", default=10)
    ap.add_argument("-t", "--thold", default=0.03)
    ap.add_argument("-m", "--minlen", default=10)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    logging.info(f"Running package with the following values:")
    logging.info(f"per: {args.per}, bands: {args.bands}, thold: {args.thold}, minlen: {args.minlen}, "
                 f"dataset: {args.dataset}")
    match args.dataset:
        case "europarl":
            europarl.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "elrcnord":
            elrcnorden.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case _:
            logging.error("Match-case defaulted somehow (???). Program will exit.")
            sys.exit(1)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
