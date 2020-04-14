""" entry script """
import logging
from argparse import ArgumentParser

import sketched_nl2sql
from torchnlp.config import Config

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--retrain", action="store_true")
    return parser.parse_args()


def main(args):
    config = Config.from_yaml(args.config_file)
    logging.info(config)
    sketched_nl2sql.train(
        config.get("data_path"),
        config,
        config.get("checkpoint_file"),
        resume=not args.retrain,
    )


if __name__ == "__main__":
    main(parse_args())
