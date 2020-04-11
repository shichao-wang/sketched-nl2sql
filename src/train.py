import logging
from argparse import ArgumentParser

import sketched_nl2sql
from torchnlp.config import Config

logging.basicConfig(level=logging.DEBUG)
parser = ArgumentParser()
parser.add_argument("config_file", type=str)
parser.add_argument("--retrain", action="store_true")
args = parser.parse_args()


config = Config.from_yaml(args.config_file)
logging.info(config)
sketched_nl2sql.train(config.get("data_path"), config, config.get("checkpoint_file"), resume=not args.retrain)
