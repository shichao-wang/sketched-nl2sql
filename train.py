import importlib

import sketched_nl2sql
from torchnlp.config import Config

importlib.reload(sketched_nl2sql.dataset)
DATA_PATH = "/Users/chaoftic/Public/数据集/WikiSQL"
config = Config(pretrained_model_name="/Users/chaoftic/Public/bert-base-uncased", hidden_dim=300)

sketched_nl2sql.train(DATA_PATH, "save", config, 1234)
