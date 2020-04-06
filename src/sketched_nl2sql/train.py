from argparse import ArgumentParser, Namespace

string_args = [
    "--data_path",
    "/Users/chaoftic/Public/数据集/WikiSQL",
    "--seed",
    "1234",
    "--pretrained_model_name",
    "/Users/chaoftic/Public/bert-base-uncased",
    "--hidden_dim",
    "300",
]


def parse_args():
    """ arg parser """
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pretrained_model_name", type=str)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_aggs", type=int)
    parser.set_defaults(num_aggs=6)
    return parser.parse_args(string_args)


def main(args: Namespace):
    """ main """



if __name__ == "__main__":
    main(parse_args())
