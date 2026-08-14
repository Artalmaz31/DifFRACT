from transcoder_training import run_training
from transcoder_training.cli import build_parser, config_from_args


def main():
    p = build_parser(default_save_dir="./output_sae")
    args = p.parse_args()
    run_training(config_from_args(args), role="sae")


if __name__ == "__main__":
    main()
