"""Train timestep-conditioned SAE baselines for the transcoder-vs-SAE comparison."""

import argparse
from transcoder_training import TrainConfig, run_training


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    p.add_argument("--dataset-id", default=TrainConfig.dataset_id)
    p.add_argument("--save-dir", default="./output_sae")
    p.add_argument("--cycles", type=int, default=TrainConfig.total_cycles)
    p.add_argument("--buffer-size", type=int, default=TrainConfig.buffer_size)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--l1-img", type=float, default=3e-4)
    p.add_argument("--l1-txt", type=float, default=5e-5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = TrainConfig(
        dataset_id=args.dataset_id,
        target_layers=tuple(args.layers),
        save_dir=args.save_dir,
        total_cycles=args.cycles,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        l1_coeff={"img": args.l1_img, "txt": args.l1_txt},
        device=args.device,
    )
    run_training(cfg, role="sae")


if __name__ == "__main__":
    main()
