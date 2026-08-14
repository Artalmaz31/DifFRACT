import argparse
from .arch import ARCHS
from .train import TrainConfig


def build_parser(default_save_dir: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Train timestep-conditioned transcoders on an MM-DiT backbone."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--model",
        default="flux-schnell",
        choices=sorted(ARCHS),
        help="architecture recipe to train against (default: flux-schnell)",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="override the recipe's checkpoint, e.g. a local snapshot for offline runs",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="block indices to train; both img and txt streams per layer (default: 6 12 18)",
    )
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--dataset-column", default=None)
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default=None)
    p.add_argument("--save-dir", default=default_save_dir)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--expansion-factor", type=int, default=None)
    p.add_argument("--time-embed-dim", type=int, default=None)
    p.add_argument("--l1-img", type=float, default=None)
    p.add_argument("--l1-txt", type=float, default=None)
    p.add_argument("--lr-img", type=float, default=None)
    p.add_argument("--lr-txt", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--timestep-scale", type=float, default=None)
    p.add_argument("--prompts-per-inference", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--val-prompts", type=int, default=None)
    p.add_argument("--stats-every", type=int, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--no-comparison-image", dest="make_comparison_image", action="store_false", default=None)

    return p


def config_from_args(args) -> TrainConfig:
    l1 = None
    if args.l1_img is not None or args.l1_txt is not None:
        defaults = TrainConfig().l1_coeff
        l1 = {
            "img": args.l1_img if args.l1_img is not None else defaults["img"],
            "txt": args.l1_txt if args.l1_txt is not None else defaults["txt"],
        }
    lr = None
    if args.lr_img is not None or args.lr_txt is not None:
        defaults = TrainConfig().lr
        lr = {
            "img": args.lr_img if args.lr_img is not None else defaults["img"],
            "txt": args.lr_txt if args.lr_txt is not None else defaults["txt"],
        }

    try:
        return TrainConfig.for_model(
            args.model,
            model_id=args.model_id,

            dataset_id=args.dataset_id,
            dataset_column=args.dataset_column,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,

            target_layers=tuple(args.layers) if args.layers else None,
            expansion_factor=args.expansion_factor,
            d_model=args.d_model,
            time_embed_dim=args.time_embed_dim,
            l1_coeff=l1,
            lr=lr,

            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            timestep_scale=args.timestep_scale,
            height=args.height,
            width=args.width,

            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            total_cycles=args.cycles,
            prompts_per_inference=args.prompts_per_inference,

            val_prompts=args.val_prompts,
            val_every=args.val_every,
            stats_every=args.stats_every,
            make_comparison_image=args.make_comparison_image,

            save_dir=args.save_dir,
            device=args.device,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\nKnown --model values: {', '.join(sorted(ARCHS))}")
