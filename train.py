"""Config-driven trainer for PGDN / DRAEM-lite (same code, physics off).

Example:
    python train.py --dataset construct --data-root /mnt/agents/work/data/construct \
        --class-name brushed_steel --run-name pgdn_bs --epochs 10
    python train.py --dataset mvtec --data-root /mnt/agents/work/data/mvtec_ad/screw \
        --run-name draem_screw --no-physics --epochs 20

Saves ``runs/<run_name>/{config.json,last.pt,log.json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from datasets.loaders import ConstructDataset, MVTecScrewDataset, make_dataloader
from losses import PGDNLoss
from models.pgdn import PGDN
from physics.operators import PhysicsPrior


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PGDN / DRAEM-lite")
    p.add_argument("--dataset", choices=["mvtec", "construct"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--class-name", default=None, help="CONSTRUCT class (default: all)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3],
                   help="1=grayscale (default), 3=RGB")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler", choices=["none", "cosine", "multistep"], default="none",
                   help="multistep: DRAEM-style MultiStepLR (gamma 0.1 at "
                        "40% and 75% of the schedule)")
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--frac", type=float, default=1.0,
                   help="defect-free data fraction (e.g. 1.0 / 0.6 / 0.4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="torch device (e.g. cpu / cuda)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-images", type=int, default=None,
                   help="cap number of training images (smoke tests)")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="cap batches per epoch (smoke tests)")
    # physics
    p.add_argument("--use-physics-arch", dest="use_physics_arch",
                   action="store_true", default=True)
    p.add_argument("--no-physics", dest="use_physics_arch", action="store_false",
                   help="DRAEM-lite: drop the physics channel AND both physics "
                        "losses (the original twin baseline)")
    p.add_argument("--lambda-phys", type=float, default=1.0)
    p.add_argument("--phys-channel", dest="phys_channel", default="on",
                   choices=["on", "off"],
                   help="factorial toggle: supply z as a segmentation input "
                        "channel (requires a prior)")
    p.add_argument("--phys-rec-loss", dest="phys_rec_loss", default="on",
                   choices=["on", "off"],
                   help="factorial toggle: derivative feature-matching loss")
    p.add_argument("--phys-couple-loss", dest="phys_couple_loss", default="on",
                   choices=["on", "off"],
                   help="factorial toggle: coupling loss (usable without the "
                        "input channel)")
    p.add_argument("--channel-variant", default="full",
                   choices=["full", "grad", "lap", "var", "shuffle"],
                   help="prior-channel control variant (review controls)")
    p.add_argument("--physics-color", action="store_true",
                   help="colour-aware prior: per-channel equilibrium stats "
                        "(RGB training; estimated from train/good)")
    # anomaly source
    p.add_argument("--anomaly-source", choices=["fractal", "external"],
                   default="fractal",
                   help="synthetic-defect source: fractal Perlin (default) or "
                        "an external natural-image bank (DTD-class)")
    p.add_argument("--source-dir", default=None,
                   help="image folder used as the external anomaly source")
    p.add_argument("--physics-stats", default=None,
                   help="path to physics_stats.json (default: <data-root>/physics_stats.json)")
    return p.parse_args(argv)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_prior(args) -> Optional[PhysicsPrior]:
    # A prior is needed if the channel is on OR any physics loss is on
    channel_on = args.use_physics_arch and args.phys_channel == "on"
    any_phys = channel_on or args.phys_rec_loss == "on" or args.phys_couple_loss == "on"
    if not any_phys:
        return None
    if args.physics_color:
        if args.channels != 3:
            raise SystemExit("--physics-color requires --channels 3")
        if args.physics_stats:
            raise SystemExit("--physics-color estimates per-channel stats from train/good")
        good = os.path.join(args.data_root, "train", "good")
        print(f"[train] estimating colour-aware prior from {good}")
        from physics.operators import PhysicsPriorColor
        return PhysicsPriorColor.estimate_from_folder_rgb(
            good, image_size=args.image_size)
    stats = args.physics_stats or os.path.join(args.data_root, "physics_stats.json")
    if os.path.isfile(stats):
        return PhysicsPrior.from_json(stats, args.class_name)
    # e.g. MVTec screw: no stats json -> estimate from train/good
    good = os.path.join(args.data_root, "train", "good")
    print(f"[train] no physics stats json; estimating prior from {good}")
    return PhysicsPrior.estimate_from_folder(good, image_size=args.image_size)


def build_dataset(args):
    kw = dict(image_size=args.image_size, frac=args.frac, seed=args.seed,
              channels=args.channels,
              source_dir=(args.source_dir
                          if args.anomaly_source == "external" else None))
    if args.dataset == "mvtec":
        return MVTecScrewDataset(args.data_root, "train", **kw)
    return ConstructDataset(args.data_root, "train", class_name=args.class_name, **kw)


def train(args) -> str:
    if not args.use_physics_arch:
        # --no-physics: the classic twin baseline -- no channel, no losses
        args.phys_channel = "off"
        args.phys_rec_loss = "off"
        args.phys_couple_loss = "off"
    set_seeds(args.seed)
    run_name = args.run_name or (
        f"pgdn_{args.dataset}_{int(time.time())}"
        if args.use_physics_arch
        else f"draem_{args.dataset}_{int(time.time())}"
    )
    run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    prior = build_prior(args)
    ds = build_dataset(args)
    if args.max_images:
        ds.items = ds.items[: args.max_images]
    loader = make_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                             seed=args.seed)
    print(f"[train] dataset={args.dataset} n_train={len(ds)} "
          f"frac={args.frac} prior={prior}")

    model = PGDN(
        base_channels=args.base_channels,
        use_physics_arch=args.use_physics_arch and args.phys_channel == "on",
        lambda_phys=args.lambda_phys,
        prior=prior,
        in_channels=args.channels,
        phys_use_rec=args.phys_rec_loss == "on",
        phys_use_couple=args.phys_couple_loss == "on",
        channel_variant=args.channel_variant,
    )
    device = torch.device(args.device)
    # CPU throughput optimization (no architecture change): channels_last
    # memory format is ~35% faster for these small conv UNets on CPU.
    model = model.to(memory_format=torch.channels_last).to(device)
    print(f"[train] model params: {model.num_parameters():,}")

    criterion = PGDNLoss(lambda_phys=args.lambda_phys,
                         use_rec=args.phys_rec_loss == "on",
                         use_couple=args.phys_couple_loss == "on")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[max(1, int(args.epochs * 0.4)),
                             max(2, int(args.epochs * 0.75))], gamma=0.1)

    log = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sums: Dict[str, float] = {}
        n = 0
        for bi, batch in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            out = model(batch["augmented"].to(device, memory_format=torch.channels_last))
            loss, parts = criterion(out, batch["clean"].to(device),
                                    batch["mask"].to(device),
                                    lambda_phys=args.lambda_phys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        if sched is not None:
            sched.step()
        entry = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                 "time_s": round(time.time() - t0, 2)}
        entry.update({k: round(v / max(n, 1), 6) for k, v in sums.items()})
        log.append(entry)
        print(f"[train] epoch {epoch:03d} " +
              " ".join(f"{k}={v}" for k, v in entry.items() if k != "epoch"))

    ckpt = {
        "model_state": model.state_dict(),
        "model_config": model.get_config(),
        "train_config": vars(args),
    }
    torch.save(ckpt, os.path.join(run_dir, "last.pt"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model_config": ckpt["model_config"],
                   "train_config": ckpt["train_config"]}, f, indent=2)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] saved run to {run_dir}")
    return run_dir


if __name__ == "__main__":
    train(parse_args())
