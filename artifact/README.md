# Artifact: experimental records for the decisive tables

Lightweight records backing the manuscript's central comparisons. Paths
are relative to the experiment project's `results/` directory; the
table generator (`make_tables.py`) consumes these files directly.

| Record | Backs |
|---|---|
| `router_results.json` | The 45 paired category/seed RGB image-AUROC records for both arms plus synthetic routing scores and decisions; source of the routing table and the paired RGB effects |
| `run_manifest.csv` | Run ID, dataset, seed, device, epochs, input/recon/coupling switches, per-penalty effective weights, and evaluation record for 216 decisive runs (CPU/CUDA distinction explicit) |
| `provenance_full.json` | Full provenance manifest: 460 entries, 468-checkpoint inventory, evaluator documentation |
| `factorial_endpoints.csv` | All 72 synthetic category/configuration/seed endpoints (absolute Perlin AUROC, smooth AUROC, family difference) |
| `matched_contrasts.json`, `matched_contrasts_seeds.json` | Matched C-on/C-off family-response contrasts (seed 42 and per-seed with mean and population SD) |
| `smooth_factorial*.json` | Smooth-factorial records for seeds 42/0/1 |
| `defect_family_sweep.json` | One-factor-at-a-time defect-family dose response (contrast/frequency/softness) with derivative-energy ratios |
| `lamc_screw_smooth*.json` | Coupling-weight intervention, seeds 42 and 0 (matched CUDA C-off control in the s0 record) |
| `gradient_probe.json`, `grad_magnitude.json`, `grad_params.json` | Logit-level opposition, loss-weighted magnitude ratios, and parameter-level norm-ratio/cosine probes |
| `compat_mcnemar.json`, `compat_metrics_v2.json` | Held-out compatibility rules: exact McNemar discordant counts and p-values, per-category descriptors |
| `repeatability.json` | Same-configuration repeat spread with deterministic controls and augmentation-stream hashes |
| `real_prior_bridge.json`, `real_moderation.json` | Real-defect bridge: per-image scores, group assignments (seed-42 z median), matched coupling effects E and moderation M with paired image-cluster bootstrap CIs across three seeds |
| `ksdd2_pixel_stats.json` | Canonical KolektorSDD2 localization evaluation: common-index paired pixel samples, AUROC/AP point estimates, image-cluster bootstrap CIs |
| `submission_checks.py`, `submission_checks.json` | Paired RGB seed/category summaries and leave-one-category-out sensitivity; separate repeatability controls, with source-file SHA-256 digests. Descriptive reanalysis, not new model fits or significance tests. |

Selected-table regeneration from a repository checkout:

    python make_tables.py --artifact

regenerates the ten artifact-backed decisive tables (smoothfact,
smoothfactrep, defectsweep, lamc, gradparam, compatv2, realmod,
ksdd2pixel, realbridge, router) from this directory, validating all required inputs first
and writing only after validation succeeds; campaign
aggregate inputs (e.g. `all_results.json`) are not part of the artifact
and their tables are skipped with a notice. Full regeneration of all 33
tables runs from the experiment project with `results/` present.
`moderation_bootstrap.py` is the exact released implementation behind
`real_moderation_ci.json` (10,000-replicate paired image-cluster
percentile bootstrap; RNG seed 20261015 + category index).

Training/evaluation code, full per-image prediction dumps beyond the
moderation records, and checkpoints reside in the experiment project
and are available from the authors.

Reproduce the submission sensitivity checks from a repository checkout
using Python 3 and the standard library only:

```sh
python artifact/submission_checks.py
```

The default paths are relative to the script, so it also works when
invoked from another directory. It writes `artifact/submission_checks.json`
without modifying either input record. The RGB cells remain paired;
the analysis does not treat the 45 category/seed cells as independent
datasets or supply image-sampling/retraining uncertainty.
