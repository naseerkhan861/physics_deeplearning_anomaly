#!/usr/bin/env python3
"""Generate ALL LaTeX tables for the physics-dl-paper manuscript from the
experimental record.  Run from pgdn_project/:  python physics-dl-paper/make_tables.py
Every number in the paper flows through this script."""
import json
import os

import numpy as np

import sys
ARTIFACT = "--artifact" in sys.argv
# default: run from the experiment project; --artifact: regenerate the
# artifact-backed decisive tables from a repository checkout (campaign
# aggregates are absent there and their tables are skipped with a notice)
R = "artifact" if ARTIFACT else "results"
RAW = "artifact" if ARTIFACT else f"{R}/raw"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else {}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def pm(m, s):
    if m is None:
        return "--"
    return f"{m:.3f}$\\pm${s:.3f}" if s is not None else f"{m:.3f}"


WIDE = {"budget", "capacity", "compatv2", "factorial", "mvteccats",
        "official", "rgbfull", "router", "seedrob"}


def esc(s):
    # escape bare underscores in plain data cells (metal_nut -> metal\_nut);
    # leave cells containing LaTeX (math/commands) untouched
    s = str(s)
    if "_" in s and "\\" not in s and "$" not in s:
        return s.replace("_", r"\_")
    return s


PENDING = {}


def w(name, rows, header, caption, label, span=""):
    os.makedirs(OUT, exist_ok=True)
    ncols = len(header)
    span = span or name in WIDE  # two-column floats for the wide tables
    cols = "l" + "c" * (ncols - 1)
    env = "table*" if span else "table"
    placement = "tp" if name == "rgbfull" else "!t"
    lines = [f"% auto-generated: {label}", f"\\begin{{{env}}}[{placement}]",
             f"\\caption{{{caption}}} \\label{{{label}}}",
             "\\centering\\footnotesize"]
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    if name == "compatv2":  # ragged-right first column, X-columns fill \linewidth
        lines.append("\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}"
                     "p{0.18\\linewidth}*{3}{>{\\centering\\arraybackslash}X}c@{}}")
    else:
        lines.append(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    lines += ["\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(c) for c in r) + " \\\\")
    endtab = "tabularx" if name == "compatv2" else "tabular"
    lines += ["\\bottomrule", f"\\end{{{endtab}}}", f"\\end{{{env}}}"]
    PENDING[name] = "\n".join(lines) + "\n"
    # body-only variant (tabular without the float) for consolidated panels
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith(("\\setlength", "\\begin{tabular")))
        end = next(i for i, l in enumerate(lines)
                   if l.startswith("\\end{tabular"))
        PENDING[name + "_body"] = "\n".join(lines[start:end + 1]) + "\n"
    except StopIteration:
        pass


if ARTIFACT:  # write only artifact-backed tables; campaign blocks are read-only
    _w_full = w
    ARTIFACT_TABLES = {"smoothfact", "smoothfactrep", "defectsweep", "lamc",
                       "gradparam", "compatv2", "realmod", "ksdd2pixel",
                       "realbridge", "router", "gradprobe"}

    def w(name, *a, **k):  # noqa: F811
        base = name.rsplit("_body", 1)[0]
        if name in ARTIFACT_TABLES or base in ARTIFACT_TABLES:
            _w_full(name, *a, **k)


def preflight():
    """Validate every required input for the tables that will be written
    before any output changes (round-6 review C2)."""
    if not ARTIFACT:
        return
    required = ["smooth_factorial.json", "smooth_factorial_s0.json",
                "smooth_factorial_s1.json", "defect_family_sweep.json",
                "lamc_screw_smooth.json", "lamc_screw_smooth_s0.json",
                "grad_params.json", "compat_metrics_v2.json",
                "real_moderation.json", "real_moderation_ci.json",
                "ksdd2_pixel_stats.json", "compat_mcnemar.json",
                "router_results.json", "gradient_probe.json"]
    missing = [f for f in required if not os.path.isfile(f"{RAW}/{f}")]
    if missing:
        raise SystemExit("artifact preflight failed, missing: "
                         + ", ".join(missing))


def main():
    preflight()
    A = load(f"{R}/all_results.json")
    RF = load(f"{R}/rgb_full_results.json")
    PL = load(f"{R}/plugin_results.json")
    RG = load(f"{R}/rgb_results.json")
    SR = load(f"{RAW}/seed_robustness.json")
    RT = load(f"{RAW}/router_results.json")
    con = A.get("construct", {})

    # ---- 1. main 37-category campaign (H1 + H2 retention + H3 spearman)
    rows = []
    for ds, d in A.get("datasets", {}).items():
        a = d["aggregate"]
        rows.append([ds.replace("Superficial", "Superf.").replace("KolektorSSD2", "KolektorSDD2"),
                     f3(a.get("pgdn_image_auroc")), f3(a.get("draem_image_auroc")),
                     f3(a.get("pc_image_auroc")), f3(a.get("pgdn_pixel_auroc")),
                     f3(a.get("pc_pixel_auroc")),
                     f"{100*a['pgdn_retention_img']:.1f}\\%" if a.get("pgdn_retention_img") else "--",
                     f3(a.get("spearman_pgdn")), f3(a.get("spearman_draem"))])
    if con.get("main"):
        rows.append(["CONSTRUCT", f3(con["main"]["pgdn"]["image_auroc"]),
                 f3(con["main"]["draem"]["image_auroc"]), f3(con["main"]["pc"]["image_auroc"]),
                 f3(con["main"]["pgdn"]["pixel_auroc"]), f3(con["main"]["pc"]["pixel_auroc"]),
                 f"{100*con['f06']['pgdn']['image_auroc']/con['main']['pgdn']['image_auroc']:.1f}\\%",
                 f3(con["explain"]["spearman_defective"]["pgdn"]["mean"]),
                 f3(con["explain"]["spearman_defective"]["draem"]["mean"])])
    # fill KolektorSDD2 pixel columns from the canonical localization record
    kp_gray = load(f"{RAW}/ksdd2_pixel_stats.json").get("gray-128", {})
    for r in rows:
        if r[0].startswith("KolektorSDD2") and kp_gray.get("pgdn_pixel_auroc"):
            r[4] = f3(kp_gray["pgdn_pixel_auroc"])
    if A:
        w("main37", rows,
      ["benchmark", "PGDN img", "\\draem{} img", "PC img", "PGDN pix", "PC pix",
       "PGDN ret.@60\\%", "Spear.\\ PGDN", "Spear.\\ \\draem{}"],
      "Gray-128 campaign over 37 category-benchmarks: macro image/pixel AUROC, retention of "
      "image AUROC at 60\\% training data, and Spearman agreement between the anomaly map and "
      "the physics residual on defective images.", "tab:main37", span=1)

    # ---- 2. per-category MVTec AD (gray) with twin pixel column
    rows = []
    for c in A.get("datasets", {}).get("MVTec AD", {}).get("categories", []):
        if "pgdn" not in c:
            continue
        rows.append([c["tag"].replace("mvtec_", ""), f3(c["pgdn"].get("image_auroc")),
                     f3(c.get("draem", {}).get("image_auroc")), f3(c.get("pc", {}).get("image_auroc")),
                     f3(c["pgdn"].get("pixel_auroc")), f3(c.get("draem", {}).get("pixel_auroc")),
                     f3(c.get("pc", {}).get("pixel_auroc"))])
    w("mvteccats", rows, ["category", "PGDN img", "\\draem{} img", "PC img",
                          "PGDN pix", "\\draem{} pix", "PC pix"],
      "Per-category results, gray-128 campaign (MVTec AD); pixel AUROC for all three methods.",
      "tab:mvteccats")

    # ---- 3. per-category VisA + LOCO (gray)
    rows = []
    for bench in ("VisA", "MVTec LOCO"):
        for c in A.get("datasets", {}).get(bench, {}).get("categories", []):
            if "pgdn" not in c:
                continue
            rows.append([f"{bench}/{c['tag'].split('_', 1)[1]}",
                         f3(c["pgdn"].get("image_auroc")), f3(c.get("draem", {}).get("image_auroc")),
                         f3(c.get("pc", {}).get("image_auroc"))])
    w("visaloco", rows, ["category", "PGDN img", "\\draem{} img", "PC img"],
      "Per-category image AUROC (gray-128): VisA and MVTec LOCO.", "tab:visaloco")

    # ---- 4. CONSTRUCT per-severity
    rows = []
    for meth, name in (("pgdn", "PGDN"), ("draem", "\\draem{}"), ("pc", "PatchCore-lite")):
        sev = con.get("main", {}).get(meth, {}).get("per_severity", {})
        for s in ("mild", "medium", "severe"):
            if s in sev:
                rows.append([name, s, sev[s]["n_anomalous"], f3(sev[s]["image_auroc"]),
                             f3(sev[s].get("image_ap")), f3(sev[s]["pixel_auroc"])])
    if rows:
        w("severity", rows, ["model", "severity", "$n$", "\\shortstack{image\\\\AUROC}",
                         "\\shortstack{image\\\\AP}", "\\shortstack{pixel\\\\AUROC}"],
      "CONSTRUCT per-severity breakdown (gray-128).", "tab:severity")

    # ---- 5. H2 data efficiency (CONSTRUCT)
    rows = []
    for frac, key in (("100\\%", "main"), ("60\\%", "f06"), ("40\\%", "f04")):
        for meth, name in (("pgdn", "PGDN"), ("draem", "\\draem{}")):
            d = con.get(key, {}).get(meth)
            if d:
                rows.append([name, frac, f3(d["image_auroc"]), f3(d["image_ap"]), f3(d["pixel_auroc"])])
    if rows:
        w("h2data", rows, ["model", "train frac", "\\shortstack{image\\\\AUROC}",
                       "\\shortstack{image\\\\AP}", "\\shortstack{pixel\\\\AUROC}"],
      "Network-fitting data retention on CONSTRUCT (gray-128): performance under 40\\% and 60\\% network-fitting reduction; the prior stays calibrated from the full defect-free directory.",
      "tab:h2data")

    # ---- 6. lambda ablation
    rows = [[f"{l['lambda']}", l["model"].replace("DRAEM-lite (lambda=0)",
             "\\shortstack[l]{\\draem{}\\\\($\\lambda{=}0$, arch off)}"),
             f3(l["image_auroc"]), f3(l["image_ap"]), f3(l["pixel_auroc"])]
            for l in con.get("lambda_ablation", [])]
    if rows:
        w("lambda", rows, ["$\\lambda_{\\mathrm{phys}}$", "model", "\\shortstack{image\\\\AUROC}",
                       "\\shortstack{image\\\\AP}", "\\shortstack{pixel\\\\AUROC}"],
      "Physics-loss weight ablation (CONSTRUCT, gray-128).", "tab:lambda")

    # ---- 7. seed robustness
    rows = []
    for bench, label in (("screw", "MVTec AD screw"), ("construct", "CONSTRUCT")):
        for meth, name in (("pgdn", "PGDN"), ("draem", "\\draem{}"), ("patchcore", "PatchCore-lite")):
            m = SR.get(bench, {}).get(meth, {})
            if not m:
                continue
            rows.append([label, name, pm(m["image_auroc"]["mean"], m["image_auroc"]["std"]),
                         pm(m["image_ap"]["mean"], m["image_ap"]["std"]),
                         pm(m["pixel_auroc"]["mean"], m["pixel_auroc"]["std"])])
    w("seedrob", rows, ["benchmark", "model", "\\shortstack{image\\\\AUROC}",
                        "\\shortstack{image\\\\AP}", "\\shortstack{pixel\\\\AUROC}"],
      "Seed robustness on the primary benchmarks (seeds 42/0/1/2, gray-128).", "tab:seedrob")

    # ---- 8. H3 channel ablation (CONSTRUCT)
    ab = con.get("explain", {}).get("physics_channel_ablation", {})
    rows = [["image AUROC", f3(ab.get("image_auroc_intact")), f3(ab.get("image_auroc_zeroed")),
             f3(ab.get("image_auroc_drop"))],
            ["pixel AUROC", f3(ab.get("pixel_auroc_intact")), f3(ab.get("pixel_auroc_zeroed")),
             f3(ab.get("pixel_auroc_drop"))]]
    if rows:
        w("h3ab", rows, ["metric", "intact", "physics channel zeroed", "drop"],
      "Physics-channel attribution on CONSTRUCT: zeroing the fixed input channel at test time.",
      "tab:h3ab")

    # ---- 9. plug-in: PatchCore +- phys
    rows = []
    for bench, a in PL.get("plugin", {}).get("per_benchmark", {}).items():
        def f(k):
            return f3(a.get(k))
        rows.append([bench, f("base_f10_img"), f("phys_f10_img"), f("base_f03_img"),
                     f("phys_f03_img"),
                     f"{100*a['base_retention']:.1f}\\%" if a.get("base_retention") else "--",
                     f"{100*a['phys_retention']:.1f}\\%" if a.get("phys_retention") else "--"])
    w("plugin", rows, ["benchmark", "base @100\\%", "+phys @100\\%", "base @30\\%",
                       "+phys @30\\%", "ret.\\ base", "ret.\\ +phys"],
      "Feature-level plug-in: PatchCore-lite with per-patch equilibrium statistics appended to "
      "the descriptors (image AUROC; fit-data retention in last columns).", "tab:plugin", span=1)

    # ---- 10. RGB multi-seed (with pixel localization columns)
    rows = []
    for c, e in RF.get("primary", {}).items():
        pc = RF.get("pc256", {}).get(c, {})
        rows.append([c, pm(e["pgdn"]["image_auroc_mean"], e["pgdn"]["image_auroc_std"]),
                     pm(e["draem"]["image_auroc_mean"], e["draem"]["image_auroc_std"]),
                     f3(pc.get("image_auroc")),
                     f3(e["pgdn"].get("pixel_auroc_mean")),
                     f3(e["draem"].get("pixel_auroc_mean")),
                     f3(pc.get("pixel_auroc"))])
    m = RF.get("primary_macro", {})
    rows.append(["\\textbf{macro}", pm(m.get("pgdn_macro_img"), m.get("pgdn_macro_img_std")),
                 pm(m.get("draem_macro_img"), m.get("draem_macro_img_std")),
                 f3(RF.get("pc256_macro", {}).get("image_auroc")), "", "", ""])
    w("rgbfull", rows, ["category", "PGDN img", "\\draem{} img", "PC img",
                        "PGDN pix", "\\draem{} pix", "PC pix"],
      f"RGB/256, 24-epoch schedule, mean$\\pm$std over seeds 42/0/1 "
      f"(image AUROC; pixel AUROC columns; PGDN wins {m.get('wins_pgdn_of_pairs', '?')} "
      "category-seed pairs).", "tab:rgbfull")

    # ---- 11. RGB extras: external source / colour prior / f06
    rows = []
    for c, r in RF.get("external_source", {}).items():
        rows.append([f"source: {c}", f3(r["pgdn"]["fractal_img"]), f3(r["pgdn"]["external_img"]),
                     f3(r["draem"]["fractal_img"]), f3(r["draem"]["external_img"])])
    for c, r in RF.get("color_prior", {}).items():
        rows.append([f"colour prior: {c}", f3(r["luma_img"]), f3(r["color_img"]), "", ""])
    for c, r in RF.get("f06", {}).items():
        rows.append([f"60\\% data: {c}", f3(r["pgdn"]["f10_img"]), f3(r["pgdn"]["f06_img"]),
                     f3(r["draem"]["f10_img"]), f3(r["draem"]["f06_img"])])
    w("rgbextras", rows,
      ["arm (seed 42)", "PGDN a", "PGDN b", "\\draem{} a", "\\draem{} b"],
      "RGB/256 manipulations (image AUROC, 24-epoch). Source rows: fractal$\\to$external "
      "anomaly source; colour rows: luminance prior$\\to$colour-aware prior; data rows: "
      "100\\%$\\to$60\\% training data.", "tab:rgbextras")

    # ---- 12. budget arms
    rows = []
    for c in ("screw", "bottle", "capsule", "cashew"):
        p24 = (RF.get("primary", {}).get(c, {}).get("pgdn", {}).get("image_auroc_seeds") or [None])[0]
        d24 = (RF.get("primary", {}).get(c, {}).get("draem", {}).get("image_auroc_seeds") or [None])[0]
        p7 = load(f"{RAW}/r700_{c}_pgdn_eval.json").get("image_auroc")
        d7 = load(f"{RAW}/r700_{c}_draem_eval.json").get("image_auroc")
        pc = RF.get("pc256", {}).get(c, {}).get("image_auroc")
        rows.append([c, f3(p24), f3(d24), f3(p7), f3(d7),
                     f"{100*(p7-d7):+.1f}" if p7 and d7 else "--", f3(pc)])
    w("budget", rows, ["category", "PGDN 24", "\\draem{} 24", "PGDN 700", "\\draem{} 700",
                       "$\\Delta$@700", "PC-256"],
      "Published-budget arms (700 epochs, seed 42): under the longer schedule "
      "the data-driven twin degrades while the physics arm holds or improves; "
      "the cause of the twin's degradation is not established here.", "tab:budget")

    # ---- 13. capacity dose-response
    rows = []
    for r in PL.get("capacity", []):
        def g(meth, cfg):
            return f3(r.get(f"{meth}_{cfg}", {}).get("image_auroc"))
        rows.append([r["tag"], g("pgdn", "1x"), g("draem", "1x"), g("pgdn", "b48"),
                     g("draem", "b48"), g("pgdn", "r256"), g("draem", "r256")])
    w("capacity", rows, ["category", "PGDN 1$\\times$", "\\draem{} 1$\\times$",
                         "PGDN 9$\\times$", "\\draem{} 9$\\times$", "PGDN 256$^2$", "\\draem{} 256$^2$"],
      "Capacity dose-response (image AUROC).", "tab:capacity")

    # ---- 14. router with constant-choice baseline and regret
    mac = RT.get("macros", {})
    # derive every supported policy from the paired records (round-12 R4);
    # only the prior-validity row is a retained historical manual analysis
    per = RT.get("per_category", {})
    EXPECT_CATS = {"bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                   "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                   "transistor", "wood", "zipper"}
    EXPECT_SEEDS = {"42", "0", "1"}
    _mar = RT.get("margin", 0.02)
    MARGIN_THR = _mar.get("threshold", 0.02) if isinstance(_mar, dict) else float(_mar)
    if set(per) != EXPECT_CATS:
        raise SystemExit("routing record: category set mismatch: "
                         f"{sorted(set(per) ^ EXPECT_CATS)}")
    cells = []
    for c, v in per.items():
        seeds = v.get("seeds", {})
        if set(seeds) != EXPECT_SEEDS:
            raise SystemExit(f"routing record {c}: seed set mismatch: {sorted(seeds)}")
        for sd, e in seeds.items():
            for domain in ("real", "syn"):
                for side in ("pgdn", "draem"):
                    x = e.get(domain, {}).get(side)
                    if not isinstance(x, (int, float)) or not (0.0 <= x <= 1.0):
                        raise SystemExit(f"routing record {c}/{sd}: "
                                         f"{domain}.{side} missing or out of "
                                         f"[0,1]: {x!r}")
            pick_a = "pgdn" if e["syn"]["pgdn"] >= e["syn"]["draem"] else "draem"
            if e["pick_argmax"] != pick_a:
                raise SystemExit(f"routing record {c}/{sd}: pick_argmax "
                                 f"{e['pick_argmax']!r} inconsistent with scores")
            # complete margin rule incl. the fallback band: select PGDN only
            # when its synthetic advantage strictly exceeds the threshold;
            # ties and small margins fall back to the twin (round-14 review)
            gap = e["syn"]["pgdn"] - e["syn"]["draem"]
            expected_margin_pick = "pgdn" if gap > MARGIN_THR else "draem"
            if e["pick_margin"] != expected_margin_pick:
                raise SystemExit(f"routing record {c}/{sd}: pick_margin "
                                 f"{e['pick_margin']!r} inconsistent with the "
                                 f"documented margin rule (threshold {MARGIN_THR})")
            tb = "pgdn" if e["real"]["pgdn"] > e["real"]["draem"] else "draem"
            if e["true_best"] != tb:
                raise SystemExit(f"routing record {c}/{sd}: true_best "
                                 f"{e['true_best']!r} inconsistent with real scores")
            cells.append((c, sd, e))
    if len(cells) != 45:
        raise SystemExit(f"routing record incomplete: {len(cells)} pairs, expected 45")

    def _mean(vals):
        return sum(vals) / len(vals)
    m_draem = _mean([e["real"]["draem"] for _, _, e in cells])
    m_pgdn = _mean([e["real"]["pgdn"] for _, _, e in cells])
    m_arg = _mean([e["real"][e["pick_argmax"]] for _, _, e in cells])
    m_mar = _mean([e["real"][e["pick_margin"]] for _, _, e in cells])
    m_orc = _mean([e["real"][e["true_best"]] for _, _, e in cells])
    n_arg = sum(e["pick_argmax"] == e["true_best"] for _, _, e in cells)
    n_mar = sum(e["pick_margin"] == e["true_best"] for _, _, e in cells)
    n_pgdn = sum(e["true_best"] == "pgdn" for _, _, e in cells)
    reg = lambda m: f"${m_draem - m:+.3f}$"
    rows = [["always \\draem{} (constant choice)", f3(m_draem),
             f"{int(45 - n_pgdn)}/45 = {100*(45-n_pgdn)/45:.1f}\\%", "$0$"],
            ["always PGDN", f3(m_pgdn), f"{n_pgdn}/45 = {100*n_pgdn/45:.1f}\\%",
             reg(m_pgdn)],
            ["routed: synthetic argmax", f3(m_arg), f"{n_arg}/45", reg(m_arg)],
            ["routed: synthetic margin", f3(m_mar), f"{n_mar}/45", reg(m_mar)],
            ["routed: prior validity (historical manual analysis, "
             "15 category decisions)", "0.800", "9/15", "$+0.008$"],
            ["oracle (upper bound)", f3(m_orc), "45/45", reg(m_orc)]]
    w("router", rows, ["ensemble", "\\shortstack{image\\\\AUROC}", "decision accuracy",
                       "regret vs.\\ constant choice"],
      "Label-free routing versus the constant-choice baseline and the oracle. Regret is the "
      "mean per-decision AUROC loss relative to always choosing \\draem{} (negative = gain); "
      "no label-free router attains negative regret.", "tab:router")

    # ---- 15. LOCO AUPRO
    pro_rows = []
    for c in ("breakfast_box juice_bottle pushpins screw_bag splicing_connectors").split():
        d = load(f"{RAW}/pro_loco_{c}.json")
        if not d:
            continue
        def gp(meth, k):
            return f3(d.get(meth, {}).get(k))
        pro_rows.append([c, gp("pgdn", "overall"), gp("draem", "overall"), gp("pc", "overall"),
                         gp("pgdn", "structural_anomalies"), gp("draem", "structural_anomalies"),
                         gp("pc", "structural_anomalies"),
                         gp("pgdn", "logical_anomalies"), gp("draem", "logical_anomalies"),
                         gp("pc", "logical_anomalies")])
    if pro_rows:
        w("pro", pro_rows,
          ["category", "PGDN", "\\draem{}", "PC", "PGDN", "\\draem{}", "PC", "PGDN", "\\draem{}", "PC"],
          "MVTec LOCO AUPRO@0.3 computed with the conventional PRO metric on "
          "our tested subset and split (not the dataset authors' official "
          "AU-sPRO protocol); "
          "overall / structural / logical.",
          "tab:pro", span=1)

    # ---- 16. published SOTA context (venues verified against primary records)
    prim = RF.get("primary", {})
    pix_p = [e["pgdn"].get("pixel_auroc_mean") for e in prim.values()]
    pix_d = [e["draem"].get("pixel_auroc_mean") for e in prim.values()]
    pix_c = [v.get("pixel_auroc") for v in RF.get("pc256", {}).values()]
    def mm(v):
        v = [x for x in v if x is not None]
        return f3(sum(v) / len(v)) if v else "--"
    rows = [["PaDiM~\\cite{ref1}", "ICPR 2021", "0.953", "0.979", "--"],
            ["FastFlow~\\cite{ref9}", "preprint 2021", "0.994", "--", "--"],
            ["DRAEM~\\cite{ref6}", "ICCV 2021", "0.980", "0.973", "--"],
            ["RD4AD~\\cite{ref3}", "CVPR 2022", "0.985", "0.985", "--"],
            ["PatchCore~\\cite{ref2}", "CVPR 2022", "0.991 (0.996)", "0.981 (0.984)", "--"],
            ["WinCLIP~\\cite{ref10}", "CVPR 2023", "0.918 (0.931)", "0.851 (0.952)", "0.781"],
            ["SimpleNet~\\cite{ref4}", "CVPR 2023", "0.996", "0.984", "--"],
            ["EfficientAD~\\cite{ref5}", "WACV 2024", "0.991", "--", "--"],
            ["RealNet~\\cite{ref7}", "CVPR 2024", "0.995", "--", "--"],
            ["GLASS~\\cite{ref8}", "ECCV 2024", "0.999", "--", "0.988"],
            ["AF-CLIP~\\cite{ref20}", "2025", "0.929", "0.923", "--"],
            ["\\textbf{PGDN (this work, lite)}", "--",
             f"\\textbf{{{f3(RF.get('primary_macro', {}).get('pgdn_macro_img'))}}} macro",
             f"\\textbf{{{mm(pix_p)}}} macro", "--",
             f"(twin {mm(pix_d)} pix)"]]
    w("sota", rows, ["method (published)", "venue", "MVTec img", "MVTec pix", "VisA img", ""],
      "Published state of the art 2021--2025 (MVTec AD means; venues per the primary "
      "publication records; reference context, not capacity-matched: RGB 224--256$^2$). "
      "All methods use large-scale pretrained backbones except DRAEM, whose networks "
      "are trained from scratch (lr $10^{-4}$, MultiStepLR, DTD anomaly source); "
      "CLIP-based methods (WinCLIP) use web-scale vision--language pretraining rather "
      "than ImageNet classification. PGDN "
      "entries are 15-category macro means; the twin's macro pixel AUROC is given in "
      "the last column for the same aggregation.", "tab:sota", span=1)

    # ---- 16b. per-category factorial + interaction (round-2 review)
    FACT_CFGS = (("base", "baseline"), ("in", "input $z$ only"), ("rec", "recon.~loss only"),
                 ("cpl", "coupling only"), ("inrec", "input + recon."),
                 ("incpl", "input + coupling"), ("recpl", "both losses"), ("full", "full PGDN"))
    FACT_CATS = ("screw", "bottle", "capsule", "cashew", "construct")
    rows = []
    for cfg, label in FACT_CFGS:
        row = [label]
        for c in FACT_CATS:
            d = load(f"{RAW}/fact_{c}_{cfg}_eval.json")
            row.append(f3(d.get("image_auroc")) if d else "--")
        d_all = [load(f"{RAW}/fact_{c}_{cfg}_eval.json") for c in FACT_CATS]
        real = [d["image_auroc"] for c, d in zip(FACT_CATS, d_all)
                if d and c != "construct"]
        row.append(f3(sum(real) / len(real)) if real else "--")
        rows.append(row)
    if rows:
        w("factorialpc", rows,
      ["configuration", "screw", "bottle", "capsule", "cashew", "CONSTRUCT",
       "real-cat.\\ macro"],
      "Per-category factorial results (image AUROC, seed 42): a macro average conceals "
      "opposite per-category effects, so both are shown; the real-category macro "
      "excludes the synthetic CONSTRUCT unit.", "tab:factorialpc", span=1)

    # interaction (channel x coupling, reconstruction off), per available seed
    rows = []
    seeds_done = []
    for sd in (42, 0, 1):
        pre = "fact" if sd == 42 else "f2"
        sfx = "" if sd == 42 else f"_s{sd}"
        vals = {}
        ok = True
        for cfg in ("base", "in", "cpl", "incpl"):
            per = []
            for c in FACT_CATS:
                d = load(f"{RAW}/{pre}_{c}_{cfg}{sfx}_eval.json")
                if not d:
                    per = None   # incomplete panel: skip the whole seed
                    break
                per.append(d["image_auroc"])
            if per is None:
                print(f"seed {sd}: incomplete interaction panel ({cfg}), skipped")
                ok = False
                break
            vals[cfg] = sum(per) / len(per)
        if not ok:
            continue
        seeds_done.append(sd)
        did = 100 * (vals["incpl"] - vals["in"] - vals["cpl"] + vals["base"])
        rows.append([f"seed {sd}", f3(vals["base"]), f3(vals["in"]), f3(vals["cpl"]),
                     f3(vals["incpl"]), f"{did:+.1f}"])
    if len(seeds_done) > 1:
        dids = [float(r[-1].replace("+", "")) for r in rows]
        rows.append(["\\textbf{mean$\\pm$SD}", "", "", "", "",
                     f"\\textbf{{{np.mean(dids):+.1f}$\\pm${np.std(dids):.1f}}}"])
    ri = load(f"{RAW}/real_interaction.json")
    if ri:
        for s in ("42", "0", "1"):
            m3 = ri["per_seed"][s]["macro3"]
            m4 = ri["per_seed"][s].get("macro4")
            if m4 is not None:
                rows.append([f"real macro3 s{s}", "", "", "", "", f"{m3:+.1f}"])
                rows.append([f"real macro4 s{s} (w/cashew)", "", "", "", "", f"{m4:+.1f}"])
            else:
                rows.append([f"real macro3 s{s}", "", "", "", "", f"{m3:+.1f}"])
    if rows:
        w("interaction", rows,
      ["run", "base", "input", "coupling", "input$+$coupling",
       "interaction (pp)"],
      "Channel$\\times$coupling interaction with the reconstruction loss disabled "
      "(difference-in-differences of macro image AUROC over the five factorial "
      "categories, percentage points; negative = antagonism). Replicated across "
      "seeds; SD is the population convention across seeds. Real-only "
      "sensitivity rows recompute the contrast on real defects (macro3: "
      "screw/capsule/bottle from the released per-image scores; macro4 "
      "adds cashew from the standard factorial evaluations recorded in "
      "the interaction record): negative at every seed, so the additive "
      "antagonism is not an artifact of the synthetic unit's limited "
      "metric headroom (on CONSTRUCT the input arm saturates at "
      "$1.000$ and coupling reaches $0.996$).", "tab:interaction", span=1)

    # ---- 16c. smooth-defect test across factorial configurations
    # Complete matched panel required (review Step 2): every (cat, cfg) must
    # exist or generation fails loudly.
    SM_CATS = ("screw", "capsule", "bottle")
    SM_CFGS = ("base", "in", "rec", "cpl", "inrec", "incpl", "recpl", "full")
    sf = load(f"{RAW}/smooth_factorial.json")
    if not sf:
        raise SystemExit("smooth-factorial record missing or empty: "
                         f"{RAW}/smooth_factorial.json")
    missing = [f"{c}/{g}" for c in SM_CATS for g in SM_CFGS
               if f"{c}/{g}" not in sf]
    if missing:
        raise SystemExit(f"smoothfact panel incomplete, missing: {missing}")
    rows = []
    for cat in SM_CATS:
        for cfg in SM_CFGS:
            e = sf[f"{cat}/{cfg}"]
            rows.append([cat if cfg == "base" else "", cfg,
                         f3(e["perlin"]), f3(e["smooth"]),
                         f"{100*(e['smooth']-e['perlin']):+.1f}"])
    w("smoothfact", rows, ["category", "configuration", "perlin", "smooth",
                           "$\\Delta$ (pp)"],
      "Smooth-defect test across all eight factorial configurations (frozen "
      "checkpoints, seed-42 training, paired generators over up to 30 held-out "
      "normal images (30/23/20 for screw/capsule/bottle), fixed seed): the drop appears "
      "with the coupling loss (screw $-26.4$ coupling-only) and is absent "
      "from baseline, input-only, and reconstruction-only arms -- "
      "component-level isolation of a category-dependent failure family.",
      "tab:smoothfact", span=1)

    # ---- 16c2. smooth-factorial replication across training seeds
    smr = [load(f"{RAW}/smooth_factorial.json")]
    smr += [load(f"{RAW}/smooth_factorial_s{s}.json") for s in (0, 1)]
    if all(smr):
        rows = []
        for cat in SM_CATS:
            for cfg in SM_CFGS:
                deltas = [100 * (d[f"{cat}/{cfg}"]["smooth"] - d[f"{cat}/{cfg}"]["perlin"])
                          for d in smr]
                rows.append([cat if cfg == "base" else "", cfg]
                            + [f"{dv:+.1f}" for dv in deltas]
                            + [f"{np.mean(deltas):+.1f}$\\pm${np.std(deltas):.1f}"])
        w("smoothfactrep", rows,
          ["category", "configuration", "$\\Delta$ s42", "$\\Delta$ s0",
           "$\\Delta$ s1", "mean$\\pm$SD"],
          "Smooth-minus-Perlin difference (pp) of Table~\\ref{tab:smoothfact} "
          "replicated across training seeds 42/0/1 (frozen checkpoints, "
          "paired-background protocol, up-to-30 backgrounds): the screw coupling failure "
          "appears in every seed; the capsule and bottle input$+$coupling "
          "failures appear in two of three seeds.", "tab:smoothfactrep")

    # ---- 16d. coupling-gradient probe
    gp = load(f"{RAW}/gradient_probe.json")
    if gp:
        rows = []
        for k, e in gp.items():
            cat, cfg = k.split("/")
            rows.append([cat, cfg, f"{e['perlin']['oppose_w']:.2f}",
                         f"{e['smooth']['oppose_w']:.2f}",
                         f"{e['perlin']['sign_coup_on_mask']:+.2f}",
                         f"{e['smooth']['sign_coup_on_mask']:+.2f}"])
        w("gradprobe", rows,
          ["category", "config", "\\shortstack{oppose\\\\(perlin)}",
           "\\shortstack{oppose\\\\(smooth)}", "\\shortstack{coupling sign\\\\(perlin)}",
           "\\shortstack{coupling sign\\\\(smooth)}"],
          "Coupling-gradient conflict probe on trained checkpoints: fraction of the "
          "segmentation-gradient mass on defect-mask pixels where the coupling term "
          "pushes the opposite direction ($|g_{\\mathrm{seg}}|$-weighted), and the mean "
          "sign of the coupling gradient on mask pixels (positive $=$ suppression, per "
          "$\\partial C/\\partial p_j=(\\bar z_p - z_j)/P$).", "tab:gradprobe")

    # ---- 16d1b. parameter-level gradient conflict (round-3 review Step 10)
    gp2 = load(f"{RAW}/grad_params.json")
    if gp2:
        rows = []
        for k, e in gp2.get("runs", {}).items():
            cat, cfg = k.split("/")
            rows.append([cat, cfg,
                         f"{e['perlin']['norm_ratio']:.1f}",
                         f"{e['perlin']['cosine']:+.2f}",
                         f"{e['smooth']['norm_ratio']:.1f}",
                         f"{e['smooth']['cosine']:+.2f}"])
        w("gradparam", rows,
          ["category", "config", "ratio (perlin)", "cos (perlin)",
           "ratio (smooth)", "cos (smooth)"],
          "Parameter-level gradient conflict on trained checkpoints: norm "
          "ratio $\\|g_C^{\\lambda}\\|/\\|g_{\\mathrm{seg}}\\|$ with "
          "$g_C^{\\lambda}=\\nabla_{\\theta}(\\lambda_C C)$ the already "
          "loss-weighted coupling gradient, and cosine alignment over the "
          "shared parameters (mean over two batches of "
          "eight images). On screw the coupling gradient is strongly "
          "anti-aligned with the segmentation gradient (cosine $-0.6$ to "
          "$-0.97$) and can dominate it in norm (ratio up to $10.8$), while "
          "the non-failing coupling-only arms of capsule and bottle remain "
          "aligned on Perlin defects.", "tab:gradparam")

    # ---- 16d2. defect-family controlled sweeps (round-3 review Step 9)
    ds_sw = load(f"{RAW}/defect_family_sweep.json")
    if ds_sw:
        sw = ds_sw.get("sweeps", {}).get("screw", {})
        ARMS8 = ("base", "in", "rec", "cpl", "inrec", "incpl", "recpl", "full")
        if sw and all(a in sw for a in ARMS8):
            rows = []
            for factor, levels in (("contrast", ("0.10", "0.18", "0.25", "0.40")),
                                   # grid-cells order: fine to coarse
                                   ("frequency", ("high", "mid", "low")),
                                   ("softness", ("hard", "mid", "soft"))):
                for i, lv in enumerate(levels):
                    key = lv if factor == "contrast" else f"{factor}:{lv}"
                    dr = sw["base"][key].get("deriv_ratio")
                    rows.append([factor if i == 0 else "", lv,
                                 f"{dr:.2f}" if dr else "--"]
                                + [f3(sw[a][key]["image_auroc"]) for a in ARMS8])
            w("defectsweep", rows,
              ["factor", "level", "deriv.\\ ratio"] + list(ARMS8),
              "Defect-family dose-response on screw: the smooth-defect "
              "generator's contrast, defect frequency, and binary mask smoothing "
              "varied one factor at a time over the same 30 paired backgrounds "
              "(frozen seed-42 factorial checkpoints). The deriv.\\ ratio is "
              "the gradient energy inside the defect mask relative to the "
              "clean background. The coupling-only arm improves "
              "substantially only at the highest contrast "
              "($0.484\\to0.849$); the input$+$coupling and full arms "
              "remain at chance across the whole family; the uncoupled "
              "arms are not uniformly robust (reconstruction-only loses "
              "$23.6$ pp at the lowest contrast; input-only loses "
              "$19.0$ pp from mid to low (coarser) defect frequency).",
              "tab:defectsweep", span=1)

    # ---- 16d3. lambda_C intervention (round-4 review Step 4: matched controls)
    lam = load(f"{RAW}/lamc_screw_smooth.json")
    lam0 = load(f"{RAW}/lamc_screw_smooth_s0.json")
    if not (lam and lam0):
        raise SystemExit("coupling-weight records missing: "
                         "results/raw/lamc_screw_smooth*.json")
    rows = []
    COFF_KEYS = ("C-off (base)", "C-off (base, cuda)")
    for tag, label in (("C-off (base)", "$0$ (C-off)"),
                       ("0.1", "$0.1$"), ("0.25", "$0.25$"),
                       ("0.5", "$0.5$"), ("1.0", "$1.0$")):
        cells = [label]
        for src in (lam, lam0):
            key = next((k for k in COFF_KEYS if k in src), tag) \
                if tag == "C-off (base)" else tag
            if key in src:
                e = src[key]
                cells += [f3(e["perlin"]), f3(e["smooth"]),
                          f"{100*(e['smooth']-e['perlin']):+.1f}"]
            else:
                cells += ["--", "--", "--"]
        rows.append(cells)
    w("lamc", rows,
      ["$\\lambda_C$", "perlin s42", "smooth s42", "$\\Delta$ s42",
       "perlin s0", "smooth s0", "$\\Delta$ s0"],
      "Coupling-weight intervention on the screw coupling-only arm "
      "(8-epoch gray recipe, 30 paired backgrounds, fixed evaluation "
      "seed). Seed 42: CUDA-trained weights with the CPU-trained "
      "factorial baseline reused as C-off. Seed 0: fully matched CUDA "
      "controls including C-off. The smooth-defect failure onsets "
      "between $\\lambda_C=0.1$ and $0.25$ at both seeds, and the "
      "over-weighted term degrades the Perlin score at $0.5$--$1.0$; "
      "absolute deterioration accompanies the family gap. The sweep "
      "supports deterioration between the sampled weights in this "
      "setting, not a universal critical threshold.", "tab:lamc", span=1)

    # ---- 16e. compatibility v2 (pooled-quantile edge density, McNemar)
    cv2m = load(f"{RAW}/compat_metrics_v2.json")
    if not cv2m:
        raise SystemExit("compatibility record missing: "
                         f"{RAW}/compat_metrics_v2.json")
    rows = []
    for c, m in sorted(cv2m.get("per_category", {}).items()):
        rows.append([c, f"{m['edge_density_pooled']:.3f}", f"{m['stationarity_cv']:.3f}",
                     f3(m.get("z_in_minus_out")), f"{m['delta']:+.3f}"])
    ho = cv2m.get("held_out", {})
    mc = load(f"{RAW}/compat_mcnemar.json").get("splits", {})
    MC_KEY = {"edge_density_pooled": "edge_devthr",
              "stationarity_cv": "statcv_devthr"}
    missing_mc = [f"{v}_{t}" for v in MC_KEY.values() for t in ("fwd", "rev")
                  if f"{v}_{t}" not in mc]
    if missing_mc:
        raise SystemExit("compatibility paired-record splits missing: "
                         + ", ".join(missing_mc)
                         + " (refusing to fall back to a chance test)")

    def holine(feat, tag):
        e = ho.get(f"{feat}_{tag}")
        if not e:
            return "--"
        mke = mc.get(f"{MC_KEY.get(feat, '')}_{tag}", {})
        if mke:
            p = f"$p_M{{=}}{mke['mcnemar_exact_p']:.2f}$"
        else:
            p = f"$p_{{\\mathrm{{bin}}}}{{=}}{e['binom_p']:.2f}$"
        return (f"{e['correct']}/{e['n']} (const {e['constant_correct']}/{e['n']}, "
                f"{p})")
    rows.append(["\\textbf{held-out acc.\\ (fwd/rev)}",
                 holine("edge_density_pooled", "fwd") + " / " + holine("edge_density_pooled", "rev"),
                 holine("stationarity_cv", "fwd") + " / " + holine("stationarity_cv", "rev"),
                 holine("z_in_minus_out", "fwd") + " / " + holine("z_in_minus_out", "rev"), ""])
    w("compatv2", rows,
      ["category", "edge density (pooled thr.)", "stationarity CV",
       "$z$ in$-$out (diag.)", "$\\Delta$ img"],
      "Compatibility v2: edge density uses a pooled 95th-percentile threshold "
      "across categories (informative; varies 0.000--0.152). Held-out rule "
      "correctness versus the constant-choice baseline, compared with exact "
      "McNemar on discordant pairs ($p_M$; thresholds fitted on development "
      "categories only); the diagnostic column reports a two-sided binomial "
      "test against chance ($p_{\\mathrm{bin}}$). No descriptor reliably beats "
      "its constant-choice baseline; reverse-split edge density is 5/7 versus "
      "4/7, a numerical edge without statistical support.",
      "tab:compatv2", span=1)

    # ---- 17. datasets (static)
    rows = [["MVTec AD~\\cite{ref14}", "15", "320 (typ.)", "160 (typ.)", "yes"],
            ["MVTec AD 2~\\cite{ref19}", "3", "137--291", "140--245", "yes"],
            ["MVTec LOCO~\\cite{ref15}", "5", "335--424", "94--117", "yes"],
            ["VisA~\\cite{ref16}", "12", "450--905", "150--201", "yes"],
            ["MTDSuperficial~\\cite{ref18}", "1", "762", "358", "yes"],
            ["KolektorSDD2~\\cite{ref17}", "1", "2085", "1004", "yes"],
            ["\\shortstack[l]{CONSTRUCT\\\\(ours, synthetic)}", "3", "450", "240", "yes"]]
    w("datasets", rows, ["benchmark", "categories", "\\shortstack{defect-free\\\\train}",
                         "test", "\\shortstack{pixel\\\\masks}"],
      "Benchmarks used in the study.", "tab:datasets")

    # ---- 18. factorial ablation (review P2)
    FACT_CFGS = [("base", "baseline (no channel, no losses)"),
                 ("in", "input $z$ only"), ("rec", "recon.~loss only"),
                 ("cpl", "coupling loss only"),
                 ("inrec", "input + recon.~loss"), ("incpl", "input + coupling"),
                 ("recpl", "both losses (no input)"), ("full", "full PGDN")]
    rows = []
    for cfg, label in FACT_CFGS:
        imgs, pixs = [], []
        for cat in ("screw", "bottle", "capsule", "cashew", "construct"):
            d = load(f"{RAW}/fact_{cat}_{cfg}_eval.json")
            if d:
                imgs.append(d["image_auroc"])
                pixs.append(d.get("pixel_auroc"))
        if imgs:
            rows.append([label, f3(np.mean(imgs)), f3(np.std(imgs)), f3(np.mean(pixs)),
                         len(imgs)])
    if rows:
        w("factorial", rows, ["configuration", "macro img AUROC", "SD", "macro pix AUROC", "$n$"],
          "Factorial ablation of the three interventions (input channel / derivative "
          "reconstruction loss / coupling loss), gray-128, five boundary categories "
          "(screw, bottle, capsule, cashew, CONSTRUCT), seed 42; SD is across those "
      "five category units (including the synthetic CONSTRUCT unit), not "
      "across training seeds.", "tab:factorial")

    # ---- 19. channel-content controls (review P2)
    rows = []
    for cat in ("screw", "construct"):
        base = load(f"{RAW}/fact_{cat}_full_eval.json")
        row = [cat, f3(base.get("image_auroc")) if base else "--",
               f3(base.get("pixel_auroc")) if base else "--"]
        for var in ("grad", "lap", "var", "shuffle"):
            d = load(f"{RAW}/ctrl_{cat}_{var}_eval.json")
            row.append(f3(d.get("image_auroc")) if d else "--")
        rows.append(row)
    if any("--" not in (r[2],) for r in rows):
        w("controls", rows,
          ["category", "full $z$ img", "full $z$ pix", "grad", "lap", "var", "shuffle"],
          "Channel-content controls: the full prior map replaced by simpler derivative "
          "descriptors or a distribution-preserving shuffle (image AUROC; full-\\(z\\) "
          "column shows img/pix).", "tab:controls")

    # ---- 20. official-protocol arms (review P1) incl. audit extension
    A2 = load(f"{R}/all_results.json")
    gray = {c["tag"].replace("mvtec_", ""): c for c in
            A2.get("datasets", {}).get("MVTec AD", {}).get("categories", [])}
    camp_n = {"cashew": ("visa_cashew_pgdn_eval.json", "visa_cashew_draem_eval.json")}
    rows = []
    for cat in ("screw", "bottle", "capsule", "cashew", "toothbrush", "leather"):
        if cat in camp_n:
            camp_p = load(f"{RAW}/{camp_n[cat][0]}")
            camp_d = load(f"{RAW}/{camp_n[cat][1]}")
        elif cat in gray and "pgdn" in gray[cat]:
            camp_p = gray[cat]["pgdn"]
            camp_d = gray[cat].get("draem", {})
        else:
            camp_p = camp_d = {}
        off_p = load(f"{RAW}/offl_{cat}_pgdn_eval.json")
        off_d = load(f"{RAW}/offl_{cat}_draem_eval.json")
        rows.append([cat,
                     f3(camp_d.get("image_auroc")) if camp_d else "--",
                     f3(off_d.get("image_auroc")) if off_d else "--",
                     f3(camp_p.get("image_auroc")) if camp_p else "--",
                     f3(off_p.get("image_auroc")) if off_p else "--"])
    if any(r[2] != "--" for r in rows):
        w("official", rows,
          ["category", "\\draem{} ours (lr $10^{-3}$)", "\\draem{} reference-style",
           "PGDN ours", "PGDN reference-style"],
          "Reference-style recipe audit (lr $10^{-4}$, MultiStepLR, 700 epochs, external "
          "source), including the audit extension to the remaining below-chance "
          "categories (toothbrush, leather). A decomposition arm on screw (scheduled "
          "lr, original fractal source) gives twin $0.669$ / PGDN $0.923$.",
          "tab:official")

    # ---- 20b. KolektorSDD2 localization: ONE canonical record
    # (ksdd2_pixel_stats.json) supplies AUROC, AP and bootstrap CIs for
    # both arms and both operating points; text and tables must quote it.
    st = load(f"{RAW}/ksdd2_pixel_stats.json")
    if set(st) != {"gray-128", "rgb-256"}:
        raise SystemExit("ksdd2 localization statistics missing/incomplete: "
                         "results/raw/ksdd2_pixel_stats.json")
    kp_rows = []
    for point, label_p, label_d in (("gray-128", "PGDN (gray-128)",
                                     "\\draem{} (gray-128)"),
                                    ("rgb-256", "PGDN (RGB-256)",
                                     "\\draem{} (RGB-256)")):
        e = st[point]
        kp_rows.append([label_p, f3(e["pgdn_pixel_auroc"]), f3(e["pgdn_pixel_ap"])])
        kp_rows.append([label_d, f3(e["twin_pixel_auroc"]), f3(e["twin_pixel_ap"])])
    ci_g = st["gray-128"]["bootstrap"]["diff_ci95"]
    ci_r = st["rgb-256"]["bootstrap"]["diff_ci95"]
    w("ksdd2pixel", kp_rows,
      ["arm", "pix AUROC", "pix AP"],
      "KolektorSDD2 localization from the canonical paired evaluation "
      "(official GT masks; 1004 test images, 110 anomalous; identical "
      "seeded pixel indices for both arms --- 4096 per image at "
      "gray-128, 4.11M pooled, and 16384 at RGB-256, 16.45M pooled; AP "
      "uses the recall-increment convention of scikit-learn). "
      "PGDN-minus-twin pixel-AUROC difference, 95\\% image-cluster "
      f"bootstrap CIs: $[{ci_g[0]:+.3f},{ci_g[1]:+.3f}]$ at gray-128 "
      "(includes zero: a difference is not established, which is not "
      f"equivalence) and $[{ci_r[0]:+.3f},{ci_r[1]:+.3f}]$ at RGB-256, "
      "conditional on the tested checkpoints and protocol; pixel AP "
      "descriptively favours PGDN at both operating points, and both "
      "arms deteriorate absolutely at RGB-256.", "tab:ksdd2pixel")

    # ---- 20c. real-defect prior-response bridge (round-4 review Step 5)
    rb = load(f"{RAW}/real_prior_bridge.json")
    if rb:
        rows = []
        for cat, e in rb.get("categories", {}).items():
            for i, arm in enumerate(("base", "in", "cpl", "incpl")):
                rows.append([cat if i == 0 else "", arm,
                             str(e["n_defects"]) if i == 0 else "",
                             f3(e[arm]["weak"]), f3(e[arm]["strong"])])
        w("realbridge", rows,
          ["category", "arm", "$n$", "weak-prior img", "strong-prior img"],
          "Real MVTec defects split at the median of the prior's own mean "
          "response $z$ inside the ground-truth mask (independent of "
          "detector outcomes); image AUROC per group with all defect-free "
          "test images as negatives (seed-42 factorial arms). The "
          "synthetic coupling failure does not predict real harm on "
          "screw, where the coupling-only arm is best on both groups; "
          "bottle shows a real coupling cost on both groups.",
          "tab:realbridge")

    # ---- 20d. matched real-defect moderation (canonical 10k bootstrap)
    rm = load(f"{RAW}/real_moderation.json")
    rci = load(f"{RAW}/real_moderation_ci.json")
    if not rm or set(rm.get("categories", {})) != {"screw", "capsule", "bottle"}:
        raise SystemExit("real-moderation record missing/incomplete: "
                         "results/raw/real_moderation.json")
    if not rci:
        raise SystemExit("canonical moderation intervals missing: "
                         "results/raw/real_moderation_ci.json "
                         "(generate with moderation_bootstrap.py)")
    rows = []
    for cat in ("screw", "capsule", "bottle"):
        e = rm["categories"][cat]
        for i, I in enumerate(("off", "on")):
            cells = [cat if i == 0 else "", f"input {I}"]
            for s in ("42", "0", "1"):
                m = e["effects"][s][I]["M"]
                lo, hi = rci["categories"][cat][s][I]["ci95"]
                cells.append(f"{m:+.1f} [{lo:+.1f},{hi:+.1f}]")
            rows.append(cells)
    w("realmod", rows,
      ["category", "context", "$M$ s42", "$M$ s0", "$M$ s1"],
      "Matched coupling moderation on real defects: $M=E_{\\mathrm{weak}}"
      "-E_{\\mathrm{strong}}$, where $E_g$ is the change in group AUROC "
      "when coupling is enabled (reconstruction off; groups fixed at the "
      "seed-42 $z$ median; shared normal negatives). Nominal exploratory "
      "95\\% percentile CIs from the released paired image-cluster "
      "bootstrap (10{,}000 replicates, pooled multinomial image "
      "resampling, groups fixed; moderation\\_bootstrap.py). All "
      "eighteen point estimates are negative; only bottle's input-"
      "enabled intervals exclude zero in every seed --- screw and "
      "capsule remain directional observations whose intervals include "
      "zero.",
      "tab:realmod", span=1)

    # ---- 21. smooth-defect test (review loss analysis)
    sm = load(f"{RAW}/smooth_defect_test.json")
    if sm:
        rows = []
        for cat, e in sm.items():
            rows.append([cat,
                         f3(e.get("pgdn", {}).get("perlin")), f3(e.get("pgdn", {}).get("smooth")),
                         f3(e.get("draem", {}).get("perlin")), f3(e.get("draem", {}).get("smooth"))])
        w("smooth", rows, ["category", "\\shortstack{PGDN\\\\perlin}",
                           "\\shortstack{PGDN\\\\smooth}", "\\shortstack{\\draem{}\\\\perlin}",
                           "\\shortstack{\\draem{}\\\\smooth}"],
          "Weak-derivative defect test: image AUROC of both arms on matched standard "
          "(Perlin) versus smooth low-contrast synthetic defects over the same "
          "defect-free images.", "tab:smooth")

    # placeholder for any conditional table missing its data (compile safety)
    for name in ("official",):
        if not os.path.isfile(f"{OUT}/{name}.tex"):
            w(name, [["(pending GPU availability -- 700-epoch arms)"]],
              ["note"], "Official-protocol audit pending GPU availability.",
              f"tab:{name}")

    for name, content in PENDING.items():
        with open(f"{OUT}/{name}.tex", "w") as fh:
            fh.write(content)
    print("wrote", len(PENDING), "staged tables to", OUT,
          f"(directory holds {len(os.listdir(OUT))})")


if __name__ == "__main__":
    main()
