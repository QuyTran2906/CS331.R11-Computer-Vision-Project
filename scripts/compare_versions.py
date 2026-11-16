"""Report comparable SD1.5 experiments on a fixed manifest."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_baseline import load_evaluation, verified_jobs
from src.eval.visualize import comparison_grid
from src.models.sd15_ip_adapter import SD15_INPAINTING
from src.evaluation.reporting import aggregate, write_json
from src.evaluation.runner import read_mask, read_rgb
import numpy as np


ROLES = {"v1": (1, "sd15_direct"), "v2": (2, "oracle_mask"),
         "v3": (3, "oracle_mask"), "v3_no_ip": (3, "oracle_mask")}
FLAGS = ("use_micro_attributes", "use_ip_adapter", "use_latent_blending", "use_strict_critic")


def check_configs(configs):
    for key, cfg in configs.items():
        if (cfg.get("pipeline_version"), cfg.get("pipeline_mode")) != ROLES[key]:
            raise ValueError(f"Profile does not match role {key}")
        if (cfg.get("inpainting_model_id") != SD15_INPAINTING or cfg.get("seed") != 42
                or cfg.get("refinement_scheduler") != "ddim"
                or cfg.get("use_occluded_mask_input") is not True
                or cfg.get("text_query_policy", "empty") != "empty"):
            raise ValueError(f"Inconsistent experiment settings for {key}")
        if key == "v1" and any(cfg.get(flag) for flag in (*FLAGS, "enable_critic")):
            raise ValueError("V1 must use direct inpainting without guidance modules or a critic")
        if key != "v1":
            expected = ((True, key == "v3", True, True) if key.startswith("v3") else (False,) * 4)
            if tuple(cfg.get(flag) for flag in FLAGS) != expected or not cfg.get("enable_critic"):
                raise ValueError(f"Wrong feature flags for {key}")
    if "v3_no_ip" in configs:
        proposed, ablation = configs["v3"], configs["v3_no_ip"]
        if dict(proposed, use_ip_adapter=False) != ablation:
            raise ValueError("V3 ablation must differ only by use_ip_adapter")
    v2, v3 = configs["v2"], configs["v3"]
    if any(v2.get(k) != v3.get(k) for k in (set(v2) | set(v3)) - {*FLAGS, "pipeline_version"}):
        raise ValueError("V2 and V3 must share hyperparameters apart from the proposed feature flags")


def critic_summary(selected):
    counts, invalid = Counter(), 0
    reviewed = passed = 0
    for _, result, _ in selected.values():
        history = result.get("critique_history", [])
        if history:
            reviewed += 1
            passed += int(bool(result.get("critic_accepted")))
        for attempt in history:
            decision = attempt["final_decision"]
            counts["REJECT" if decision == "FAIL" else decision] += 1
            invalid += int(attempt.get("valid_json") is False)
    judged = counts["PASS"] + counts["REJECT"]
    return dict(attempt_decisions=dict(counts), invalid_json=invalid, reviewed_samples=reviewed,
                sample_pass_rate=passed / reviewed if reviewed else None,
                attempt_pass_rate=counts["PASS"] / judged if judged else None,
                attempt_reject_rate=counts["REJECT"] / judged if judged else None)


def compare_versions(records, profiles, output, fid_reference=None, device="cuda"):
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Expected a nonempty manifest with unique sample IDs")
    if set(profiles) not in ({"v1", "v2", "v3"}, set(ROLES)) or len(set(profiles.values())) != len(profiles):
        raise ValueError("Provide distinct profiles for V1, V2, V3 and optionally V3-no-IP")
    jobs, reports, configs = {}, {}, {}
    protocol = None
    for key, profile in profiles.items():
        directory, rows, settings = load_evaluation(profile, records)
        if protocol is not None and protocol != settings:
            raise ValueError("Metric protocols differ; rerun evaluation in the same runtime")
        protocol = settings
        if any(row.get("status") != "completed" for row in rows.values()):
            raise ValueError(f"Incomplete metrics for {key}")
        jobs[key] = verified_jobs(profile, records, directory, rows)
        reports[key] = rows
        configurations = [result["config"] for _, result, _ in jobs[key].values()]
        configs[key] = configurations[0]
        if any(cfg != configs[key] for cfg in configurations):
            raise ValueError(f"Mixed configs in {key}")
    check_configs(configs)
    paired = []
    for record in records:
        sid = record["sample_id"]
        base, _, hashes = jobs["v1"][sid]
        entry = {"sample_id": sid, "category": record.get("name"), "hidden_ratio": record.get("hidden_ratio")}
        for key in profiles:
            job, result, current = jobs[key][sid]
            for name in ("input.png", "visible.png", "occluded_input.png", "amodal.png"):
                read = read_rgb if name == "input.png" else read_mask
                if current[name] != hashes[name] and not np.array_equal(read(job / name), read(base / name)):
                    raise ValueError(f"Input or mask differs: {key}/{sid}/{name}")
            entry.update({f"{key}_{metric}": value for metric, value in reports[key][sid]["metrics"].items()})
            entry[f"{key}_seconds"] = result.get("seconds")
        paired.append(entry)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in paired for k in row)))
        writer.writeheader()
        writer.writerows(paired)
    summary = dict(profiles={k: p.name for k, p in profiles.items()}, configs=configs,
                   requested=len(records), protocol=protocol,
                   metrics={k: aggregate(list(rows.values()))["metrics"] for k, rows in reports.items()},
                   critic={k: critic_summary(jobs[k]) for k in profiles},
                   previews=comparison_grid(output, records, jobs),
                   fid=None)
    if fid_reference:
        from src.eval.fid import evaluate_fid
        summary["fid"] = evaluate_fid(records, jobs, fid_reference, device)
    write_json(output / "comparison.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["version", "metric", "mean", "std", "n"])
        for key, metrics in summary["metrics"].items():
            for metric, stat in metrics.items():
                writer.writerow([key, metric, stat["mean"], stat["std"], stat["n"]])
            critic = summary["critic"][key]
            for metric in ("attempt_pass_rate", "attempt_reject_rate", "sample_pass_rate"):
                n = critic["reviewed_samples"] if metric.startswith("sample") else sum(
                    critic["attempt_decisions"].get(decision, 0) for decision in ("PASS", "REJECT"))
                writer.writerow([key, metric, critic[metric], "", n])
            if summary["fid"]:
                writer.writerow([key, "FID_192", summary["fid"]["scores"][key], "", len(records)])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=role != "v3_no_ip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fid-reference", type=Path, help="Folder of clean, isolated, unoccluded RGB crops")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    profiles = {key: getattr(args, key) for key in ROLES if getattr(args, key)}
    compare_versions(records, profiles, args.output, args.fid_reference, args.device)
    print("Comparison:", args.output)


if __name__ == "__main__":
    main()
