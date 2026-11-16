# Phạm vi: Ô 4: chọn và xuất tập mẫu COCO-A cố định để mọi phiên bản xử lý đúng cùng đối tượng.
"""Audit COCO-A and export reproducible object subsets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.cocoa_loader import CocoaDataset, audit, export_samples


# Chọn config split, bảo vệ test và ghi manifest/preview sau khi audit.
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data_dev50.json")
    parser.add_argument("--verify-hashes", action="store_true", help="Read every JPG and verify against its manifest")
    parser.add_argument("--allow-test", action="store_true", help="Prepare final test subset only after freezing settings")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root, out = ROOT / cfg["dataset_root"], ROOT / cfg["output_dir"]
    if cfg["split"] not in ("train", "dev", "test") or not 1 <= cfg["limit"] <= 10000:
        raise ValueError("Use train/dev/test and a positive limit <= 10000")
    if cfg["split"] == "test" and not args.allow_test:
        raise ValueError("Test selection requires --allow-test after freezing configuration")
    old_config = out / "selection_config.json"
    if old_config.is_file() and json.loads(old_config.read_text(encoding="utf-8")) != cfg:
        raise ValueError("Output directory already belongs to a different selection; choose a new output_dir")
    report = audit(root, args.verify_hashes)
    out.mkdir(parents=True, exist_ok=True)
    records = export_samples(CocoaDataset(root, cfg["split"]), cfg, out)
    report["selected_targets"] = len(records)
    report["selected_unique_images"] = len({r["image_id"] for r in records})
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "selection_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} {cfg['split']} objects. Output: {out}", flush=True)
    print("Blind input: image + visible. Mask-given input: image + visible + gt_hidden. GT RGB for hidden regions is unavailable.")


if __name__ == "__main__":
    main()
