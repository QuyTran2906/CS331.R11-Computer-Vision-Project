"""V2 notebook cells, kept separate from the unchanged V1/SD2 workflow."""


def append_v2_cells(md, code):
    md("## 16. V2 SD1.5 + IP-Adapter Plus — chọn subset\nV2 bật micro-attributes, IP-Adapter Plus, latent blending và strict critic. Baseline mới là `sd15_direct.json`; SD2 Direct và V1 giữ nguyên ở ô 4–15. Mặc định thử 3 mẫu đầu của dev50, sau đó đặt `V2_LIMIT=50` và chạy lại 16–19. Đổi sang test400 chỉ sau khi chốt dev: `V2_LIMIT=400`, `V2_FROZEN=True`. Toàn pipeline còn Qwen/Pix2Gestalt/CLIP ViT-H: không suy ra T4 RAM thấp sẽ đủ chỉ từ backbone SD1.5. Xem `docs/PIPELINE_V2.md`.")
    code('''from PIL import Image
from IPython.display import display
import json
V2_CONFIG = "configs/oracle_mask_v2.json"
SD15_CONFIG = "configs/sd15_direct.json"
V2_BATCH = "dev50"
V2_LIMIT = 3
V2_FROZEN = False
v2_sizes = {"dev50": 50, "test400": 400}
if V2_BATCH not in v2_sizes or not 1 <= V2_LIMIT <= v2_sizes[V2_BATCH]:
    raise ValueError("Chọn dev50 (1..50) hoặc test400 (400)")
if V2_BATCH == "test400" and (not V2_FROZEN or V2_LIMIT != 400):
    raise ValueError("Chốt cấu hình trên dev rồi đặt V2_FROZEN=True và V2_LIMIT=400")
V2_EXTRA = ["--allow-test"] if V2_BATCH == "test400" else []
sh(PYTHON, "scripts/check_v2_environment.py", "--config", V2_CONFIG, "--check-access")
source_manifest = PROJECT / f"outputs/{V2_BATCH}/manifest.json"
if not source_manifest.is_file():
    sh(PYTHON, "scripts/prepare_data.py", "--config", f"configs/data_{V2_BATCH}.json", *V2_EXTRA)
v2_all_records = json.loads(source_manifest.read_text())
if len(v2_all_records) != v2_sizes[V2_BATCH]:
    raise ValueError("Manifest không đúng số mẫu đã chuẩn bị")
v2_records = v2_all_records[:V2_LIMIT]
# Same parent preserves relative paths; generation resume uses sample content, not manifest filename.
V2_MANIFEST = source_manifest if V2_LIMIT == len(v2_all_records) else source_manifest.with_name(f"v2_first{V2_LIMIT}.json")
if V2_MANIFEST != source_manifest:
    V2_MANIFEST.write_text(json.dumps(v2_records, ensure_ascii=False, indent=2), encoding="utf-8")
v2_config = json.loads((PROJECT / V2_CONFIG).read_text())
V2_RUN_FILE = PROJECT / f"outputs/v2_runs/{V2_BATCH}_{v2_config['pipeline_mode']}_{V2_LIMIT}.json"
SD15_RUN_FILE = PROJECT / f"outputs/v2_runs/{V2_BATCH}_sd15_direct_{V2_LIMIT}.json"
for config in (SD15_CONFIG, V2_CONFIG):
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", V2_MANIFEST,
       "--config", config, "--limit", str(V2_LIMIT), "--validate-only", *V2_EXTRA)
print(json.dumps(v2_config, indent=2))
print("Selected:", V2_LIMIT, "| Max total attempts:", v2_config["max_critic_iter"])
''')
    md("## 17. Chạy baseline SD1.5 Direct\nChỉ SD1.5 Inpainting: prompt chung cố định, strength=1, 28 bước, guidance=7, seed=42, DDIM. Không Pix2Gestalt/Qwen/IP-Adapter/blending. Dùng đúng các mẫu và mask ở ô 16; cần sinh mới vì kết quả SD2 cũ không phải baseline SD1.5. Tự backup và resume. Tăng từ 3 lên 50 mẫu với cùng code/config/môi trường sẽ giữ lại 3 mẫu đã sinh.")
    code('''sh(PYTHON, "scripts/run_experiment.py", "--manifest", V2_MANIFEST,
   "--config", SD15_CONFIG, "--limit", str(V2_LIMIT), "--backup-every", "5",
   "--backup-dir", str(DRIVE / "results"), "--profile-output", SD15_RUN_FILE, *V2_EXTRA)
SD15_PROFILE = json.loads(SD15_RUN_FILE.read_text())["profile"]
print("SD1.5 Direct profile:", SD15_PROFILE)
''')
    md("## 18. Chạy Main V2 SD1.5 + IP-Adapter Plus\nNạp adapter và CLIP ViT-H lần đầu có thể mất thời gian. Image prompt chỉ chứa pixel visible của target trên nền trắng. `max_critic_iter=3` là tối đa 3 lần sinh tổng cộng (2 retries). Sai JSON được yêu cầu định dạng lại một lần; vẫn sai thì FAIL. Nếu hết lượt mà không PASS, giữ ứng viên tốt nhất và ghi `exhausted_retries`. Điểm critic là đánh giá tính hợp lý, không phải ground truth.")
    code('''sh(PYTHON, "scripts/run_experiment.py", "--manifest", V2_MANIFEST,
   "--config", V2_CONFIG, "--limit", str(V2_LIMIT), "--backup-every", "5",
   "--backup-dir", str(DRIVE / "results"), "--profile-output", V2_RUN_FILE, *V2_EXTRA)
V2_PROFILE = json.loads(V2_RUN_FILE.read_text())["profile"]
print("V2 profile:", V2_PROFILE)
v2_root = PROJECT / "outputs/runs" / V2_PROFILE
active_ids = {r["sample_id"] for r in v2_records}
log_rows = [json.loads(line) for line in (v2_root / "critic_logs.jsonl").read_text().splitlines() if line.strip()]
log_rows = [row for row in log_rows if row["sample_id"] in active_ids]
from collections import Counter
print("Selected outcomes:", dict(Counter(row["sample_outcome"] for row in log_rows if row["selected"])))
print("Critic log:", v2_root / "critic_logs.jsonl")
for path in [p for p in sorted(v2_root.glob("*/*/preview.jpg")) if p.parent.parent.name in active_ids][:5]:
    display(Image.open(path))
''')
    md("## 19. Chấm và so sánh SD1.5 Direct → V2\nCùng manifest, ảnh, visible/hidden mask, seed, checkpoint và scheduler. Đây là **so sánh toàn hệ thống**, ngân sách sinh và hậu xử lý khác nhau. Đọc paired metrics và tối đa 20 mẫu trực quan cố định, không chọn ảnh theo điểm critic. COCO-A không có hidden RGB GT; các metric visible gần hoàn hảo không đo chất lượng phần sinh. Những profile SD2 Direct/V1 cũ giữ nguyên, xem ô 14–15; không ghép chúng vào bảng cùng backbone này.")
    code('''if v2_config["pipeline_mode"] != "oracle_mask":
    raise ValueError("So sánh cùng mask yêu cầu oracle_mask_v2; improved_v2 cần đánh giá blind riêng")
SD15_PROFILE = json.loads(SD15_RUN_FILE.read_text())["profile"]
V2_PROFILE = json.loads(V2_RUN_FILE.read_text())["profile"]
for profile in (SD15_PROFILE, V2_PROFILE):
    sh(PYTHON, "scripts/evaluate_runs.py", "--profile", profile,
       "--manifest", V2_MANIFEST, "--device", "cuda", "--backup-dir", str(DRIVE / "results"))
V2_COMPARE_DIR = PROJECT / "outputs/comparisons" / f"{V2_BATCH}_{V2_LIMIT}_{SD15_PROFILE}_vs_{V2_PROFILE}"
sh(PYTHON, "scripts/compare_baseline.py", "--manifest", V2_MANIFEST,
   "--baseline", SD15_PROFILE, "--main-profile", V2_PROFILE,
   "--comparison-kind", "direct", "--baseline-label", "SD1.5 Direct", "--main-label", "V2 SD1.5 + IP",
   "--output-dir", V2_COMPARE_DIR, "--preview-count", "20")
comparison = json.loads((V2_COMPARE_DIR / "comparison.json").read_text())
for name in comparison["previews"]:
    display(Image.open(V2_COMPARE_DIR / name))
archive = Path(shutil.make_archive(str(V2_COMPARE_DIR), "zip", V2_COMPARE_DIR))
destination = DRIVE / "results/comparisons"
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(archive, destination / archive.name)
print("Saved comparison:", destination / archive.name)
''')
    md("## 20. Tùy chọn: ablation IP-Adapter trên cùng V2\nMặc định bỏ qua. Đặt `RUN_NO_IP_ABLATION=True` để chạy V2 chỉ tắt `use_ip_adapter`, rồi so với V2 đầy đủ. Những tham số khác phải giống hệt. So sánh này mới kiểm tra đóng góp adapter trong pipeline; số retries thực tế có thể khác vì critic quyết định. Tinh chỉnh trên dev; test chỉ chạy với cấu hình đã chốt. Không cần chạy lại V2 đầy đủ nếu đã có kết quả cùng code/config/môi trường.")
    code('''RUN_NO_IP_ABLATION = False
if RUN_NO_IP_ABLATION:
    NO_IP_CONFIG = "configs/oracle_mask_v2_no_ip.json"
    no_ip = json.loads((PROJECT / NO_IP_CONFIG).read_text())
    changed_keys = {k for k in set(no_ip) | set(v2_config) if no_ip.get(k) != v2_config.get(k)}
    if changed_keys != {"use_ip_adapter"} or no_ip["use_ip_adapter"]:
        raise ValueError("Ablation phải chỉ thay đổi use_ip_adapter so với config V2 đã chọn")
    NO_IP_RUN_FILE = PROJECT / f"outputs/v2_runs/{V2_BATCH}_no_ip_{V2_LIMIT}.json"
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", V2_MANIFEST,
       "--config", NO_IP_CONFIG, "--limit", str(V2_LIMIT), "--backup-every", "5",
       "--backup-dir", str(DRIVE / "results"), "--profile-output", NO_IP_RUN_FILE, *V2_EXTRA)
    NO_IP_PROFILE = json.loads(NO_IP_RUN_FILE.read_text())["profile"]
    V2_PROFILE = json.loads(V2_RUN_FILE.read_text())["profile"]
    for profile in (NO_IP_PROFILE, V2_PROFILE):
        sh(PYTHON, "scripts/evaluate_runs.py", "--profile", profile,
           "--manifest", V2_MANIFEST, "--device", "cuda", "--backup-dir", str(DRIVE / "results"))
    output = PROJECT / "outputs/comparisons" / f"{V2_BATCH}_{V2_LIMIT}_ablation_ip_{NO_IP_PROFILE}_vs_{V2_PROFILE}"
    sh(PYTHON, "scripts/compare_baseline.py", "--manifest", V2_MANIFEST,
       "--baseline", NO_IP_PROFILE, "--main-profile", V2_PROFILE,
       "--comparison-kind", "mask_given", "--baseline-label", "V2 no IP", "--main-label", "V2 with IP",
       "--output-dir", output, "--preview-count", "20")
    summary = json.loads((output / "comparison.json").read_text())
    for name in summary["previews"]:
        display(Image.open(output / name))
    archive = Path(shutil.make_archive(str(output), "zip", output))
    destination = DRIVE / "results/comparisons"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, destination / archive.name)
    print("Ablation saved:", destination / archive.name)
else:
    print("Bỏ qua ablation; chạy 16–19 trước để kiểm tra pipeline và baseline mới.")
''')
