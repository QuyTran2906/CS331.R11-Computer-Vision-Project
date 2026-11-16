"""Generate the runnable notebook with exactly eight sequential cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
'''# 1. Môi trường và Google Drive
import subprocess, sys, os, shutil, zipfile, json, base64, re
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
def sh(*cmd, extra_env=None):
    env = {
        **os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
    }
    env.update(extra_env or {})
    with subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env) as p:
        for line in p.stdout:
            # Older Pix2Gestalt/CarveKit tqdm bars may ignore TQDM_DISABLE.
            # Filter only progress-bar lines; keep status messages and errors.
            if re.search(r"[0-9]{1,3}%[|].*[|]", line):
                continue
            print(line, end="", flush=True)
        if p.wait() != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
sh("nvidia-smi")
from google.colab import drive
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/CS331") if Path("/content/drive/MyDrive/CS331").is_dir() else Path("/content/drive/MyDrive/amodal_project")
(DRIVE / "results").mkdir(parents=True, exist_ok=True)
print("DRIVE:", DRIVE)
for secret_name in ("HF_TOKEN", "GITHUB_TOKEN"):
    try:
        from google.colab import userdata
        val = userdata.get(secret_name)
        if val: os.environ[secret_name] = val
    except Exception:
        pass
''',
'''# 2. Đồng bộ mã và dữ liệu
REPO_URL = "https://github.com/tbaouit/CS331.R11-Computer-Vision-Project.git"
PROJECT = Path("/content/cs331")
PYTHON = str(PROJECT / ".venv-gpu/bin/python")

git_env = {}
if os.environ.get("GITHUB_TOKEN"):
    auth = base64.b64encode(("x-access-token:" + os.environ["GITHUB_TOKEN"]).encode()).decode()
    git_env = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
               "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + auth}
if (PROJECT / ".git").is_dir():
    sh("git", "-C", PROJECT, "pull", "--ff-only", "origin", "main", extra_env=git_env)
elif PROJECT.exists() and any(PROJECT.iterdir()):
    raise RuntimeError("Move the existing non-Git /content/cs331 directory aside before cloning")
else:
    sh("git", "clone", "--branch", "main", REPO_URL, PROJECT, extra_env=git_env)

os.chdir(PROJECT)
sh("git", "log", "-n", "1", "--oneline")

if not (PROJECT / "dataset/cocoa/dataset.json").is_file():
    candidates = [
        DRIVE / "cocoa_dataset.zip",
        DRIVE / "data/cocoa_dataset.zip",
        Path("/content/drive/MyDrive/CS331/cocoa_dataset.zip"),
        Path("/content/drive/MyDrive/amodal_project/cocoa_dataset.zip"),
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Upload cocoa_dataset.zip vào {DRIVE}")
    local_zip = Path("/content/cocoa_dataset.zip")
    print(f"Copying {source} -> {local_zip} ...", flush=True)
    shutil.copy2(source, local_zip)
    destination = PROJECT / "dataset"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as bundle:
        for item in bundle.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        bundle.extractall(destination)
    local_zip.unlink()
    print("Dataset extracted to", destination / "cocoa", flush=True)
else:
    print("Dataset already extracted at", PROJECT / "dataset/cocoa", "(skipped)", flush=True)

print("Free GiB:", round(shutil.disk_usage(PROJECT).free / 1024**3, 1))
''',
'''# 3. Cài thư viện, tải trọng số và kiểm thử
sh(sys.executable, "scripts/setup_colab.py")
sh(PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v")
''',
'''# 4. Chuẩn bị cùng tập mẫu cho mọi phiên bản
BATCH = "dev50"  # Switch to "test400" only after selecting the final settings.
FINAL_TEST = False
SMOKE_LIMIT = None  # Optional: use 3 for a first check on dev50.
sizes = {"dev50": 50, "test400": 400}
if BATCH not in sizes:
    raise ValueError("BATCH must be dev50 or test400")
if BATCH == "test400" and (not FINAL_TEST or SMOKE_LIMIT is not None):
    raise ValueError("test400 requires FINAL_TEST=True and no smoke limit")
N = sizes[BATCH] if SMOKE_LIMIT is None else SMOKE_LIMIT
if type(N) is not int or not 1 <= N <= sizes[BATCH]:
    raise ValueError("Invalid smoke limit")
extra = ["--allow-test"] if BATCH == "test400" else []
manifest = PROJECT / "outputs" / BATCH / "manifest.json"
if not manifest.is_file():
    sh(PYTHON, "scripts/prepare_data.py", "--config", f"configs/data_{BATCH}.json", *extra)
records = json.loads(manifest.read_text(encoding="utf-8"))
if len(records) != sizes[BATCH] or any(r["split"] != BATCH.rstrip("0123456789") for r in records):
    raise ValueError("Manifest does not match the requested split or size")
if N < len(records):
    manifest = manifest.with_name(f"first_{N}.json")
    manifest.write_text(json.dumps(records[:N], ensure_ascii=False, indent=2), encoding="utf-8")
configs = dict(v1="v1_sd15_direct", v2="v2_sd15_pipeline",
               v3="v3_sd15_proposed")
sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
   "--config", "configs/v1_sd15_direct.json", "--validate-only", *extra)

def marker(role):
    return PROJECT / "outputs" / "experiment_profiles" / f"{BATCH}_{N}_{role}.json"

def run_version(role, config=None):
    config = config or f"configs/{configs[role]}.json"
    sh(PYTHON, "scripts/run_experiment.py", "--manifest", manifest, "--limit", N,
       "--config", config, "--backup-dir", DRIVE / "results",
       "--profile-output", marker(role), *extra)
''',
'''# 5. Chạy V1: SD1.5 Direct
run_version("v1")
''',
'''# 6. Chạy V2: pipeline vòng kín cơ bản
run_version("v2")
''',
'''# 7. Chạy V3 và tùy chọn ablation
RUN_NO_IP_ABLATION = False
run_version("v3")
if RUN_NO_IP_ABLATION:
    ablation_path = marker("v3_no_ip").with_name("v3_no_ip_config.json")
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    proposed = json.loads((PROJECT / "configs/v3_sd15_proposed.json").read_text(encoding="utf-8"))
    ablation_path.write_text(json.dumps(dict(proposed, use_ip_adapter=False), indent=2), encoding="utf-8")
    run_version("v3_no_ip", ablation_path)
''',
'''# 8. Đánh giá, so sánh và lưu báo cáo
from PIL import Image
from IPython.display import display
FID_REFERENCE = None  # For test400, set to a Drive folder of >=400 clean object crops.
roles = ["v1", "v2", "v3"] + (["v3_no_ip"] if RUN_NO_IP_ABLATION else [])
profiles = {role: json.loads(marker(role).read_text(encoding="utf-8"))["profile"] for role in roles}
for profile in profiles.values():
    sh(PYTHON, "scripts/evaluate_runs.py", "--profile", profile, "--manifest", manifest,
       "--device", "cuda", "--backup-dir", DRIVE / "results")
output = PROJECT / "outputs" / "comparisons" / (
    f"{BATCH}_{N}_" + "_".join(profiles.values()))
args = [arg for role, profile in profiles.items()
        for arg in ("--" + role.replace("_", "-"), PROJECT / "outputs" / "runs" / profile)]
fid_args = ["--fid-reference", FID_REFERENCE] if FID_REFERENCE else []
sh(PYTHON, "scripts/compare_versions.py", "--manifest", manifest, "--output", output, *args, *fid_args)
report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
for role in roles:
    relevant = {k: v["mean"] for k, v in report["metrics"][role].items()
                if k in ("CLIP_score", "seam_mae", "color_hist_sim")}
    print(role, relevant, "critic:", report["critic"][role])
if report["fid"]:
    print("FID:", report["fid"])
for name in report["previews"]:
    display(Image.open(output / name))
backup = DRIVE / "results" / "comparisons"
backup.mkdir(parents=True, exist_ok=True)
archive = Path(shutil.make_archive(str(output), "zip", output))
shutil.copy2(archive, backup / archive.name)
print("CSV, JSON and previews:", backup / archive.name)
'''
]


def main():
    cells = []
    for index, source in enumerate(CELLS, 1):
        compile(source, f"cell_{index}", "exec")
        cells.append(dict(cell_type="code", id=f"amodal-{index:02d}", metadata={},
                          source=source.splitlines(keepends=True), outputs=[], execution_count=None))
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator="GPU", kernelspec=dict(name="python3", display_name="Python 3")))
    target = ROOT / "notebooks/Amodal_Completion_Colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
