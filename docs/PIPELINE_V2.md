# Pipeline V2 — SD2, micro-attributes, latent blending và strict critic

## Phạm vi đã chốt

V2 dùng **SD1.5 Inpainting + IP-Adapter Plus**. Bốn thành phần được bật:
Qwen JSON micro-attributes, IP-Adapter Plus, latent blending trong DDIM và strict
JSON critic. Backbone là `stable-diffusion-v1-5/stable-diffusion-inpainting`,
mirror hiện có của model RunwayML cũ. Adapter chính thức là
`h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors`, dùng patch embeddings
của CLIP ViT-H. Đây là điều kiện hóa thị giác, không bảo đảm sao chép đúng texture.
SD2 không tương thích checkpoint adapter này; preflight sẽ từ chối cấu hình sai.

Baseline cũ giữ nguyên:

| Nhóm | Config | Hành vi |
|---|---|---|
| SD2 Direct | `configs/sd2_direct.json` | Prompt chung cố định, mask ẩn có sẵn, một lần sinh, không critic |
| V1 mask-given | `configs/oracle_mask_v1.json` | Bản sao nguyên giá trị của `oracle_mask.json` |
| V1 blind | `configs/improved_v1.json` | Bản sao nguyên giá trị của `improved.json` |
| SD1.5 Direct | `configs/sd15_direct.json` | SD1.5 + DDIM, prompt cố định, một lần sinh, không adapter/critic |
| V2 no IP | `configs/oracle_mask_v2_no_ip.json` | Chỉ tắt adapter so với V2 đầy đủ, dùng cho ablation |
| V2 mask-given | `configs/oracle_mask_v2.json` | Nhận visible + hidden mask, chỉ sinh RGB vùng ẩn |
| V2 blind | `configs/improved_v2.json` | Chỉ nhận RGB + visible, tự dự đoán amodal mask |

Không đổi config SD2 Direct sang dùng nhãn annotation: baseline đã công bố số liệu
dùng prompt chung, không lấy tên lớp. V1 đã có xử lý màu/frequency lock theo prior
và target-aware critic; không gọi V1 là pipeline “không giữ texture”. Các module
`oracle_completer.py`, `improved_completer.py`, `sd2_refiner.py` và
`sd2_direct_completer.py` được tái sử dụng; baseline mới chỉ thay backbone, scheduler và nhãn log, thuật toán direct giữ nguyên.

## Luồng V2

1. **Qwen3-VL trước khi sinh prior:** nhận crop ngữ cảnh đánh dấu target và crop
   chỉ có pixel nhìn thấy; trả JSON `object_name`, `primary_color`,
   `material_and_texture`, `completion_description`. Thuộc tính không chắc chắn
   dùng `unknown`. Sai JSON được yêu cầu định dạng lại một lần; vẫn sai thì ghi lỗi
   và dùng prompt chung, không chuyển văn bản lỗi thành prompt tự do.
2. **Pix2Gestalt + CarveKit:** giữ hàm sinh prior của V1. Nhánh mask-given dùng
   amodal mask được cho để lấy ROI; nhánh blind tự dự đoán mask. RGB prior, alpha,
   khử nhiễm occluder và tạo seed canvas vẫn là xử lý 2D, không phải mô hình 3D.
3. **CLIP kiểm tra danh tính:** giữ kiểm tra prompt của V1. Nếu CLIP chấp nhận Qwen
   và JSON hợp lệ, giữ prompt gồm micro-attributes, kể cả khi CLIP thích câu tên
   vật thể ngắn hơn. Nếu không, dùng fallback đã ghi trong `prompt_selection.json`.
4. **Hybrid SD1.5 ROI + IP-Adapter Plus:** image prompt chỉ chứa crop visible của target trên nền trắng; không đưa pixel occluder vào image prompt. Adapter được nạp thật vào attention, scale mặc định 0.5. Phần refinement giữ crop vuông, mask mở rộng gần biên và khóa màu/tần số theo
   prior. V2 dùng **DDIM** với cùng scheduler cho cả ablation bật/tắt blending.
   Trong callback sau mỗi denoising step:

   ```text
   z_reference(t_prev) = add_noise(encode(clean_seed), fixed_noise, t_prev)
   z_next = M_edit * z_generated + (1 - M_edit) * z_reference(t_prev)
   ```

   `t_prev` là mức nhiễu sau bước DDIM, không phải timestep trước bước đó. Bước cuối
   dùng latent sạch. Mask được max-pool về kích thước latent để giữ các vùng sửa
   nhỏ; reference là canvas gồm visible + prior + nền trắng, không chứa RGB gốc
   của vật che tại vùng ẩn. Noise tham chiếu dùng generator riêng có seed cố định.
   VAE vẫn có thể gây hiệu ứng ở biên; blending không bảo đảm triệt tiêu seam.
   Sau sinh, các pixel visible được chép lại chính xác và alpha theo amodal mask.
5. **Strict Qwen critic:** nhận bốn crop cùng hệ tọa độ: ảnh gốc, target visible,
   completion, bản đồ visible/hidden. JSON gồm `color_match`,
   `structure_continuity`, `final_decision`, `reason`, `repair_instruction`.
   Chỉ PASS khi model trả PASS **và cả hai điểm đạt ngưỡng** (mặc định 7).
   Điểm tổng hợp là min của hai tiêu chí. JSON sai không được gán điểm chất lượng 0
   hoặc coi là PASS; giá trị điểm là null và được ghi nhận riêng.
6. **Thử lại có giới hạn:** `max_critic_iter=3` nghĩa là **tổng cộng 3 lần sinh**,
   tương đương tối đa 2 retries. Đổi seed `42, 1051, 2060`, tăng guidance theo
   `retry_guidance_step`, dùng repair instruction dương tính trong prompt.
   Strength giữ theo mức thiếu prior, không tăng theo lượt thử. Dừng ở PASS đầu
   tiên. Nếu hết lượt, chọn ứng viên có điểm hợp lệ cao nhất; nếu tất cả JSON lỗi,
   giữ ứng viên đầu và ghi rõ `exhausted_retries`. Không biến FAIL thành PASS chỉ
   vì điểm cao. VLM critic đánh giá tính hợp lý, không xác minh RGB vùng khuất.

V2 còn thay đổi thứ tự reasoning/prior, prompt selection, scheduler và cách retry.
Vì vậy **SD1.5 Direct so với V2 là so sánh hệ thống**, không phải bằng chứng riêng cho từng
thành phần. V1 SD2 và V2 còn khác backbone; không quy chênh lệch cho adapter. Config `oracle_mask_v2_no_ip.json` chỉ thay cờ adapter và ô 20 chạy so sánh này. Có thể tắt các cờ trên cùng cấu hình V2 để làm ablation; tắt
`enable_critic` chạy một lần, `use_strict_critic=false` dùng critic V1. Giữ cùng
seed, scheduler, tập mẫu và backbone khi so một cờ. Không dùng critic vốn chọn ảnh
làm tiêu chí duy nhất để chứng minh chất lượng tăng.

## Chạy Colab

Phiên mới: **1 → 2 → 3 → 16 → 17 → 18 → 19** trong
`notebooks/Amodal_Completion_Colab.ipynb`. Nếu phiên đã setup đúng các version,
chạy ô 2 để cập nhật, sau đó 16–19. Nếu ô 16 báo sai API/version, chạy lại ô 3.
Không cần đổi sang `diffusers==0.27.2`: vẫn ghim `diffusers==0.37.1`,
`transformers==5.0.0`; preflight kiểm tra trước khi tải model.

- Ô 16: mặc định `V2_BATCH="dev50"`, `V2_LIMIT=3`,
  `V2_CONFIG="configs/oracle_mask_v2.json"` để kiểm tra GPU trước.
- Sau khi ba mẫu chạy được, đặt `V2_LIMIT=50` và chạy lại 16–19. Cùng
  code/config/môi trường thì tự resume ba mẫu; không đổi seed giữa chừng.
- Ô 17 sinh baseline SD1.5 Direct mới; không thể dùng ảnh SD2 cũ thay thế.
- Ô 18 sinh V2; ô 19 chấm và so sánh cùng backbone/mask/scheduler.
- Ô 20 tùy chọn ablation chỉ tắt IP-Adapter. Mặc định bỏ qua.
- V1 `a0ab44af3e253863` và SD2 Direct `48163056f1ee9fc7` vẫn được đọc ở
  ô 14–15, không sinh lại. Script so sánh từ chối ghép checkpoint khác nhau.
- Khi chốt dev, đặt `V2_BATCH="test400"`, `V2_LIMIT=400`, `V2_FROZEN=True`.
  Test400 cũ đã được xem trong quá trình phát triển; báo cáo cần nói rõ điều này,
  không gọi nó là test chưa từng quan sát.
- V2 blind có thể chạy ô 16 rồi 18 bằng `improved_v2.json`. Không đưa nó vào bảng
  “cùng mask cho sẵn” ở ô 19; đầu vào và bài toán khác nhau.

Model vẫn nạp lazy và chỉ một stage ở GPU. CPU RAM giữ cache các model;
runtime 12.7 GiB RAM từng bị SIGKILL với V1. Plus thêm image encoder ViT-H; đổi sang SD1.5 không bảo đảm toàn pipeline chạy được trên T4 RAM thấp. Nên dùng runtime high-RAM đã chạy được
V1 và kiểm tra smoke trước. Thử lại có thể làm V2 chậm hơn V1; không hứa tốc độ
hay chất lượng khi chưa chạy foundation models trên Colab.

## Profile, log, backup

Thêm code làm **code hash thay đổi**, nên lần sinh mới có profile mới dù dùng
config V1. Đây là provenance, không phải thay đổi tên hoặc ghi đè profile cũ.
Notebook dùng ID cụ thể để đọc các baseline đã có, không chọn “profile mới nhất”.

Trong `outputs/runs/<profile>/<sample>/<signature>/`:

- `micro_attributes.json`: JSON và raw responses/lỗi parser.
- `visible_reference.png`, `prior_rgb.png`, `prompt_selection.json`.
- `candidate_01.png`, `candidate_02.png`, ...: từng ảnh sinh.
- `critic_input_01.png`, ...: panel thực sự đưa cho critic.
- `critic_logs.jsonl`, `critique_history.json`: seed, prompt, score, decision,
  reason, repair, thời gian và lịch timestep của từng lần sinh.
- `v2_selection.json`: ứng viên được chọn, số lần thử, accepted/exhausted.
- `completion.png`, `result.json`: output cuối và checksum; completed chỉ nghĩa
  là chương trình tạo được kết quả, **không đồng nghĩa critic PASS**.

`outputs/runs/<profile>/critic_logs.jsonl` là log gọn tổng hợp batch, khử trùng
khi resume. Raw JSON và các lần thử của mẫu bị exception vẫn ở thư mục mẫu.
Backup Drive mỗi 5 mẫu và cuối batch chứa toàn bộ log/ứng viên; bảng paired và
trang preview được ô 19 nén riêng trong `results/comparisons`.

Metric giữ protocol hiện tại: CLIP là độ phù hợp lớp, seam/histogram là proxy;
LPIPS/SSIM/PSNR/feature similarity đo bảo toàn visible. COCO-A không có hidden RGB
ground truth. Cần đọc mẫu ngẫu nhiên cố định, lỗi texture/anatomy/text/residual
occluder và thống kê exhausted/invalid JSON; không đánh giá V2 chỉ bằng critic.

## Kiểm tra và nguồn API

```bash
python -m unittest discover -s tests -v
python scripts/check_v2_environment.py --config configs/oracle_mask_v2.json
```

`test_v2_latents.py` dùng tensor DDIM và một inpainting pipeline rất nhỏ với
trọng số ngẫu nhiên để kiểm tra callback; không tải checkpoint. Đó là kiểm tra
API/shape/timestep, **không phải kiểm chứng chất lượng SD1.5/IP-Adapter/Qwen trên GPU**.

- [IP-Adapter model card chính thức](https://huggingface.co/h94/IP-Adapter)
- [Diffusers inpaint v0.37.1](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py)
- [DDIM step v0.37.1](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/schedulers/scheduling_ddim.py)

Cấu hình đầy đủ đã bật đường nạp checkpoint Plus chính thức; kiểm thử bằng
network nhỏ và trọng số ngẫu nhiên xác nhận đường gọi image encoder, patch projection,
IP attention và callback. Vẫn phải chạy smoke trên Colab với checkpoint thật để xác
nhận bộ nhớ và chất lượng. Không khẳng định Plus khắc phục hoàn toàn lỗi texture.
