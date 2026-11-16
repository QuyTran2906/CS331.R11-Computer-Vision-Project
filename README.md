# CS331 — Amodal Appearance Completion trên COCO-A

Ba phiên bản nhận cùng ảnh RGB, visible mask và mask vùng khuất cho sẵn. Chúng
dùng chung Stable Diffusion 1.5 Inpainting, DDIM và seed 42. Mục tiêu là sinh
diện mạo hợp lý trong vùng khuất; hình dạng đầu ra được cố định bởi mask.

| Phiên bản | Cấu hình | Luồng |
| --- | --- | --- |
| V1 Direct | `configs/v1_sd15_direct.json` | Prompt cố định, một lượt inpainting, giữ nguyên pixel nhìn thấy |
| V2 Pipeline | `configs/v2_sd15_pipeline.json` | Qwen cơ bản → Pix2Gestalt/CarveKit → CLIP → tinh chỉnh SD1.5 → critic tổng hợp |
| V3 Proposed | `configs/v3_sd15_proposed.json` | Qwen micro-attributes, IP-Adapter Plus, latent blending và strict critic |

Mở [notebook Colab](notebooks/Amodal_Completion_Colab.ipynb) và chạy **8 ô theo
thứ tự**. Ô 4 mặc định `dev50`; đặt `SMOKE_LIMIT = 3` để kiểm tra nhanh. Ô 7 có
cờ `RUN_NO_IP_ABLATION` để chạy V3 với IP-Adapter tắt. Ô 8 xuất bảng CSV/JSON và lưới ảnh sáu cột vào Drive.
Khi đã chốt cấu hình trên dev, dùng `BATCH = "test400"`, `FINAL_TEST = True`.

- [Cài đặt Colab](docs/COLAB_SETUP.md)
- [Luồng xử lý và phân công module](docs/PIPELINE.md)
- [V3 và ablation](docs/PIPELINE_V3.md)
- [Protocol đánh giá](docs/EVALUATION.md)

COCO-A không cấp RGB thật của phần khuất. CLIP, SeamMAE và histogram màu là
chỉ số gián tiếp; PSNR, SSIM, LPIPS và Feature Similarity chỉ kiểm tra phần
nhìn thấy. Mã, cấu hình và tài liệu lịch sử có thể xem ở commit `9c3cfb8`.
