# Kiến trúc và phân công module

Bài toán chính: nhận RGB, mask nhìn thấy và mask phần bị che. Ghép phần RGB
sinh vào mask vùng khuất, giữ pixel nhìn thấy và đặt alpha bằng mask được cho.
Cả V1/V2/V3 dùng cùng SD1.5 Inpainting, DDIM và seed 42. V1 sinh một lượt;
V2/V3 có thêm prior, Qwen và tối đa ba lần sinh nên ngân sách tính toán khác nhau.

| Phần sở hữu | Mã nguồn chính |
| --- | --- |
| Thành viên 1 — V1, COCO-A, đánh giá | `src/data/cocoa_loader.py`, `src/pipelines/v1_direct.py`, `src/eval/metrics.py`, `src/eval/visualize.py`, `src/eval/fid.py`, `scripts/compare_versions.py` |
| Thành viên 2 — V2 và tài nguyên | `src/models/pix2gestalt_prior.py`, `src/vlm/qwen_basic.py`, `src/vlm/clip_verify.py`, `src/pipelines/v2_pipeline.py`, `src/models/memory_manager.py` |
| Thành viên 3 — V3 và ablation | `src/vlm/micro_attributes.py`, `src/models/sd15_ip_adapter.py`, `src/vlm/strict_critic.py`, `src/pipelines/v3_proposed.py`, `tests/test_v3.py` |

V1 dùng ảnh nền trắng với phần khuất khởi tạo xám và câu lệnh cố định.
V2 dùng Qwen mô tả cơ bản, Pix2Gestalt + CarveKit tạo prior RGB 2D, CLIP
kiểm tra prompt và SD1.5 tinh chỉnh ROI. V2 dùng critic chấm điểm tổng hợp.
V3 dùng cùng vòng điều phối và hyperparameter, thay lần lượt cách mô tả,
điều kiện thị giác, can thiệp latent, và cách chấm critic. Các chế độ cùng dùng
crop vuông/padding để tránh bóp méo tỉ lệ; manifest giữ tọa độ ảnh và mask gốc.

Memory manager nạp model theo nhu cầu, giữ trọng số ở RAM hệ thống khi
chuyển stage, giải phóng GPU trước khi chuyển model kế tiếp. Cách này giảm
VRAM đỉnh; giới hạn RAM còn phụ thuộc runtime.

Chỉ ba cấu hình chính được lưu trong repo. Ô 7 có thể tạo cấu hình V3 tắt IP-Adapter trong thư mục kết quả để đối chứng.
