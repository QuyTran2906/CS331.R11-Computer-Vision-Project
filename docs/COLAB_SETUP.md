# Chạy trên Colab

Đặt `cocoa_dataset.zip` trong Google Drive tại `MyDrive/CS331/`. Chọn GPU;
L4 với High-RAM là điểm khởi đầu phù hợp cho thử nghiệm vì Qwen, Pix2Gestalt
và ViT-H cần nhiều RAM hệ thống dù chỉ một stage được đặt lên GPU. T4 cần
chạy thử trước khi tăng batch.

1. Kiểm tra GPU, mount Drive và đọc secrets tùy chọn `HF_TOKEN` và `GITHUB_TOKEN`.
2. Clone hoặc `git pull --ff-only` nhánh `main`; giải nén dataset nếu chưa có.
3. Cài `.venv-gpu`, tải checkpoint Pix2Gestalt và cache Qwen, SD1.5, IP-Adapter Plus,
   rồi chạy `unittest`. CLIP kiểm tra prompt và CarveKit tải khi dùng lần đầu.
4. Tạo hoặc dùng lại manifest cố định. Mặc định 50 mẫu dev; có thể thử 3 mẫu.
5. Chạy V1 Direct.
6. Chạy V2 Pipeline.
7. Chạy V3 Proposed. Đặt `RUN_NO_IP_ABLATION = True` để chạy thêm V3 no IP.
8. Chấm ảnh đã sinh, so sánh, xem lưới và lưu ZIP vào `results/comparisons`.

Tăng `SMOKE_LIMIT = 3` thành `None` rồi chạy lại 4–8 để dùng đủ dev50.
Kết quả có thể resume theo profile và checksum khi mã, môi trường, cấu hình
và mẫu khớp. ZIP Drive lưu dưới `MyDrive/CS331/results`. Khi sang test400,
đặt `FINAL_TEST = True`, dùng toàn bộ mẫu và giữ cấu hình đã chọn từ dev.
