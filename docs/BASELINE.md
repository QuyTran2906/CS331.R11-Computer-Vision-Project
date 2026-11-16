# V1 Direct

`configs/v1_sd15_direct.json` nhận ảnh, visible mask và hidden mask đã cho.
SD1.5 Inpainting chạy một lượt, DDIM, seed 42, guidance 7, 28 bước, strength 1.
Câu lệnh mô tả một vật thể tổng quát, không lấy nhãn COCO-A. Đầu ra ghép chính
xác pixel nhìn thấy từ ảnh gốc. Ô 5 chạy baseline, ô 8 so với V2 và V3 trên
cùng manifest. Không dùng profile lịch sử thay thế trong phép so sánh này.
