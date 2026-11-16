# V3 Proposed

- Qwen3-VL đọc crop ngữ cảnh và crop chỉ chứa phần thấy được, trả JSON
  `object_class`, `primary_color`, `secondary_colors`, `material_and_texture`, `sd_prompt`.
- Prior Pix2Gestalt/CarveKit và kiểm tra bằng CLIP kế thừa từ V2.
- [IP-Adapter Plus](https://huggingface.co/h94/IP-Adapter) dùng CLIP ViT-H mã hóa
  crop visible trên nền trắng; scale mặc định 0,5. Latent blending tham chiếu
  trạng thái đúng mức nhiễu ở mỗi bước DDIM, không đưa RGB vật che vào phần cần sinh.
- Critic chấm riêng `color_match` và `structure_continuity`. Cả hai điểm phải
  đạt ngưỡng và model phải trả PASS; JSON không hợp lệ được ghi là REJECT sau
  khi thử sửa định dạng. Nếu bị từ chối, đổi seed, tăng guidance 0,5 và thêm
  phản hồi vào prompt. Có tối đa 3 lần sinh tổng cộng. Nếu hết lượt, giữ ứng
  viên có điểm hợp lệ tốt nhất và ghi `exhausted_retries`.

`critic_logs.jsonl` lưu quyết định từng lượt. Kết quả đã sinh có thể REJECT.
Ô 7 tạo cấu hình đối chứng bằng cách tắt đúng một cờ so với V3; hai nhánh có thể tự chọn số
lượt thử khác nhau do critic. So V2 với V3 để đo bốn thay đổi cùng lúc; so
V3-no-IP với V3 để phân tích tác động của riêng adapter trong hệ thống vòng kín.
