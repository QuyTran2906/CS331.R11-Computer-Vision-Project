# V2 Pipeline

Trong cấu trúc hiện tại, V2 là pipeline vòng kín cơ bản dùng SD1.5 Inpainting:
Qwen3-VL mô tả đối tượng → Pix2Gestalt/CarveKit tạo prior → CLIP kiểm tra
prompt → DDIM tinh chỉnh ROI → Qwen chấm điểm tổng hợp. Cấu hình
`configs/v2_sd15_pipeline.json` không bật micro-attributes, IP-Adapter, latent
blending hoặc strict critic. Tên “V2” trong các backup từ trước khi tái cấu
trúc chỉ một thiết lập khác; xem [V3 hiện tại](PIPELINE_V3.md).
