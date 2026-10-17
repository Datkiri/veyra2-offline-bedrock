# Veyra2 ngoại tuyến — ILwHee, phiên bản 1.0.2

Người làm addon: **ILwHee**.
YouTube: https://www.youtube.com/@Datkiri001

Model gốc: **Veyra2-Blueberry-10M-Instruct**, tác giả **Veyra AI**.
Nguồn: https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct

Đã fine-tune LoRA để model học thông tin tác giả addon, kênh YouTube và phân biệt
với tác giả model gốc; adapter đã được ghép trước khi xuất INT8. Không chèn tên
ILwHee vào system prompt hay trả lời tác giả bằng bộ định tuyến cố định.

Gõ `!ai trogiup`. Ngữ cảnh mặc định **tắt** cho từng người chơi:

```text
!ai tai
!ai Who made this addon?
!ai What is ILwHee YouTube link?
!ai Who made the AI model?
!ai ngucanh bat
!ai ngucanh tat
!ai thongke
```

Đổi chế độ ngữ cảnh sẽ xóa lịch sử cũ. Cache template cố định 82 token vẫn được
dùng khi tắt ngữ cảnh; cache này không có lịch sử người chơi.

Giao diện tiếng Việt, model chủ yếu hội thoại tiếng Anh. Phép tính như `!ai tính
1+1` được xử lý bằng JavaScript. Chỉ bật một pack chat nhận `!ai` trong thế giới.
Pack cần Beta APIs, dependency `@minecraft/server: "beta"`.

Kiểm tra PC ở `KIEM_TRA_MODEL.json` và `KIEM_TRA_LENH.json`; chưa có kiểm tra
trực tiếp Minecraft. Mã nguồn public vẫn có thể bị sửa hoặc huấn luyện lại.

Giấy phép: Apache License 2.0; xem `LICENSE-APACHE-2.0.txt` và `NOTICE`.
Mã nguồn và adapter: https://github.com/Datkiri/veyra2-offline-bedrock
