# Veyra2 ngoại tuyến cho Minecraft Bedrock — ILwHee

Addon chat AI và tính toán chạy hoàn toàn trong Behavior Pack, không cần máy
chủ AI hay Internet khi chơi.

**Người làm addon:** ILwHee · [YouTube](https://www.youtube.com/@Datkiri001)

**Model gốc:** [Veyra2-Blueberry-10M-Instruct](https://huggingface.co/veyra-ai/Veyra2-Blueberry-10M-Instruct)
do **Veyra AI** tạo ra. ILwHee là tác giả addon và phần fine-tune nhận diện trong
dự án này, không phải tác giả model gốc. Mã addon và model dẫn xuất cùng dùng
Apache License 2.0; xem [LICENSE](LICENSE) và [NOTICE](NOTICE).

## Tải và cài

Tải [veyra2-ilwhee-1.0.2.mcpack](https://github.com/Datkiri/veyra2-offline-bedrock/releases/download/v1.0.2/veyra2-ilwhee-1.0.2.mcpack)
ở [trang phát hành](https://github.com/Datkiri/veyra2-offline-bedrock/releases/tag/v1.0.2),
hoặc chép thư mục `veyra2-10m` vào `behavior_packs`.

1. Import pack và bật **Veyra2 ngoại tuyến — ILwHee** trong thế giới.
2. Tắt các pack chat khác cùng nhận lệnh `!ai`.
3. Bật Beta APIs phù hợp. Dependency giữ `@minecraft/server: "beta"`, yêu cầu
   engine tối thiểu `[1,26,30]` như bản trước.
4. Mở lại thế giới sau khi cập nhật. Log phải ghi addon **1.0.2**.

## Lệnh

| Lệnh | Chức năng |
| --- | --- |
| `!ai trogiup` | Xem trợ giúp |
| `!ai <câu hỏi>` | Hỏi AI |
| `!ai tai` | Nạp model và chọn kernel |
| `!ai trangthai` | Trạng thái nạp, ngữ cảnh, hàng đợi |
| `!ai thongke` | Thời gian và số token của lượt gần nhất |
| `!ai ngucanh` | Xem ngữ cảnh đang bật hay tắt |
| `!ai ngucanh bat` | Bật lịch sử hội thoại cho người chơi hiện tại |
| `!ai ngucanh tat` | Tắt lịch sử hội thoại cho người chơi hiện tại |
| `!ai xoa` | Xóa lịch sử, giữ lựa chọn bật/tắt |
| `!ai huy` | Hủy câu trả lời đang chạy |
| `!ai do kernel` | Đo và chọn kernel trên thiết bị |
| `!ai tính (2+3)*4` | Tính bằng JavaScript, không cần nạp model |

Các lệnh cũ `help`, `load`, `status`, `stats`, `cancel`, `reset`, `bench kernels`
vẫn dùng được. Ngữ cảnh cũng nhận `!ai context on`, `off`, `status`.

**Ngữ cảnh mặc định tắt theo từng người chơi.** Khi tắt, prompt không có lịch sử
các câu hỏi/câu trả lời trước và addon không lưu phiên KV hội thoại. Khi bật,
addon giữ tối đa hai lượt gần nhất trong cửa sổ token. Mỗi lần đổi chế độ sẽ
xóa lịch sử cũ; nếu đang có câu trả lời, thay đổi áp dụng từ câu hỏi tiếp theo.
Lựa chọn này chỉ tồn tại trong phiên chạy, mở lại thế giới trở về mặc định tắt.

Cache tiền tố cố định 82 token vẫn được dùng trong cả hai chế độ. Nó chứa phần
template chung, không chứa lịch sử riêng của người chơi.

## Thông tin tác giả được học trong trọng số

Phiên bản này đã fine-tune LoRA trên checkpoint gốc, ghép adapter vào trọng số
FP32 rồi xuất INT8. Mô hình học các thông tin:

- Người làm addon là **ILwHee**.
- YouTube: **https://www.youtube.com/@Datkiri001**.
- Tác giả model gốc là **Veyra AI**.
- Nguồn model là Hugging Face của Veyra AI; giấy phép Apache 2.0.

Có thể hỏi trực tiếp:

```text
!ai Who made this addon?
!ai Ai làm addon này?
!ai What is ILwHee YouTube link?
!ai Who made the AI model?
```

Không có bộ định tuyến trả lời sẵn cho câu hỏi tác giả. `test_model.mjs` gọi
trực tiếp inference INT8 với template gốc, không thêm tên ILwHee vào system
prompt. Dữ liệu huấn luyện tự biên soạn, adapter và báo cáo có trong `training`.
Các câu validation dùng chọn checkpoint được ghi rõ trong báo cáo; bài kiểm tra
phát hành có tập câu hỏi riêng.

Mã nguồn public và giấy phép Apache 2.0 cho phép người khác sửa hoặc huấn luyện
lại. Fine-tune không tạo khóa chống chỉnh sửa hay bảo đảm model luôn trả lời
đúng. Các nghĩa vụ giữ thông báo bản quyền/giấy phép nằm trong LICENSE và NOTICE.

## Phạm vi và kiểm tra

- Model 9.976.832 tham số, Qwen3, 7 layer, vocab 4.096, INT8 tile4.
- Snapshot tiền tố được tạo lại từ trọng số fine-tune; tổng trọng số giải mã
  khoảng 9,62 MiB, snapshot thêm khoảng 0,56 MiB. Đây không phải tổng RAM.
- Cửa sổ 256 token; prompt tối đa 224 token; trả lời tối đa 48 token và được
  cắt theo phần cửa sổ còn lại. Mức 48 cho phép trả trọn URL nguồn model.
- Giao diện, thông báo và trợ giúp bằng tiếng Việt. Model vẫn chủ yếu dùng tiếng
  Anh; fine-tune có thêm câu nhận diện tiếng Việt, không phải huấn luyện đầy đủ
  khả năng hội thoại tiếng Việt.
- Bộ tính toán nhận cộng, trừ, nhân, chia và ngoặc; dùng số thực JavaScript,
  giới hạn độ lớn 10^12, độ sâu 16 và 12 chữ số có nghĩa khi hiển thị.
- Đã kiểm tra trên PC: nhận diện tác giả/nguồn, kernel, cache và lệnh ngữ cảnh.
  Chi tiết trong `veyra2-10m/KIEM_TRA_MODEL.json` và `KIEM_TRA_LENH.json`.
- Chưa chạy kiểm tra trực tiếp trong Minecraft. Không dùng số đo 1.0.1 để cam
  kết tốc độ hay chất lượng của model fine-tune 1.0.2.

## Xây lại từ nguồn

Cần Python 3.12, các thư viện trong `requirements.txt` và Node.js 24. Công cụ
Python chỉ dùng trên PC. Khi chơi, pack chỉ import dữ liệu JavaScript và API
Minecraft.

```text
python -m pip install -r requirements.txt
python tools/fetch_veyra2.py
python tools/merge_adapter.py
python tools/convert_veyra2.py
node tools/build_veyra2_prefix.mjs
python tools/reference_model.py
node --expose-internals tools/strip_comments.cjs
python tools/strip_python_comments.py
node tools/test_commands.mjs
node tools/test_model.mjs
python tools/package_release.py
```

File cài và checksum được tạo trong `dist`, không đưa vào lịch sử Git. Adapter
được ghép với model gốc có revision và hash đã ghim. Muốn huấn luyện lại thay
adapter, chạy `train_identity.py --base tools/checkpoints/veyra2-10m --out
tools/checkpoints/veyra2-ilwhee-sft --steps 600` rồi thực hiện lại các bước xuất.

Các bản raw checkpoint, môi trường Python, log và dữ liệu xác thực không nằm
trong repository. Chú thích trong mã JavaScript/Python đã được loại bỏ bằng bộ
phân tích cú pháp; chuỗi URL, biểu thức chính quy, LICENSE và NOTICE được giữ.
Bản model card gốc được lưu ở `licenses/MODEL_CARD_GOC.md` làm tài liệu nguồn;
nội dung giấy phép và tài liệu gốc tiếng Anh được giữ nguyên.
