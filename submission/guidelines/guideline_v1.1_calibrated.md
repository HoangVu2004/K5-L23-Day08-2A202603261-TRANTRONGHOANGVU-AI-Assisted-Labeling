# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603261 - Trần Trọng Hoàng Vũ

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Xe đạp điện có bàn đạp trợ lực cơ học gán `bicycle`; xe máy điện dáng scooter không có bàn đạp (VinFast Klara, Evo, Feliz...) gán `motorbike`. | `img_002.jpg` | CAL-02 |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Xe máy gần camera nếu nhận diện được từng xe thì bắt buộc vẽ từng box riêng; chỉ dùng `ignore_region` cho các xe ở hậu cảnh xa bị che >80% hoặc cao < 20px. | `img_005.jpg` | CAL-03 |
| 3 | Hàng hoá nhô ra khỏi xe: box xe có gồm hàng không? | Hàng hoá nhô ra <= 20% chiều dài xe thì tính vào box xe; nhô ra > 20% thì chỉ vẽ box ôm thân xe, không mở rộng box bao phủ hàng hoá cồng kềnh. | `img_006.jpg` | CAL-03 |
| 4 | Người ngồi trong xe ba bánh có mui che nhìn thấy thấp thoáng? | Người ngồi trong xe nếu nhìn thấy >= 20% thân người phải có một box `person` riêng độc lập với box `three_wheeler`. | `img_004.jpg` | CAL-04 |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe hay không? | Kính chắn gió và cửa sổ thuộc mask xe; người ngồi bên trong là đối tượng riêng phía trước/sau, không gộp người vào mask xe. | `img_013.jpg` | CAL-04 |
| 6 | Box máy chồng nhau quá nhiều trong ảnh đêm? | Giữ nguyên box có IoU cao với vật thật, xoá box trùng có IoU >= 0.6 nếu cùng chỉ về một thực thể vật lý. | `img_027.jpg` | CAL-01 |
| 7 | Mức tin cậy tối thiểu để chấp nhận box máy không cần sửa? | Không đặt ngưỡng `conf` cứng để chấp nhận box tự động; bắt buộc kiểm tra bằng mắt theo checklist 4 bước. | `img_021.jpg` | CAL-01 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg` | Xe tuk-tuk có mui chở khách | YOLO26 gọi là `car` với conf 0.88 | Đổi nhãn thành `three_wheeler` | Mục 2 & 4.3: Xe ba bánh bất kể có mui che hay thùng hàng đều gán `three_wheeler`. Bắt buộc remap khi máy nhận nhầm sang `car`/`truck`. |
| `img_017.jpg` | Xe ba bánh chở hàng ở xa | YOLO26 gọi là `truck`, YOLOE box hơi lỏng | Giữ nhãn `three_wheeler`, kéo sát mép box | Mục 4.1.1: Box ba bánh phải ôm sát khung xe và bánh xe, không bao gồm mặt đường. |
| `img_004.jpg` | Mask xe tuk-tuk bao trùm cả người lái | SAM 2.1 tạo mask trùm cả người lái khi click thân xe | Đặt click âm loại bỏ người lái | Mục 4.2.4: Người lái là đối tượng `person` riêng, mask `three_wheeler` không được gộp người lái. |
| `img_026.jpg` | Người đi bộ bị gán nhầm thành xe đạp (bẫy) | Model gán nhãn `bicycle` cho người | Sửa lại nhãn `person` | Mục 2: Luôn kiểm tra lớp của box trước khi chấp nhận, không tin tưởng mù quáng vào conf. |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | Ban hành ban đầu |
| v1.1 | Chốt 7 câu hỏi mở (phân định xe điện, xử lý hàng cồng kềnh, quy tắc kính xe và người trong tuk-tuk); thêm quy tắc remap tuk-tuk từ car/truck sang three_wheeler và bóc tách mask SAM giữa người và xe. | CAL-01, CAL-02, CAL-03, CAL-04 (lỗi máy trên `img_013.jpg`, `img_017.jpg`, câu hỏi mở trên `img_002.jpg`, `img_005.jpg`) |
