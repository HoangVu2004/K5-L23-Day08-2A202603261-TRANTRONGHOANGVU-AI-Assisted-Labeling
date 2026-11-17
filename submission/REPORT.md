# Báo cáo Day 08 — 2A202603261 - Trần Trọng Hoàng Vũ

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26-m | person=0.50, bicycle=0.30, motorbike=0.30, car=0.30, bus=0.30, truck=0.30 | Dẫn `r01` (conf 0.25, 215 box) và `r02` (conf 0.50, 139 box). YOLO26-m chạy nhanh trên CPU (0.92–1.06s/ảnh), nhận diện tốt các lớp COCO chuẩn. Từ kết quả `threshold_sweep.md`: lớp `person` chọn ngưỡng 0.50 đạt recall 93.3% (≥ 90%) và precision 63.6% (15 vật gold). Các lớp `bicycle`, `bus`, `truck` có quá ít vật trong gold (< 5 vật) và `motorbike`, `car` không đạt recall 90% ở bất kỳ ngưỡng nào ≤ 0.50 nên giữ ngưỡng mặc định 0.30 để tránh bỏ sót. |
| three_wheeler | YOLOE-26s (prompt: "tuk-tuk, cyclo, cargo tricycle") | 0.15 | Dẫn `r03` (conf 0.25, 8 box trên pilot, 0.44s/ảnh). YOLO26 là closed-set không có lớp `three_wheeler` và thường gán nhầm tuk-tuk thành `car`/`truck`. YOLOE open-vocabulary cho phép phát hiện phương tiện ba bánh địa phương. Từ `threshold_sweep.md`: hạ ngưỡng xuống 0.15 đạt recall 100% (≥ 90%) và precision 66.7% trên 6 vật gold. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
```bash
python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
```
*(Trong lệnh gộp, file YOLOE được xếp trước YOLO26 để box `three_wheeler` được ưu tiên đè lên box `car`/`truck` của YOLO26 khi IoU ≥ 0.6).*

- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do:
  + Phiên bản chính chọn: **SAM 2.1** (checkpoint `sam2.1_b.pt` - Base, chạy trong `r04` và Part 5).
  + Lý do:
    * **Chất lượng mask:** SAM 2.1 cải thiện vượt bậc cấu trúc đường biên so với SAM 1 (`mobile_sam.pt` / `sam_b.pt`), bám sát các mép cong cơ học của khung mui xe tuk-tuk và nan hoa bánh xe, ít bị răng cưa hay lem nhem.
    * **Tốc độ và tài nguyên:** `sam2.1_b.pt` (~400MB) chạy ổn định trên CPU/GPU phổ thông, không tốn lượng RAM khổng lồ như SAM 3 (~3.5GB).
    * **Quyền truy cập:** Trọng số SAM 2.1 là mã nguồn mở mở hoàn toàn, tải trực tiếp không cần đăng ký tài khoản Hugging Face gated như SAM 3, dễ tích hợp và kiểm thử offline qua script `sam2_points.py`.

- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  + **Bản chất lỗi:** Đây là lỗi bất tương thích nhãn (taxonomy mismatch / domain gap). Do YOLO26 được huấn luyện trên 80 lớp COCO chuẩn (không có khái niệm `three_wheeler`), model buộc phải ánh xạ các đặc trưng hình ảnh của xe tuk-tuk (mui xe, đèn pha, kính chắn gió) vào lớp gần nhất là `car` hoặc `truck` với điểm tin cậy rất cao. Việc tăng hay giảm ngưỡng `conf` chỉ làm ẩn hoặc hiện box đó chứ không thể biến `car` thành `three_wheeler`.
  + **Cách sửa:** Kết hợp mô hình open-vocabulary (YOLOE-26s) với text prompt chuyên biệt (`"tuk-tuk, cyclo, cargo tricycle"`). Sau đó dùng `merge_prelabels.py` với cơ chế ưu tiên: đặt YOLOE lên trước YOLO26, thiết lập ngưỡng đè IoU 0.6 để box `three_wheeler` ghi đè và triệt tiêu box `car`/`truck` tại cùng vị trí. Ở khâu review người, người gán nhãn phải chủ động đổi lớp (remap class) theo checklist bước 3.

- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  + **Điểm cao mà sai (False Positive / Sai lớp có conf cao):** Trên ảnh `img_013.jpg` và `img_017.jpg`, chiếc xe tuk-tuk bị YOLO26 gán là `car` với điểm tin cậy `conf` lên tới 0.88–0.90. Người trên biển quảng cáo hoặc hình phản chiếu qua cửa kính (`img_006.jpg`) cũng bị model gán là `person` với conf > 0.85 dù theo guideline 4.1.7 không được gán.
  + **Điểm thấp mà đúng (True Positive có conf thấp):** Trên ảnh `img_003.jpg` và `img_013.jpg`, xe tuk-tuk ở góc nghiêng hoặc người đi bộ bị che khuất ở hậu cảnh xa chỉ đạt `conf` khoảng 0.16–0.22 từ YOLOE/YOLO26 nhưng thực tế đúng là đối tượng hợp lệ theo hợp đồng nhãn. Nếu áp ngưỡng lọc cứng 0.50, các vật này sẽ bị bỏ sót hoàn toàn.

- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:
  + Tập gold chỉ gồm 4 ảnh (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) với tổng cộng 40 đối tượng (trong đó `bus` và `truck` có 0 vật, `bicycle` chỉ có 3 vật).
  + Việc tối ưu hóa và quét ngưỡng trên chính 4 ảnh gold rồi đánh giá lại chất lượng trên cùng tập này là một trường hợp điển hình của **in-sample evaluation (data leakage)**. Số đo Precision (51.5%) và Recall (85.0%) mang tính lạc quan hơn thực tế đường phố. Kích thước mẫu 4 ảnh quá nhỏ khiến khoảng tin cậy Wilson 95% rất rộng (Precision 40%–63%, Recall 71%–93%), không đủ độ tin cậy thống kê để đại diện hoàn hảo cho toàn bộ 24 ảnh batch phức tạp.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 428 | 428/428 = 100.0% | 428 box máy |
| edited | 0 | 0/428 = 0.0% | 428 box máy |
| deleted | 0 | 0/428 = 0.0% | 428 box máy |
| added | 0 | 0/428 = 0.0% | 428 đối tượng sau duyệt |

*(Ghi chú: Đã loại 34 đối tượng nằm quanh các khu vực bẫy nhân tạo khỏi bảng AI error profile để phản ánh đúng chất lượng gốc của máy. Phân bố accepted theo lớp: person: 250, bicycle: 9, motorbike: 91, car: 49, bus: 3, truck: 7, three_wheeler: 19).*

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  + **Lớp `three_wheeler`:** Máy tệ nhất cả về phân loại và hình học. YOLO26 mù hoàn toàn lớp này nên gọi thành `car`/`truck` (`img_013.jpg`, `img_017.jpg`). Khi dùng YOLOE, box máy thường bị lỏng mép, thừa nhiều khoảng trống mặt đường hoặc bao trùm luôn cả hành khách ngồi trong (`img_003.jpg`, `img_004.jpg`).
  + **Lớp `person` trong đám đông và người ngồi trên xe:** Trên các ảnh đường phố đông đúc (`img_005.jpg`, `img_026.jpg`), máy thường bỏ sót người ngồi sau xe máy hoặc dính box người vào xe, vi phạm quy tắc bóc tách hai box độc lập cho người và phương tiện theo guideline 4.1.4.

- So với gold: precision / recall của **máy** Precision: 51.5% (34/66 box máy dự đoán, KTC 95%: 40%–63%) / Recall: 85.0% (34/40 đối tượng gold, KTC 95%: 71%–93%) (mẫu số: 40 đối tượng gold trên 4 ảnh gold; TP=34, FP=32, FN=6).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 0/12 = 0.0% (KTC 95%: 0%–24%); loại nào lọt: Lọt toàn bộ cả 4 loại bẫy nhân tạo (12/12 bẫy):
  + `missing` (0/3): `img_003.jpg` (bỏ sót three_wheeler), `img_028.jpg` (bỏ sót person), `img_031.jpg` (bỏ sót person).
  + `extra` (0/3): `img_005.jpg` (box ma car), `img_027.jpg` (box ma car), `img_006.jpg` (box ma person).
  + `wrong_class` (0/3): `img_021.jpg` (car đổi thành truck), `img_026.jpg` (person đổi thành bicycle), `img_025.jpg` (person đổi thành bicycle).
  + `loose` (0/3): `img_010.jpg` (box lỏng), `img_017.jpg` (box lỏng), `img_023.jpg` (box lỏng).
  + Vì sao (bạn đã bỏ bước nào của checklist)?
    Người duyệt đã bỏ qua hoàn toàn **Bước 0 — Quét ảnh trước** (không ẩn nhãn máy trong 15 giây đầu để tự đếm số lượng đối tượng trước khi hiển thị nhãn máy). Khi mở ảnh, mắt người duyệt lập tức bị "neo" vào các khung hình chữ nhật có sẵn (anchoring bias), ngộ nhận rằng các box phủ kín màn hình là đã đủ và đúng, dẫn đến việc chấp nhận hàng loạt (accept all) mà không kiểm tra từng bước theo checklist: không tìm vật sót (Bước 1), không xoá box ma ở khoảng trống (Bước 2), không soi lại tên lớp (Bước 3), và không co kéo lại mép box lỏng (Bước 4).

- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt?
  + Tỉ lệ `accepted` đạt tuyệt đối 100.0% (428/428 box) trong khi tỉ lệ bắt bẫy đạt 0.0% (0/12) là minh chứng kinh điển cho **Thiên kiến tự động hóa (Automation Bias)** và **Sự thừa nhận thụ động (Passive Complacency)**.
  + Nó phản ánh rằng người duyệt đã hoàn toàn mất vai trò "chốt chặn chất lượng" (quality gatekeeper), để AI chi phối hoàn toàn nhận thức thị giác. Người duyệt đã không thực hiện bất kỳ phản biện hay thao tác kiểm tra thực tế nào đối với dữ liệu máy sinh ra, khiến toàn bộ các lỗi nghiêm trọng (box rác, box ma, gán nhầm người thành xe đạp) được chuyển nguyên vẹn vào bộ dữ liệu cuối.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 6.0 phút/ảnh (24 phút / 4 ảnh gold) vs duyệt 1.25 phút/ảnh (30 phút / 24 ảnh batch). Thiên lệch (bạn gán gold trước nên đã quen cảnh):
  Khi gán tay 4 ảnh gold đầu tiên, người dán nhãn phải vừa đọc kỹ label contract vừa làm quen với tỉ lệ khung hình và các đối tượng nhỏ, dẫn đến thời gian ban đầu dài (6.0 phút/ảnh). Khi chuyển sang duyệt batch 24 ảnh, người duyệt đã có trí nhớ thị giác và quen thuộc với bối cảnh đường phố nên thao tác tự nhiên nhanh hơn. Tuy nhiên, tốc độ 1.25 phút/ảnh ở đây chứa đựng thiên lệch tiêu cực rất lớn: người duyệt lướt qua quá nhanh mà không kiểm tra đối soát, dẫn đến "nhanh nhưng sai toàn bộ bẫy".

- Kết luận: **Chỉ đáng dùng có điều kiện cho các lớp phổ biến (person, motorbike, car) VỚI ĐIỀU KIỆN phải tuân thủ nghiêm ngặt quy trình chống thiên kiến; KHÔNG ĐÁNG DÙNG nếu phó mặc cho máy hoặc dùng cho lớp địa phương (`three_wheeler`)**, dựa trên số liệu:
  + Về mặt tốc độ, pre-label giúp tiết kiệm gần 80% thời gian vẽ khung ban đầu (1.25 phút vs 6.0 phút), và recall của máy đạt 85.0% giúp bắt được phần lớn phương tiện rõ ràng.
  + Nhưng về chất lượng, Precision của máy chỉ đạt 51.5% (tức gần một nửa số box máy gợi ý cần phải chỉnh sửa hoặc loại bỏ) và tỉ lệ bắt bẫy thực tế bằng 0%. Nếu không có kỷ luật review nghiêm ngặt, việc dùng pre-label sẽ biến quá trình dán nhãn thành một quy trình hợp thức hóa lỗi sai của máy, làm suy giảm nghiêm trọng độ tin cậy của tập dữ liệu huấn luyện.

- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc "Mù 15 giây" (Cưỡng bức thực hiện Bước 0):** Bắt buộc ấn phím tắt ẩn toàn bộ nhãn máy (`toggle visibility`) trong 15 giây đầu tiên khi mở mỗi ảnh; tự đếm và ghi nhẩm số lượng từng lớp (`person`, `motorbike`, `car`, `three_wheeler`) và vị trí các vật bị che khuất vào bảng kiểm tra trước khi bật lại nhãn máy. Kiểm tra bằng: nhật ký ghi chép đếm nhẩm trước khi hiện box trên 100% ảnh.
  2. **Quy tắc duyệt phân vùng và đơn lớp (Sub-grid & Class-by-class Review):** Chia ảnh thành 4 góc phần tư và duyệt kiểm tra lần lượt từng lớp đơn lẻ thay vì nhìn toàn cảnh; tập trung 100% sự chú ý vào các mép ảnh và vùng khuất để tìm lỗi bỏ sót (`missing`), tuyệt đối không nhìn thuộc tính `conf` để phán đoán độ đúng của box. Kiểm tra bằng: số lượng thao tác chỉnh sửa/vẽ thêm `added` > 0 trên từng ảnh.
  3. **Quy tắc giới hạn phiên làm việc và kiểm tra bẫy chéo (Pomodoro & Trap Spot-checking):** Duyệt tối đa 8–10 ảnh (khoảng 20 phút) phải nghỉ giải lao 3–5 phút để phục hồi sự tập trung thị giác, tránh mỏi mắt; định kỳ chèn ngẫu nhiên các ảnh bẫy đã biết. Nếu tỉ lệ bắt bẫy của lô duyệt đạt dưới 80% (dưới 10/12 bẫy), bắt buộc từ chối nghiệm thu lô và thực hiện duyệt lại toàn bộ từ đầu.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_004.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 clicks (2 pos trên thân và mui xe, 1 neg trên người lái) | Không gồm người lái (đã tách ra bằng click âm); ôm trọn thân và bánh xe tuk-tuk | Mask có xu hướng lan vào khoảng trống gầm xe qua thanh chắn; cần click âm phụ để loại bỏ nền đường | `tuktuk_img004_sam2.jpg` |
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 clicks (pos "400,480", pos "600,110", neg "720,300") | Tách sạch người lái khỏi xe sau click âm neg; mask ôm sát mép mui và kính xe | Viền bóng đổ dưới gầm xe trên mặt đường hơi bị dính vào mép dưới của bánh xe | `tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | SAM 1 (`sam_b.pt` / `mobile_sam.pt`) | 3 clicks (tọa độ tương tự SAM 2.1 để so sánh đối chứng) | Khó tách rời người lái hơn; mask bao trùm một phần cánh tay người lái | Đường biên răng cưa ở các chi tiết hẹp như tay vịn và kính chắn gió; biên mask kém mượt hơn SAM 2.1 | `tuktuk_img013_sam1.jpg` |
| `img_017.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 clicks (2 pos trên thân xe/mui, 1 neg trên người/nền) | Không gồm người lái; bao phủ trọn khung kim loại của xe ba bánh | Xe ở xa và góc chụp nghiêng khiến mask bị phân tách thành 2 đa giác rời rạc (multi-polygon) ở phần khung đỡ | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?
  + **Quy tắc giúp ích:**
    * Quy tắc 4.2.3 (*Bóng đổ không thuộc vật; kính xe thuộc vật*): Định hướng rõ ràng việc đặt click âm để cắt bỏ phần bóng đen đổ trên mặt đường dưới gầm xe tuk-tuk.
    * Quy tắc 4.2.4 (*Hai vật chạm nhau không chồng lên nhau*): Giúp xác định ranh giới bóc tách: người lái ngồi trong cabin là một đối tượng `person` riêng biệt ở phía trước, không gộp vào mask của `three_wheeler`.
    * Quy tắc 4.2.2 (*Lỗ thủng nhỏ < 5% diện tích không vẽ lỗ*): Giúp bỏ qua các lỗ nhỏ li ti giữa các thanh nan hoa bánh xe hoặc khe giằng mui, tránh làm tăng đột biến số đỉnh polygon không cần thiết.
  + **Quy tắc còn thiếu / mơ hồ:**
    * Chưa có quy định cụ thể về việc xử lý đối tượng bị chia cắt quang học thành nhiều phần rời rạc (như khung xe bị cột chắn chia đôi tạo thành multi-polygon): guideline chưa nêu rõ có cho phép lưu multi-polygon trong định dạng COCO segment hay bắt buộc gộp thành polygon đơn.
    * Quy tắc kính chắn gió trong suốt nhìn thấy người bên trong: Chưa có hướng dẫn chi tiết về việc diện tích nhìn xuyên qua kính có bị trừ bớt khi tính diện tích thân xe hay không.

- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline:
  + **Thử nghiệm trong CVAT:** Chạy môi trường Docker local trên CPU gặp lỗi timeout (504 Gateway Timeout) khi CVAT Nuclio cố gắng nạp mô hình SAM nặng vào bộ nhớ RAM hệ thống; Nuclio function báo trạng thái không khả dụng khi thiếu GPU chuyên dụng.
  + **So sánh với script offline (`sam2_points.py`):**
    * Script offline chạy ổn định tuyệt đối, kiểm soát chính xác toạ độ từng điểm click dương/âm (`--pos`, `--neg`), ghi nhận diện tích mask và xuất file ảnh kiểm tra trực tiếp vào `submission/masks/`.
    * CVAT Interactor mang lại trải nghiệm tương tác trực quan thời gian thực (real-time interactive feedback), cho phép undo/redo điểm click ngay trên giao diện web và chuyển đổi trực tiếp sang polygon/mask của CVAT để lưu vào định dạng dữ liệu chuẩn.

## 6. Calibration (Part 6)

- Số quyết định: 4 quyết định (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`):
  + Quan trọng nhất là **`CAL-01`** (`loi_may` trên `img_013.jpg|img_017.jpg`): Xử lý triệt để lỗi phân loại hệ thống của máy — remap toàn bộ xe ba bánh có mui (tuk-tuk) hoặc ba bánh chở hàng từ `car`/`truck` về đúng lớp `three_wheeler`.
  + Quyết định bản lề thứ hai là **`CAL-02`** (`cau_hoi_mo` trên `img_002.jpg|img_011.jpg`): Chốt quy tắc phân loại xe điện hai bánh dựa trên bàn đạp trợ lực cơ học (có bàn đạp → `bicycle`; dáng scooter/không bàn đạp → `motorbike`).

- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:
  + Chốt định nghĩa phân biệt xe điện hai bánh (Mục 2).
  + Quy tắc nhận diện và remap xe ba bánh địa phương có mui che và xe chở hàng (Mục 2 & Mục 4.3).
  + Quy tắc phân định ranh giới giữa vẽ từng box xe và gộp `ignore_region` cho bãi đỗ xe máy đông đúc gần camera (Mục 4.1.6).
  + Quy tắc giới hạn bao viền cho hàng hoá nhô ra cồng kềnh: nhô ra ≤ 20% chiều dài xe thì tính vào box xe, > 20% thì chỉ vẽ ôm thân xe (Mục 4.1.5).
  + Quy tắc bóc tách mask SAM giữa kính chắn gió xe và người lái ngồi bên trong cabin xe ba bánh (Mục 4.1.4 & Mục 4.2.3).

- Nguyên nhân gốc: Phân tích kết hợp cả ba yếu tố:
  + `guideline mơ hồ`: Bản guideline v1.0 để ngỏ 7 câu hỏi mở, chưa có hướng dẫn cho xe điện hai bánh, hàng hoá cồng kềnh nhô ra và xử lý đám đông xe máy gần camera.
  + `máy`: YOLO26 bị giới hạn tập nhãn COCO 80 lớp gây domain gap nghiêm trọng với phương tiện địa phương; YOLOE sinh box lỏng lẻo; SAM có xu hướng nuốt trọn cả người lái vào mask xe.
  + `chưa nắm quy tắc / thiên kiến người duyệt`: Người duyệt buông lỏng kiểm soát, không thực hiện bước quét mù 15 giây dẫn đến việc thừa nhận nguyên vẹn các box lỗi của máy.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ảnh ngẫu nhiên: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`, seed 42), tổng cộng 94 đối tượng; người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt lô vừa là người kiểm tra QC, dễ bị lặp lại điểm mù nhận thức và thiên kiến cá nhân).
- Lỗi theo **đối tượng**: 4/94 = 4.3% · theo **ảnh**: 4/5 = 80.0% (ngưỡng Release ≤ 2%).
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại:
  + `bo_sot` (2 lỗi): Bỏ sót xe ba bánh tuk-tuk trên `img_003.jpg` và bỏ sót người đi bộ trên `img_031.jpg` (do bẫy missing lọt qua khâu review).
  + `thua` (1 lỗi): Box ma người xuất hiện ở khoảng trống không có người thật trên `img_006.jpg` (bẫy extra lọt).
  + `sai_vi_tri_kich_thuoc` (1 lỗi): Box xe ô tô ở mép ảnh bị lệch rộng, thừa khoảng trống nền trên `img_035.jpg`.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`):
  Với 4 ảnh lỗi trong mẫu 5 ảnh, cận trên khoảng tin cậy 95% (Wilson score interval) của tỉ lệ ảnh lỗi thật lên tới **96%** (thậm chí nếu quan sát thấy 0 lỗi thì cận trên 95% vẫn là 43% theo quy tắc 3/n). Kích thước mẫu n=5 ảnh là quá nhỏ về mặt toán thống kê để có thể chứng minh hay bác bỏ một cách vững chắc rằng tỉ lệ lỗi tổng thể đạt ngưỡng Release ≤ 2%. Kết luận đạt chuẩn trên mẫu nhỏ là rất yếu và cần được mở rộng mẫu khi nghiệm thu thực tế.
- Script kiểm được / không kiểm được:
  + **Script kiểm được:** Cú pháp COCO JSON, kích thước ảnh, ID và danh mục lớp có trong taxonomy, tọa độ box nằm trong khung hình, đối soát định lượng accepted/edited/deleted/added, tính IoU ghép cặp và số đo độ lệch tự động giữa hai file.
  + **Script KHÔNG kiểm được:** Độ ôm sát thực tế của box quanh viền vật thật (chỉ mắt người mới phân biệt được box có bị dư nền hay lẹm chi tiết), sự chính xác về ngữ nghĩa lớp (ví dụ tuk-tuk nhìn giống car thì script không biết nếu không có ground truth), các vật bị cả người lẫn máy cùng bỏ sót (joint omission), và script không thể kiểm tra được liệu người duyệt có thực sự làm theo checklist 4 bước hay chỉ click duyệt hàng loạt.

**Quyết định: REWORK**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release):
  1. *Kiểm tra Hold:* Guideline v1.1 đã được bổ sung hiệu chỉnh đầy đủ các ca khó, dữ liệu ảnh không bị lỗi format kỹ thuật hay hỏng hóc → Không chọn Hold.
  2. *Kiểm tra Rework:*
     - Tỉ lệ bắt bẫy nhân tạo đạt 0.0% (0/12 bẫy), thấp hơn rất nhiều so với ngưỡng tối thiểu bắt buộc `trap_catch_min ≥ 80%` (cần ≥ 10/12 bẫy).
     - Tỉ lệ ảnh có lỗi trong mẫu kiểm tra QC đạt 80.0% (4/5 ảnh), vượt xa ngưỡng tối đa cho phép `error_rate_img_max ≤ 2%`.
     - Tỉ lệ `accepted` đạt 100.0% chứng minh người duyệt mắc thiên kiến tự động hóa, chưa thực sự thực hiện việc rà soát và sửa lỗi của máy.
     Theo cây quyết định: Dừng ngay tại **REWORK**.

Nếu Rework / Hold, việc cần làm tiếp:
  1. Nạp lại file `pre_traps_import.coco.json` vào CVAT, tiến hành quy trình review lại toàn bộ 24 ảnh của lô batch.
  2. Áp dụng triệt để Checklist 4 bước (bắt buộc thực hiện Bước 0: ẩn toàn bộ nhãn máy trong 15 giây đầu để tự đếm số lượng vật).
  3. Bắt và sửa đúng tối thiểu 10/12 bẫy nhân tạo (vẽ bổ sung các box bị `missing`, xoá các box ma `extra`, đổi lại nhãn đúng cho các box `wrong_class`, kéo lại mép các box `loose`).
  4. Xuất file nhãn đã sửa thành `submission/labels/qc_fixed.json`, chạy lại `scripts/build_release.py` và `scripts/review_stats.py` để cập nhật các số đo đạt chuẩn trước khi tiến hành nghiệm thu Release.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không):
  Dùng Antigravity / Gemini assistant để hỗ trợ phân tích code trong thư mục `scripts/`, đối soát cú pháp checklist, giải thích các công thức thống kê (Wilson score interval, IoU matching threshold, automation bias) và định dạng báo cáo markdown. Mọi số liệu đo đạc (precision, recall, số lượng box, tỉ lệ bẫy, IoU) đều được kiểm tra chéo trực tiếp từ kết quả thực thi các script `review_stats.py`, `threshold_sweep.py`, `qc_report.py`, và `check_submission.py`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?
  **Không**. 4 ảnh gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) được dán nhãn thủ công hoàn toàn độc lập từ đầu trong Part 1 dựa trên guideline v1.0, không xem trước kết quả pre-label từ bất kỳ model nào và không hỏi AI về tọa độ hay nhãn của các ảnh này.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt:
  + Phần cứng & Hệ điều hành: Windows 11, chạy trên CPU (Intel Core), Python 3.11/3.12.
  + CVAT: Chạy local qua Docker Compose (CVAT v2.81.0).
  + Vướng mắc cài đặt & xử lý: Trên Windows PowerShell, các chuỗi tham số chứa dấu phẩy hoặc dấu `=` (`--classes "person,motorbike,car"`, `--thr "person=0.5,three_wheeler=0.15"`) bắt buộc phải bọc trong dấu nháy kép để tránh bị ngắt chuỗi; khi import annotation vào CVAT ở chế độ Replace cần bấm thêm nút xác nhận "Replace annotations" ở hộp thoại thứ hai; chạy model SAM bằng script Python offline (`sam2_points.py`) do CVAT Nuclio trên môi trường CPU dễ bị quá tải/timeout.
- Dùng cache `data/cache/` thay vì tự chạy?
  Có tự chạy tay Part 2 trên các ảnh pilot (`t_y26`, `t_y26_c05`, `t_yoloe`, `t_sam`) và kết hợp sử dụng dữ liệu cache `data/cache/yolo26m_all_c005.coco.json` cùng `yoloe26s_three_wheeler_all_c005.coco.json` cho bước quét ngưỡng Part 3 trên toàn bộ ảnh nhằm tiết kiệm thời gian tính toán trên CPU.
- Lỗi / bug script đã gặp (nếu có):
  + Lỗi mã hóa ký tự mặc định cp1252 trên Windows console khi in các chuỗi UTF-8 tiếng Việt; đã khắc phục bằng cách thiết lập biến môi trường `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8`.
  + Cú pháp PowerShell nhiều dòng cần thay ký tự ngắt dòng `\` (của bash) bằng dấu backtick ``` ` ``` hoặc viết gọn trên một dòng lệnh duy nhất.
- Một điều bạn làm khác lần sau:
  + Luôn tuân thủ kỷ luật Bước 0: tắt hoàn toàn nhãn máy trong 15 giây đầu tiên để tự quét ảnh bằng mắt thường và đếm số lượng đối tượng trước khi nhìn thấy gợi ý của AI.
  + Chia nhỏ thời gian làm việc thành từng phiên 20 phút (Pomodoro) để duy trì độ tỉnh táo của thị giác, tránh để sự mệt mỏi dẫn đến việc tin tưởng mù quáng vào model và bỏ lọt các bẫy cơ bản.
