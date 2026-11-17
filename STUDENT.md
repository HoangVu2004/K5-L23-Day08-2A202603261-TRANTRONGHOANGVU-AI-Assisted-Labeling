# Thông tin học viên

- Họ và tên: Trần Trọng Hoàng Vũ
- MSHV: 2A202603261
- Lớp (L2 / L3 / khác): K5-L23
- Tài khoản GitHub: hoangvu2004
- Repo bài nộp (URL): https://github.com/HoangVu2004/K5-L23-Day08-2A202603261-TRANTRONGHOANGVU-AI-Assisted-Labeling
- Commit chốt bài (hash): 
- Máy: Windows 11, CPU (Intel Core, không có GPU rời)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 428/428 = 100.0% / 0/428 = 0.0% / 0/428 = 0.0% (mẫu số: 428 box máy); added: 0/428 = 0.0% (mẫu số: 428 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 51.5% (34/66 box máy) / 85.0% (34/40 đối tượng gold)
- Bắt bẫy: 0/12 = 0.0%
- Thời gian: gán tay 6.0 phút/ảnh (24 phút / 4 ảnh); duyệt có pre-label 1.25 phút/ảnh (30 phút / 24 ảnh)
- Lỗi QC: 80.0% theo ảnh (4/5), 4.3% theo đối tượng (4/94)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ bắt bẫy đạt 0/12 (0.0% < 80%) và ảnh có lỗi trong mẫu QC đạt 80.0% (> 2%), bộc lộ thiên kiến tự động hóa nên bắt buộc phải duyệt lại lô.
