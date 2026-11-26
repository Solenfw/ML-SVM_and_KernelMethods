# Review C5 (P1 → P2)

Người review: Nguyễn Huy Hoàng (P1) · Gửi: Nguyễn Quốc Kỳ (P2) · Ngày: 26/11/2026

Đây là nhận xét theo mốc review chéo M4. **Mình không sửa file nào của P2**; Kỳ quyết định có sửa hay không.
Mọi con số dưới đây đều tái lập được bằng đúng dữ liệu và tham số của `experiments/c5_timing.py`.

---

## 1. DCD dừng vì chạm `max_iter`, chưa hội tụ (quan trọng)

**Hiện tượng:** trong `experiments/c5_timing_results.csv`, cột `mean_iter` của DCD bằng **1000 ở mọi n**,
trùng với `max_iter=1000` (số epoch) truyền vào `LinearSVMDCD` ở dòng 54 của script.

**Kiểm chứng:** chạy lại `LinearSVMDCD(C=1.0, tol=1e-3, fit_intercept=True, random_state=42)` trên
`make_linear_benchmark_data(n)`:

| n | max_iter | n_iter | converged | pg_gap | dual | thời gian |
|---|---|---|---|---|---|---|
| 1 000 | 1 000 | 1 000 | **False** | 0,0108 | 279,5024 | 0,12 s |
| 1 000 | 20 000 | 1 778 | True | 0,0008 | 279,5037 | 0,13 s |
| 10 000 | 1 000 | 1 000 | **False** | 0,0349 | 2412,0075 | 0,69 s |
| 10 000 | 20 000 | 9 090 | True | 0,0010 | 2412,0284 | 1,19 s |

Với `max_iter=1000`, `pg_gap` lớn hơn `tol = 1e-3` từ 10 đến 35 lần. Như vậy thời gian DCD trong CSV là thời gian chạy
**1000 epoch**, không phải thời gian để đạt cùng độ chính xác với SMO, nên so sánh hiện tại chưa công bằng.

**Đề xuất:**
- Nâng `max_iter` của DCD (ví dụ 50 000), rồi chạy lại.
- Ghi thêm cột `converged` và `pg_gap` (hoặc `kkt_gap`) vào CSV, để trong báo cáo có thể khẳng định cả hai thuật toán cùng đạt `tol`.

Kết luận định tính có lẽ vẫn giữ nguyên: DCD hội tụ ở n = 10⁴ sau khoảng 1,2 s, so với 27,6 s của SMO. Nhưng con số và độ dốc trên đồ thị log-log sẽ thay đổi.

## 2. SMO bị cắt ở 300 s, khác với mức 30 phút trong kế hoạch

**Hiện tượng:** cột `timeout_s` trong CSV bằng **300**. Trong khi đó docstring và giá trị mặc định của `--timeout`
là 1800 s (30 phút, đúng kế hoạch T2.6). Nhiều khả năng lần chạy đã dùng `--timeout 300`.
Hệ quả là SMO **không có số liệu ở n = 3·10⁴ và 10⁵**, nên đồ thị chỉ có 3 điểm cho SMO.

**Đề xuất (chọn một):**
- Chạy lại hai mức n đó với `--timeout 1800`, chẳng hạn trên Colab bằng `P2_Colab_Runner.ipynb`.
- Giữ 300 s nhưng ghi rõ trong báo cáo, và trên đồ thị đánh dấu "> 300 s" cho hai điểm bị cắt. Có thể thêm đường ngoại suy từ độ dốc của 3 điểm đầu (SMO đo được: 0,93 s, 3,4 s và 27,6 s ở n = 10³, 3·10³, 10⁴).

## 3. Gợi ý khi viết phần C5 trong báo cáo

- **Hàm mục tiêu đối ngẫu của DCD và SMO không so trực tiếp được.**
  - `fit_intercept=True` trong DCD thêm cột hằng số 1 vào x, nên b cũng bị chính quy hóa.
  - Bài toán đối ngẫu tương ứng vì vậy **không có** ràng buộc Σαᵢyᵢ = 0. Tập khả thi rộng hơn nên giá trị đối ngẫu của DCD luôn ≥ SMO. Số liệu khớp với điều này: 2412,03 so với 2412,02 ở n = 10⁴.
  - Nên ghi rõ điểm này. Nếu muốn so độ chính xác, có thể so độ chính xác trên tập kiểm tra.
- **Về độ phức tạp:** mỗi vòng của SMO tốn O(n·d) (tính 2 cột kernel), còn mỗi epoch của DCD cũng tốn O(n·d). Khác biệt nằm ở **số vòng**: SMO cần hàng chục nghìn cặp cập nhật (14 254 → 64 230 khi n tăng 10 lần), còn DCD chỉ cần vài nghìn epoch và có shrinking. Điều này giải thích độ dốc khác nhau trên đồ thị log-log.

---

Nếu cần, mình (P1) có thể chạy lại phép kiểm tra ở mục 1 cho các mức n khác. Phần sửa code và CSV vẫn để Kỳ làm.
