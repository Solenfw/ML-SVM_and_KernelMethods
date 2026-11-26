# Hướng dẫn tự học phần P1 (B1–B6, C1–C4)

Tài liệu này giúp bạn **tự hiểu** những gì đã viết trong báo cáo, đủ để trình bày lại và trả lời
khi giảng viên hỏi. Mở song song `report/main.pdf` (chạy `cd report && make`) để đọc.

---

## 0. Đọc theo thứ tự nào

Đừng đọc theo thứ tự B1 → B6. Hãy đi theo mạch logic, mỗi bước dựa trên bước trước:

```
B1 (lề là gì) → B2 (bài toán đối ngẫu, KKT) → C1 (áp dụng B1 + B2 trên 3 điểm)
     → B3 (nhìn SVM như hinge loss)
     → B5 (hàm nhân nào hợp lệ) → C3 (một hàm KHÔNG hợp lệ) → C2 (kernel trick trên XOR)
     → B4 (vì sao nghiệm chỉ cần n hệ số) → B6 (vector hỗ trợ ↔ sai số) → C4 (kiểm chứng thực nghiệm)
```

**Cách đọc mỗi chứng minh, 3 lượt:**
1. **Lượt 1, chỉ đọc phát biểu.** Giả thiết là gì? Kết luận là gì? Thử với một ví dụ nhỏ (2D, 2–3 điểm).
2. **Lượt 2, chỉ đọc các dòng "Bước k".** Đó là bộ khung. Tự hỏi mỗi bước giải quyết việc gì.
3. **Lượt 3, đọc chi tiết.** Với mỗi dấu "=" hay "≤", hỏi: dùng giả thiết hoặc kết quả nào?

**Cách kiểm tra mình đã hiểu:** gập báo cáo lại, viết bộ khung các bước ra giấy trong 5 phút.
Viết được thì bạn đã hiểu.

---

## 1. B1: Khoảng cách và lề cứng

**Một câu tóm tắt:** khoảng cách từ x tới siêu phẳng là `|wᵀx + b| / ‖w‖`. Cực đại lề tương đương với
`min ½‖w‖²` sau khi "cố định thước đo" bằng điều kiện `yᵢ(wᵀxᵢ + b) ≥ 1`.

**Mệnh đề 2.1, bộ khung:**
- Muốn chứng minh "min = d" luôn cần **hai chiều**: mọi điểm ≥ d (cận dưới), và có một điểm = d (đạt được).
- Cận dưới: mẹo "trừ đi số 0" `wᵀz + b = 0`, sau đó dùng Cauchy–Schwarz.
- Đạt được: đi từ x dọc theo w (w vuông góc với siêu phẳng), giải ra t.

**Mệnh đề 2.2, bộ khung 6 bước:**

| Bước | Làm gì | Vì sao cần |
|---|---|---|
| 1 | Tính chất của F: khác rỗng, w ≠ 0, lồi | Để các bước sau hợp lệ. "Mỗi lớp có ≥ 1 mẫu" dùng ở đây |
| 2 | Bất biến tỉ lệ γ(cw, cb) = γ(w, b) | Ý tưởng chính: được phép chuẩn hóa lề hàm về 1 |
| 3 | (2) có nghiệm (Weierstrass) | Không thể nói về "nghiệm" nếu nó không tồn tại |
| 4 | Tại nghiệm, có ràng buộc đạt dấu bằng | Nếu không, co (w, b) lại sẽ được nghiệm tốt hơn |
| 5 | Nghiệm của (2) cực đại lề | Chiều chính của mệnh đề |
| 6 | Nghiệm duy nhất, nên cùng siêu phẳng | Để nói "siêu phẳng tối ưu trùng nhau" |

**Câu hỏi tự kiểm tra** (mở ra để xem đáp án):

<details><summary>Vì sao cần c > 0 mà không phải c ≠ 0 ở Bước 2?</summary>
c &lt; 0 đảo dấu mọi yᵢ(wᵀxᵢ + b): siêu phẳng giữ nguyên nhưng hai phía đổi chỗ, nên lề hình học đổi dấu.
</details>

<details><summary>Nếu dữ liệu chỉ có một lớp thì phần nào hỏng?</summary>
Khi đó w = 0, b = 1 thỏa mọi ràng buộc, cho ½‖w‖² = 0: bài toán (2) có nghiệm vô nghĩa, và Bước 1b sai.
</details>

<details><summary>Vì sao dùng ½‖w‖² mà không dùng ‖w‖?</summary>
Cùng tập nghiệm, vì t ↦ ½t² tăng trên [0, ∞). Nhưng ½‖w‖² khả vi mọi nơi và đạo hàm bằng w, nên bài toán thành quy hoạch toàn phương lồi.
</details>

**Hay bị hỏi:** "lề hình học khác lề hàm thế nào?"
- Lề hàm là `yᵢ(wᵀxᵢ + b)`, phụ thuộc thang đo: nhân w lên 2 lần thì nó cũng gấp đôi.
- Lề hình học là lề hàm chia cho ‖w‖, tức là khoảng cách thật.

---

## 2. B2: Đối ngẫu và KKT (phần quan trọng nhất)

**Một câu tóm tắt:** gắn "giá phạt" αᵢ cho mỗi ràng buộc, cực tiểu theo (w, b, ξ) rồi cực đại theo α.
Bài toán còn lại chỉ phụ thuộc dữ liệu qua `k(xᵢ, xⱼ)`, và đó là cửa ngõ của kernel trick.

**Bộ khung 4 bước:** Lagrangian → hàm đối ngẫu (cực tiểu theo b, ξ, w) → đối ngẫu mạnh + KKT → 3 trường hợp.

**Ba điều then chốt cần nhớ:**
1. **Ràng buộc `Σαᵢyᵢ = 0` đến từ b.** L tuyến tính theo b, nên nếu hệ số của b khác 0 thì inf L = −∞.
2. **Cận trên `αᵢ ≤ C` đến từ ξ.** Ta có αᵢ + μᵢ = C với μᵢ ≥ 0.
3. **`w = Σαᵢyᵢφ(xᵢ)` đến từ việc hoàn thành bình phương theo w.**

**Bảng ba trường hợp, nên thuộc lòng:**

| αᵢ | Suy ra | Vị trí điểm | Tên |
|---|---|---|---|
| 0 | yᵢf(xᵢ) ≥ 1 | ngoài lề, đúng phía | không phải SV |
| (0, C) | yᵢf(xᵢ) = 1 | đúng trên biên lề | SV tự do (dùng để tính b) |
| C | yᵢf(xᵢ) ≤ 1 | trong lề hoặc sai phía | SV bị chặn |

**Liên hệ với code của P2:** `I_up` và `I_low` trong `notes/kkt_cheatsheet.md` chính là bảng trên viết lại theo
gradient. SMO dừng khi không còn cặp nào vi phạm bảng này quá `tol`.

**Câu hỏi tự kiểm tra:**

<details><summary>Vì sao đối ngẫu mạnh xảy ra?</summary>
Bài toán lồi, mọi ràng buộc affine và có điểm khả thi (w = 0, b = 0, ξ = 1). Đó là điều kiện Slater dạng affine.
</details>

<details><summary>Một điểm có yᵢf(xᵢ) = 0,5 thì αᵢ bằng bao nhiêu?</summary>
αᵢ = C (Nhận xét 2.6 ngay sau Định lý 2.5: yᵢf &lt; 1 ⇒ ξᵢ &gt; 0 ⇒ μᵢ = 0 ⇒ αᵢ = C).
</details>

<details><summary>Vì sao chỉ dùng SV tự do để tính b?</summary>
Chỉ với SV tự do mới chắc chắn có đẳng thức yᵢf(xᵢ) = 1. SV bị chặn chỉ cho bất đẳng thức.
</details>

---

## 3. C1: Áp dụng B1 + B2 trên 3 điểm

**Nên tự làm lại trên giấy** (khoảng 15 phút). Đây là bài kiểm tra tốt nhất xem bạn đã hiểu B2 chưa.

**Hai cách giải, kiểm tra chéo nhau:**
- **Gốc:** `−b ≥ 1` ⇒ b ≤ −1 ⇒ w₁, w₂ ≥ 1 ⇒ ½‖w‖² ≥ 1, đạt được khi w = (1, 1), b = −1.
- **Đối ngẫu:** thay `α₃ = α₁ + α₂` rồi cực đại hàm bậc hai, được α = (½, ½, 1).

**Ý hình học:**
- Đường `x₁ + x₂ = 1` nằm giữa gốc và đoạn nối (2,0)–(0,2).
- Cả 3 điểm cách đường này đúng 1/√2, nên cả 3 đều là SV.

**Kiểm chứng bằng số:** `python experiments/c1_c3_check.py`. Script chạy SMO của P2 và LIBSVM, cả hai cho đúng kết quả trên.

<details><summary>Nếu thêm điểm âm (−1, −1) thì nghiệm có đổi không?</summary>
Không. Điểm đó có y·f = −(−1 − 1 − 1) = 3 &gt; 1, nằm ngoài lề nên α = 0. Đây chính là ý của B6: bỏ hay thêm điểm không phải SV thì nghiệm không đổi.
</details>

---

## 4. B3: SVM = hinge loss + chính quy hóa

**Một câu tóm tắt:** tại nghiệm, ξᵢ đúng bằng hinge loss của điểm i. Chia hàm mục tiêu cho Cn thì C biến thành
λ = 1/(Cn).

**Trực giác để giải thích khi được hỏi:**
- Mất mát 0–1 (đếm số lỗi) đúng nghĩa nhất nhưng không lồi và NP-khó.
- Hinge là **cận trên lồi** của nó. Hinge vẫn phạt cả điểm đúng nhưng nằm quá sát biên (0 < yf < 1), và đó chính là cái tạo ra "lề".
- C lớn tương đương λ nhỏ, tức chính quy hóa yếu: mô hình bám dữ liệu, lề hẹp, dễ quá khớp.

**Tự vẽ:** hàm bậc thang 1[t ≤ 0] và đường gãy max{0, 1 − t} trên cùng một trục.
Thấy được đường gãy luôn nằm trên bậc thang là bạn đã hiểu Bước 4.

---

## 5. B5 + C3: Hàm nào là kernel hợp lệ

**Định nghĩa cần thuộc:** k xác định dương ⇔ **mọi** ma trận Gram đều nửa xác định dương (cᵀKc ≥ 0).

**Bộ công cụ** (đây là các phép toán dùng để lắp ráp ra kernel mới):

| Phép toán | Vì sao giữ PD | Công cụ |
|---|---|---|
| k₁ + k₂ | cᵀ(K₁ + K₂)c = tổng hai số ≥ 0 | trực tiếp |
| c·k₁, c ≥ 0 | nhân với số ≥ 0 | trực tiếp |
| k₁·k₂ | Gram là K₁ ∘ K₂ | **định lý Schur** |
| lim kₘ | bất đẳng thức "≥ 0" qua giới hạn vẫn đúng | giới hạn |
| exp(k₁) | chuỗi Taylor = tổng các lũy thừa với hệ số dương | (i)–(iv) |

**Kernel Gauss**, công thức then chốt: `exp(−γ‖x − z‖²) = g(x)·g(z)·exp(2γxᵀz)`.
Lắp ráp: tuyến tính → nhân 2γ → exp → nhân với g(x)g(z).

**C3:** chỉ cần **một** phản ví dụ.
- k(x, x) = 0 trên đường chéo, nhưng k(x, z) = d > 0 ngoài đường chéo, nên ma trận `[[0, d], [d, 0]]` có trị riêng −d.
- Cách nhận ra nhanh: kernel PD phải thỏa `k(x, z)² ≤ k(x, x)·k(z, z)`. Ở đây vế phải bằng 0.

<details><summary>Vì sao trong chứng minh Schur ta được cᵀ(A∘B)c ≥ 0?</summary>
Viết A, B theo phân tích phổ. Khi đó cᵀ(A∘B)c = Σ λₖμₗ ((uₖ∘vₗ)ᵀc)², là tổng các số không âm.
</details>

---

## 6. C2: XOR và kernel trick

**Câu chuyện:**
- XOR không tách được bằng đường thẳng: cộng các bất đẳng thức lại ra 2b > 0 và −2b > 0.
- Ánh xạ sang 6 chiều bằng φ, khi đó tọa độ `√2·x₁x₂` tách được dữ liệu ngay.

**Điểm hay để trình bày:**
- Siêu phẳng chọn bằng tay `v = (0, 0, −1/√2, 0, 0, 0)` **trùng** với nghiệm SVM thật (α = 1/8 cho cả 4 điểm, b = 0). Script đã xác nhận điều này bằng SMO.
- Biên quyết định trong không gian gốc là `x₁x₂ = 0`, tức hai trục tọa độ. Đây là một đường cong bậc hai, không phải đường thẳng.

**Ý nghĩa của kernel trick:** SMO chỉ tính `(xᵀz + 1)²`, một phép toán trong 2 chiều, mà không bao giờ dựng φ 6 chiều.
Với kernel Gauss, φ thậm chí vô hạn chiều.

---

## 7. B4: Định lý biểu diễn

**Một câu tóm tắt:** phần nào của f "vuông góc" với dữ liệu thì không thấy được tại dữ liệu nhưng vẫn làm tăng chuẩn,
nên tốt nhất là bỏ nó đi.

**Bộ khung:** tách f = f∥ + f⊥ → tính chất tái sinh cho f(xᵢ) = f∥(xᵢ) → Pythagoras cho ‖f‖ ≥ ‖f∥‖ → Ω tăng nghiêm ngặt ⇒ f⊥ = 0.

**Vì sao quan trọng:** đây là lý do bài toán trên không gian hàm vô hạn chiều chỉ cần **n số** α₁..αₙ. Đó cũng là lý do
bài toán đối ngẫu ở B2 có n biến.

<details><summary>Giả thiết "Ω tăng nghiêm ngặt" dùng ở đâu? Nếu chỉ không giảm thì sao?</summary>
Dùng ở Bước 4 để có bất đẳng thức ngặt. Nếu Ω chỉ không giảm thì vẫn có ít nhất một nghiệm dạng Σαᵢk(xᵢ,·), nhưng không phải mọi nghiệm đều có dạng đó.
</details>

---

## 8. B6: Số vector hỗ trợ chặn sai số LOO

**Một câu tóm tắt:** bỏ một điểm có α = 0 thì nghiệm không đổi, nên điểm đó vẫn được dự đoán đúng.
Lỗi LOO chỉ có thể xảy ra ở các SV.

**Bộ khung:** SV có cả hai lớp → bỏ điểm i với αᵢ = 0 thì nghiệm cũ vẫn thỏa KKT nên vẫn là nghiệm (duy nhất) → yᵢf(xᵢ) ≥ 1 → đếm.

**Ý nghĩa thực tế:**
- Mô hình **thưa** (ít SV) thì có cận sai số tổng quát hóa tốt.
- Đây là một cận tính được ngay sau **một** lần huấn luyện, trong khi tính LOO thật cần n lần.

**Mở rộng:** Nhận xét 2.17(2) giải thích vì sao cận vẫn đúng với lề mềm. C4 kiểm chứng điều này trên 60 cấu hình.

---

## 9. C4: Đọc kết quả thực nghiệm

**Chạy lại:** từ thư mục gốc repo, chạy
- `python experiments/c4_C_gamma.py` (khoảng 5 phút), hoặc
- `python experiments/c4_C_gamma.py --plot-only` để chỉ vẽ lại hình từ CSV.

Nếu máy báo lỗi đa tiến trình thì thêm `--jobs 1`. Số liệu thô nằm trong `experiments/c4_grid_results.csv`.

**Thí nghiệm đo gì, nói gọn:**
- 2 bộ dữ liệu: moons 2 chiều, breast cancer 30 chiều.
- 30 cặp (C, γ) với kernel Gauss, huấn luyện bằng SMO của P2.
- Với mỗi cặp đo: số SV, độ rộng lề, sai số train, CV, LOO.
- Thêm nested CV để có sai số kiểm tra "trung thực".

### Cách đọc từng hình

**Hình heatmap (C4_heatmap):**
- Mỗi ô là sai số CV vòng trong của một cặp (C, γ). Màu nhạt là sai số thấp.
- Thấy được một **dải nhạt chạy chéo**: C lớn đi với γ nhỏ, hoặc ngược lại. Hai tham số cùng điều khiển độ phức tạp nên bù trừ nhau.
- Dòng tiêu đề "sai số ngoài" là con số nên báo cáo:
  - moons: 7,0 ± 1,1%
  - breast cancer: 1,9 ± 1,0%
- Cột γ ≥ 1 của breast cancer đậm đều ở mức 37,3%, đúng bằng tỉ lệ lớp thiểu số. Mô hình chỉ đoán mọi mẫu là lớp đa số.

**Hình (a)(b)(c) theo C (C4_sv_margin):**
- (a) và (b): khi C tăng, cả số SV lẫn độ rộng lề đều giảm. Điều này đúng với **mọi** γ, đã kiểm tra bằng code.
- Lý do (theo B3): C = 1/(λn), nên C lớn nghĩa là chính quy hóa yếu và lề hẹp.
- (c): sai số train giảm mãi, còn CV giảm rồi tăng lại (hình chữ U). Chỗ hai đường tách xa nhau chính là **quá khớp**.

**Hình 9 ô biên quyết định (C4_boundaries):** đọc theo cột trước (γ), rồi đến hàng (C).
- γ nhỏ: biên gần thẳng (thiếu khớp).
- γ = 1: biên ôm đúng hình trăng khuyết.
- γ = 10: biên quấn quanh từng cụm điểm (quá khớp).
- Vòng tròn đen là SV. Đếm sẽ thấy chúng tập trung ở vùng giáp ranh hai lớp.

**Hình LOO (C4_loo_bound):**
- Mỗi chấm là một cấu hình. Định lý B6 nói mọi chấm phải nằm **dưới** đường chéo, và cả 62/62 chấm đều như vậy.
- Chấm ở góc (100%, 100%) là 6 cấu hình moons với mọi α = C. Đọc kỹ giải thích trong báo cáo (lớp cân bằng 100/100), đây là ví dụ đẹp để kể khi vấn đáp.

### Các con số nên nhớ

| | moons | breast cancer |
|---|---|---|
| Cấu hình ô viền đậm (C, γ) | (1, 1) | (10, 10⁻²) |
| Sai số kiểm tra nested CV | 7,0 ± 1,1% | 1,9 ± 1,0% |
| nSV/n của cấu hình đó | 30% | 11,2% |
| Trung vị LOO / (nSV/n) | 0,37 | 0,36 |

### Câu hỏi tự kiểm tra

<details><summary>Vì sao cần nested CV, không dùng luôn sai số CV nhỏ nhất trên heatmap?</summary>
Vì ô nhỏ nhất được chọn chính bằng dữ liệu đó, nên con số bị lạc quan (chọn rồi tự chấm). Nested CV tách riêng: vòng trong chọn tham số, vòng ngoài chấm trên dữ liệu chưa từng dùng để chọn.
</details>

<details><summary>Khi mọi α = C, vì sao độ rộng lề tỉ lệ nghịch với C?</summary>
w = Σ αᵢyᵢφ(xᵢ) = C·Σ yᵢφ(xᵢ), nên ‖w‖ tỉ lệ thuận với C và 2/‖w‖ tỉ lệ nghịch. Số liệu: 25,1 → 2,51 → 0,251.
</details>

<details><summary>Vì sao γ lớn trên breast cancer lại cho sai số 37%?</summary>
Bình phương khoảng cách trung vị giữa hai mẫu khoảng 41, nên k ≈ e⁻⁴¹ ≈ 0. Ma trận Gram gần bằng I: mô hình nhớ từng điểm huấn luyện nhưng với điểm mới thì f(x) ≈ b, tức là đoán lớp đa số.
</details>

<details><summary>Cận B6 có dùng được để chọn C, γ không?</summary>
Không nên dùng một mình. Cấu hình thiếu khớp trên moons có nSV/n ≈ 35% (không tệ) nhưng sai số 17%. Cận chỉ là chặn trên của sai số; nó cho biết mô hình thưa thì "an toàn", không cho biết mô hình nào tốt nhất.
</details>

<details><summary>Tại sao trường hợp lề cứng trên moons dùng γ = 3 mà không phải γ = 1?</summary>
Với γ = 1, có cặp điểm khác lớp nằm quá sát nhau. Ngay cả khi C = 10⁶ vẫn có α chạm C (LIBSVM cũng vậy), tức đó vẫn là lề mềm. Với γ = 3, mọi α &lt; C, đúng giả thiết lề cứng của B6.
</details>

---

## 10. Các câu hỏi vấn đáp có thể gặp

1. Vector hỗ trợ là gì? Vì sao mô hình chỉ phụ thuộc vào chúng? (B2: α = 0 với các điểm khác; Định lý biểu diễn)
2. C và γ ảnh hưởng thế nào tới quá khớp? (B3 + C4)
3. Tại sao phải giải bài toán đối ngẫu thay vì bài toán gốc? (Kernel trick; số biến bằng n, không phụ thuộc số chiều)
4. Tại sao ‖x − z‖ không phải kernel nhưng exp(−γ‖x − z‖²) thì phải? (C3 vs B5)
5. Làm sao biết SMO hội tụ đúng? (KKT, m(α) − M(α) < tol, đối chiếu LIBSVM ở A4)
6. Nested CV khác CV thường thế nào? Vì sao cần? (C4)
