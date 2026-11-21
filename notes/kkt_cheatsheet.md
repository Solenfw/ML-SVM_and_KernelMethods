# Cheat sheet KKT cho SMO (P1 → P2)

Mọi công thức dưới đây đã được chạy thử bằng một SMO tối giản và khớp LIBSVM (sklearn 1.9.1)
trên 12 cấu hình: moons với linear/poly/rbf × C ∈ {0.1, 1, 10}, và breast cancer với rbf × C ∈ {0.1, 1, 10}.
Sai lệch tương đối của hàm mục tiêu đối ngẫu ≤ 4e-11, |Δb| ≤ 4e-6, tập SV trùng hoàn toàn, dự đoán khớp 100%.

## 1. Ký hiệu

- Dữ liệu (xᵢ, yᵢ), yᵢ ∈ {−1, +1}, i = 1..n
- Kᵢⱼ = k(xᵢ, xⱼ),  Qᵢⱼ = yᵢ yⱼ Kᵢⱼ,  e = (1, …, 1)

## 2. Bài toán đối ngẫu (dạng cực tiểu, giống LIBSVM)

```
min_α  F(α) = ½ αᵀQα − eᵀα
s.t.   0 ≤ αᵢ ≤ C,   yᵀα = 0
```

Hàm quyết định:  f(x) = Σⱼ αⱼ yⱼ k(xⱼ, x) + b

## 3. Gradient (biến trung tâm của SMO)

```
G = Qα − e,   tức là  Gᵢ = yᵢ (f(xᵢ) − b) − 1
Khởi tạo:  α = 0  ⇒  G = −e
```

Liên hệ với sai số Eᵢ = f(xᵢ) − yᵢ trong bài báo Platt:  yᵢGᵢ = Eᵢ − b,  nên  Eᵢ − Eⱼ = yᵢGᵢ − yⱼGⱼ.

## 4. Điều kiện KKT

Dạng dễ hiểu (dùng cho unit test, kiểm tra với tolerance):

| αᵢ | Điều kiện | Ý nghĩa |
| --- | --- | --- |
| αᵢ = 0 | yᵢ f(xᵢ) ≥ 1 | đúng lớp, nằm ngoài lề |
| 0 < αᵢ < C | yᵢ f(xᵢ) = 1 | nằm đúng trên lề (SV tự do) |
| αᵢ = C | yᵢ f(xᵢ) ≤ 1 | trong lề hoặc phân loại sai (SV bị chặn) |

Dạng dùng trong code (Keerthi et al. 2001; Fan, Chen & Lin 2005):

```
I_up(α)  = { t : (yₜ = +1 và αₜ < C)  hoặc  (yₜ = −1 và αₜ > 0) }
I_low(α) = { t : (yₜ = +1 và αₜ > 0)  hoặc  (yₜ = −1 và αₜ < C) }

m(α) = max_{t ∈ I_up}  −yₜ Gₜ
M(α) = min_{t ∈ I_low} −yₜ Gₜ

α tối ưu  ⇔  m(α) ≤ M(α)
Dừng khi    m(α) − M(α) < tol      (LIBSVM mặc định tol = 1e-3)
```

## 5. Chọn cặp vi phạm lớn nhất (WSS1)

```
i = argmax_{t ∈ I_up}  −yₜ Gₜ
j = argmin_{t ∈ I_low} −yₜ Gₜ
```

## 6. Cập nhật giải tích cho cặp (i, j) (Platt 1998)

```
Cận cho αⱼ:
  yᵢ ≠ yⱼ:  L = max(0, αⱼ − αᵢ),      H = min(C, C + αⱼ − αᵢ)
  yᵢ = yⱼ:  L = max(0, αᵢ + αⱼ − C),  H = min(C, αᵢ + αⱼ)

η = Kᵢᵢ + Kⱼⱼ − 2Kᵢⱼ ;   nếu η ≤ 0 thì đặt η = 1e-12   (giống LIBSVM)

αⱼ_new = clip( αⱼ + yⱼ (yᵢGᵢ − yⱼGⱼ) / η ,  L, H )
αᵢ_new = αᵢ + yᵢ yⱼ (αⱼ − αⱼ_new)
```

**⚠️ BẮT BUỘC: làm tròn về đúng biên sau mỗi lần cập nhật**

```
eps = 1e-12 · C
với α ∈ {αᵢ_new, αⱼ_new}:   α < eps ⇒ α = 0 ;   α > C − eps ⇒ α = C
```

Không có bước này, sai số số thực để lại các α cỡ 1e-17 thay vì đúng 0 (hoặc lệch C một chút).
Điểm đó vẫn bị tính là "α > 0" nên được chọn lại mãi mà bước nhảy bằng 0, và SMO chạy tới max_iter.
Khi thử không làm tròn, 5/12 cấu hình bị kẹt, cho kết quả sai tới 88%.

## 7. Cập nhật gradient: O(n) mỗi vòng, chỉ cần 2 cột kernel

```
Δαᵢ = αᵢ_new − αᵢ ,   Δαⱼ = αⱼ_new − αⱼ
G ← G + y ⊙ ( yᵢ Δαᵢ K[:, i] + yⱼ Δαⱼ K[:, j] )
```

## 8. Tính b sau khi hội tụ

```
Free = { t : 0 < αₜ < C }
Có SV tự do:      b = trung bình của (−yₜ Gₜ) trên Free
Không có SV tự do: b = (m(α) + M(α)) / 2
```

(LIBSVM lưu ρ = −b.)

## 9. Giá trị hàm mục tiêu đối ngẫu (để log và so A4)

```
D(α) = Σαᵢ − ½ αᵀQα = −F(α) = −½ Σᵢ αᵢ (Gᵢ − 1)     ← có sẵn từ G, không cần tính lại Qα
```

Dòng `obj = …` LIBSVM in ra là F(α) = −D(α).

Với kernel tuyến tính:  w = Σ αᵢ yᵢ xᵢ ;  với kernel bất kỳ:  ‖w‖² = αᵀQα ;  độ rộng lề = 2 / ‖w‖.

## 10. Đối chiếu với sklearn (A4)

- `SVC` hai lớp: lớp `classes_[1]` ứng với y = +1.
- `dual_coef_[0, s] = y_s α_s` (chỉ cho các SV), `support_` = chỉ số các SV, `intercept_[0] = b`.
- D phía sklearn:  `Σ|dual_coef_| − ½ · dual_coef_ · K(SV, SV) · dual_coef_ᵀ`.
- Quy ước kernel phải truyền giống nhau ở cả hai bên:
  - linear: xᵀz
  - poly: (γ xᵀz + coef0)^degree
  - rbf: exp(−γ ‖x − z‖²)
- Mặc định của sklearn là `gamma='scale'`, `coef0=0`, `degree=3`, khác mặc định trong interface của nhóm (coef0 = 1).
  Khi so sánh, luôn truyền γ, coef0, degree một cách tường minh.
- Đặt `tol` nhỏ (ví dụ 1e-6) ở cả hai bên. Đếm SV với ngưỡng α > 1e-6 · C.

## 11. Test nhanh (đáp án đã kiểm chứng)

Bài C1: X = [(2,0), (0,2), (0,0)], y = [+1, +1, −1], kernel tuyến tính, C = 1e6 (xấp xỉ lề cứng):

```
α = (0.5, 0.5, 1),  w = (1, 1),  b = −1,  D = 1
y·f(x) = 1 với cả 3 điểm (cả 3 đều là SV nằm trên lề)
```

Số vòng lặp tham khảo (tol = 1e-6, moons n = 200, có làm tròn biên): 80–1700 vòng; poly với C = 10 cần khoảng 16 000 vòng.
