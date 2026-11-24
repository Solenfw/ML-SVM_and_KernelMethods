# Phần P2 — Thư viện `svmlib`, kiểm thử A4 và thí nghiệm C5

Gói mã nguồn này được thiết kế để copy thẳng vào thư mục gốc của repo `ML-SVM_and_KernelMethods` (khớp 100% cấu trúc thư mục và quy ước trong `notes/kkt_cheatsheet.md`).

## Cấu trúc bàn giao

- `svmlib/`
  - `_base.py`: Lớp cơ sở tương thích `sklearn.base.BaseEstimator` / `ClassifierMixin` (chạy được trong `Pipeline`, `GridSearchCV`, `cross_val_score` trên Colab; có fallback thuần NumPy).
  - `kernels.py` (**T2.1**): Hàm nhân `linear`, `poly`, `rbf` chuẩn hóa theo quy ước sklearn, tính ma trận Gram vector hóa, tính cột động `compute_kernel_column` cho $n$ lớn, và kiểm tra bán xác định dương `is_psd`.
  - `smo.py` (**A1 / T2.2**): `BinarySMO` (chọn cặp WSS1, cập nhật giải tích Platt 1998, làm tròn biên `1e-12 * C`, cập nhật gradient $O(n)$, log hàm mục tiêu và độ vi phạm KKT) và lớp thống nhất `SVC` hỗ trợ cả nhị phân lẫn đa lớp.
  - `dcd.py` (**A2 / T2.4**): `LinearSVMDCD` cài đặt thuật toán Hsieh et al. (2008) có active-set shrinking và tự động tăng tốc bằng Numba khi chạy trên Colab.
  - `krr.py` (**A2 / T2.4**): `KernelRidge` giải hệ $(K + \lambda I)\alpha = y$ bằng phân rã Cholesky (`scipy.linalg.cho_factor` / `cho_solve`).
  - `platt.py` (**A3 / T2.5**): Platt scaling theo Lin, Lin & Weng (2007) với CV nội bộ 5-fold, pairwise coupling theo Wu, Lin & Weng (2004), Brier score và dữ liệu reliability diagram.
  - `multiclass.py` (**A3 / T2.5**): `OneVsOneClassifier` (bỏ phiếu + phá hòa bằng tổng decision value + xác suất pairwise coupling) và `OneVsRestClassifier` (argmax + chuẩn hóa xác suất).
- `tests/test_vs_libsvm.py` (**A4 / T2.3**): Bộ `pytest` đối chiếu `svmlib` với `sklearn.svm.SVC` (LIBSVM), `LinearSVC(loss="hinge", dual=True)`, `KernelRidge`, `sklearn.metrics.pairwise`, và đáp án giải tay C1.
- `experiments/`
  - `a4_report_table.py`: Xuất bảng CSV đối chiếu chi tiết với LIBSVM trên 4 bộ dữ liệu × 3 hàm nhân × 3 giá trị $C$.
  - `generate_a1_a3_figures.py`: Xuất hình vector `report/figures/A1_convergence.pdf` và `report/figures/A3_calibration.pdf`.
  - `c5_timing.py` (**C5 / T2.6**): So sánh thời gian SMO tuyến tính (tính cột động, không cấp phát ma trận Gram) và DCD cho $n \in \{10^3, 3\cdot 10^3, 10^4, 3\cdot 10^4, 10^5\}$, xuất `experiments/c5_timing_results.csv` và `report/figures/C5_timing.pdf`.
- `report/figures/`: Đã tạo sẵn 3 hình vector PDF (`A1_convergence.pdf`, `A3_calibration.pdf`, `C5_timing.pdf`) đúng quy ước đặt tên của P1.
- `P2_Colab_Runner.ipynb`: Notebook chạy 1-click toàn bộ test A4, xuất bảng và chạy lại C5 trên Google Colab.

## Cách dùng cho P1 và P3

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from svmlib import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", decision_function_shape="ovo", probability=True)),
])
pipe.fit(X_train, y_train)          # nhận cả nhãn {-1, +1}, {0..K-1}, hoặc chuỗi tên giống gạo
preds = pipe.predict(X_test)
probs = pipe.predict_proba(X_test)  # dùng ngưỡng từ chối open-set cho phần D5
```
