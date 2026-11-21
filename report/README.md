# Báo cáo LaTeX (Đề tài 4)

## Biên dịch

- **Máy local (macOS, TeX Live):** `cd report && make` sẽ tạo ra `main.pdf`. `make clean` xóa file tạm.
- **Overleaf:** upload cả thư mục `report/`, rồi vào Menu → Compiler → **XeLaTeX**.

Bắt buộc dùng XeLaTeX vì tiếng Việt cần `fontspec` và `babel`. pdfLaTeX sẽ báo lỗi.

## Ai sửa file nào

| File | Người phụ trách |
| --- | --- |
| `main.tex`, `notation.tex`, `refs.bib` | P1 (người khác muốn đổi thì báo P1) |
| `sections/00_intro.tex`, `B_theory.tex`, `99_conclusion.tex`, `Z_contributions.tex` | P1 |
| `sections/A_implementation.tex` | P2 |
| `sections/C_exercises.tex` | P1 (C1–C4), P2 (C5) |
| `sections/D_application.tex` | P3 |

Hình đặt trong `figures/`, đặt tên có tiền tố phần: `A1_convergence.pdf`, `C4_heatmap.pdf`, `D5_confusion.pdf`.
Ưu tiên PDF (vector) cho đồ thị. Ảnh chụp thì dùng PNG hoặc JPG.

## Quy ước

- **Ký hiệu:** chỉ dùng macro trong `notation.tex`, ví dụ `\vw`, `\vx`, `\valpha`, `\mK`, `\T`, `\norm{}`, `\Hk`.
  Cần thêm macro thì báo P1, không tự `\newcommand` trong file section.
- **Label:** đặt theo dạng `loại:Phần-tên`.
  - Mục: `sec:B2`
  - Định lý: `thm:B2-dual`
  - Phương trình: `eq:B2-primal`
  - Hình: `fig:C4-heatmap`
  - Bảng: `tab:A4-libsvm`
- **Trích dẫn:**
  - `\citet{key}` khi tên tác giả là một phần của câu.
  - `\citep{key}` khi trích trong ngoặc.
  - Thêm mục mới vào `refs.bib` thì báo P1.
- **Việc chưa xong:** đánh dấu `\todo{...}`, hiện chữ đỏ trong PDF. Trước khi nộp, PDF phải không còn chữ đỏ nào.

## Gói chưa có trên TeX Live basic

Môi trường pseudo-code `algorithm` / `algorithmic` cần cài thêm:

```
sudo tlmgr install algorithms algorithmicx
```

Overleaf có sẵn. Khi chưa cài, `main.tex` vẫn biên dịch được, chỉ là chưa dùng được môi trường này.
