# Biên dịch báo cáo: `make` (tạo main.pdf), `make clean` (xóa file tạm).
# TeX Live trên macOS nằm ở /Library/TeX/texbin; đổi TEXBIN nếu máy khác.
TEXBIN ?= /Library/TeX/texbin

TEX = $(TEXBIN)/xelatex -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX = $(TEXBIN)/bibtex

main.pdf: main.tex notation.tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	$(TEX) main
	$(BIBTEX) main
	$(TEX) main
	$(TEX) main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz sections/*.aux

.PHONY: clean
