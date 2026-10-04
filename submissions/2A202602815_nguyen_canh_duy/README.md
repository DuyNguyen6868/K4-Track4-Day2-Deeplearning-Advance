# Bài làm Lab Day 2: DeepWeeds (backbone, công thức huấn luyện, suy luận)

- **Họ tên / MSSV:** Nguyễn Cảnh Duy / 2A202602815
- **Link chạy lại (Kaggle notebook):** <https://www.kaggle.com/code/cnhduynguyn/notebookea0baa0d83>
- **Nơi lưu checkpoint:** không chia sẻ riêng; `best.pt` của từng lần chạy nằm trong Output của notebook Kaggle (`runs/<exp_id>/seed<k>/`).
- **Kết quả chính (test, 3 seed):** `F01` macro-F1 0,9649 ± 0,0006, top-1 0,9734 ± 0,0002; mốc `T00 + I00` macro-F1 0,9497 ± 0,0045. Chi tiết: `report.md`.

> **Trạng thái các sản phẩm.** Đủ 6 sản phẩm: `results.xlsx`, `report.md`, `curves/` (26 ảnh), `code/`, README này,
> `predictions/` (15 file). Mọi số liệu sinh ra từ lần chạy notebook trên Kaggle (GPU Tesla T4) ngày 2026-10-04.

## Cấu trúc

```
README.md            # file này
report.md            # báo cáo
results.xlsx         # 7 sheet: Backbones, Training, Inference, Final, PerClass, Latency, Summary  (do results.py sinh)
report_tables.md     # bảng markdown sinh từ cùng số liệu để dán vào report.md
decisions.json       # các quyết định chọn cấu hình (chỉ dùng val), có lý do
curves/              # <exp_id>_<mota>.png, mỗi lần huấn luyện một ảnh (loss, F1 val, LR)
predictions/         # <exp_id>_seed<k>_{val,test}.csv: Filename, y_true, y_pred, p0..p8 (softmax)
figures/             # EDA, kiểm tra pipeline, đánh đổi F1-độ trễ, độ tin cậy, ma trận nhầm lẫn
scores/              # đầu ra eval.py score/grade (bản chép của eval_out/, vì eval_out/ bị .gitignore)
logs/                # config.json, summary.json, history.csv của từng lần chạy (truy ngược exp_id -> log)
code/
  lab_day2.ipynb     # notebook chạy toàn bộ quy trình
  dataset.py  model.py  losses.py  train.py  inference.py  benchmark.py   # thành phần
  experiments.py     # định nghĩa B/T/F, quy tắc chọn, chạy hàng loạt (idempotent)
  study.py           # Bước 3: suy luận + độ trễ trên val
  final.py           # Bước 4: test đúng một lần mỗi seed, chấm bằng eval.py, phân tích lỗi
  results.py         # dựng results.xlsx, kiểm tra nhất quán
  eda.py             # EDA và kiểm tra pipeline
  evalref.py         # nạp eval.py gốc của repo (không sửa)
  tests/             # test CPU với dữ liệu giả
```

## Chạy lại

Cần GPU (Kaggle: bật *Internet* và GPU). **Không cần chạy tay từng lệnh**: mở `code/lab_day2.ipynb`, đặt `REPO_URL` và `SUB_NAME`
ở ô đầu, rồi chạy từ trên xuống. Mỗi giai đoạn idempotent (đã có `summary.json` thì bỏ qua, bị ngắt thì tiếp tục từ `last.pt`).
Tương đương dòng lệnh (từ thư mục làm việc có `data/`):

```bash
C=submissions/2A202602815_nguyen_canh_duy/code
COMMON="--images-dir data/images --labels-dir data/labels --out-dir runs --pred-dir predictions --curves-dir curves"
python $C/eda.py eda   --images-dir data/images --labels-dir data/labels
python $C/eda.py sanity --backbone resnet50 --images-dir data/images --labels-dir data/labels
python $C/experiments.py $COMMON backbones          # B01-B07, seed 0
python $C/experiments.py $COMMON select-backbone
python $C/experiments.py $COMMON noise              # T00 x3 seed
python $C/experiments.py $COMMON training           # T01-T13 (thêm --include-extra cho T14-T19)
python $C/experiments.py $COMMON winners
python $C/experiments.py $COMMON combo              # T20
python $C/experiments.py $COMMON recipe
python $C/experiments.py $COMMON final-train        # F01 x3 seed
python $C/study.py --images-dir data/images --labels-dir data/labels --out-dir runs   # Bước 3 (chỉ val)
python $C/final.py --images-dir data/images --labels-dir data/labels --out-dir runs --pred-dir predictions --yes-open-test          # Bước 4: MỞ TEST, mỗi seed một lần
python $C/results.py --xlsx results.xlsx            # results.xlsx + report_tables.md
python $C/results.py --xlsx results.xlsx --check    # kiểm tra nhất quán
```

`eval.py` gốc nằm ở gốc repo (không sửa); `code/evalref.py` tìm nó bằng cách đi lên các thư mục cha
hoặc biến môi trường `LAB_EVAL_PY`.

### Cách tôi đã chạy trên Kaggle (hai lượt)

1. **Version 1** (*Save & Run All*, `CONFIRM_OPEN_TEST = False`): Bước 0-3, chỉ dùng train và val, mất 4 giờ 52 phút.
2. **Lượt mở test**: gắn Output của Version 1 làm Input, đặt `RESTORE_FROM` tới thư mục chứa `runs/`, đổi
   `CONFIRM_OPEN_TEST = True`, rồi *Save & Run All*. Các lần train đã xong được bỏ qua; Bước 3 chạy lại trên val, Bước 4
   chạy test đúng một lần cho mỗi (exp_id, seed), Bước 5 dựng `results.xlsx`.

Chạy một lượt duy nhất với `CONFIRM_OPEN_TEST = True` và `RESTORE_FROM = ""` cũng cho kết quả tương đương (khoảng 5 giờ trên T4).
Ô cuối của notebook xoá `data/` để Kaggle không lưu 17.509 ảnh vào Output.

## Thiết lập tái lập

| Mục | Giá trị |
|---|---|
| Dữ liệu | DeepWeeds, ảnh 17.509 (MD5 `b7b30f96d466fba86016aa5a26606e0f`), fold 0 chia sẵn, không sửa |
| Seed | 0 cho Bước 1-3; 0, 1, 2 cho `T00` và chung kết (`F01`) |
| GPU / torch / timm / Python | Tesla T4 (Kaggle) / 2.11.0+cu128 / 1.0.29 / 3.13.15 |
| Trọng số | timm, tag ghim (`code/model.py` → `SUGGESTED_BACKBONES`); tag thực dùng ở cột "tag trọng số" của `Backbones` |
| Số epoch / ngân sách | 12 epoch, batch 64 (mặc định của `code/train.py`), không hạ; không chạy `T14`-`T19`; không có `T20` vì không yếu tố nào thắng |
| Tất định | seed Python/NumPy/PyTorch cố định; AMP và vài toán tử CUDA không tất định bit, nên chạy lại cho kết quả gần, không nhất thiết trùng |

## Các quyết định thiết kế (giải thích ngắn)

- **Test chỉ mở ở `final.py`**, mỗi (exp_id, seed) một lần (`test_started.json`). Mọi lựa chọn dùng val; T (temperature) khớp trên val theo từng seed.
- **Quy tắc chọn đặt trước khi xem kết quả** (docstring `experiments.py`, bảng trong `report.md` mục 2); mọi quyết định ghi vào `decisions.json`.
- **Mốc công bằng:** `T00 + I00` (1 view, không TS) cùng số seed với chung kết; trục G (độ phân giải, epoch) chỉ phân tích, không ghép vào công thức cuối để giữ cùng ngân sách tính toán.
- **Dùng lại lần chạy giống hệt** qua `alias_run` (ví dụ `F01` = `T00` khi công thức cuối là `T00`): ghi `alias_of` trong config/summary và cột ghi chú của `results.xlsx`.
- **Độ trễ** đo bằng `benchmark.py`: warmup ≥ 10, `cuda.synchronize`, ≥ 50 lần, p50/p95/p99, batch 1; **chưa gồm** tiền xử lý ảnh và đọc đĩa.
- **Mọi số trong `results.xlsx`** đọc từ log chạy thật hoặc tính lại từ `predictions/` bằng `eval.py`; `results.py --check` đối chiếu `exp_id` ↔ `curves/` ↔ `predictions/` ↔ xlsx.

## Kiểm thử mã (không cần GPU)

```bash
cd code
python -m unittest discover -s tests
```

Test dùng ảnh giả 256×256 và resnet18 nhỏ để kiểm tra đường ống, các chốt chặn (test-once, chia dữ liệu) và logic quyết định;
các con số trong test **không có ý nghĩa khoa học**.

## Công cụ hỗ trợ

Có dùng trợ lý AI (Claude Code) để viết và rà code, và để soạn thảo báo cáo từ số liệu đã chạy. Mọi thí nghiệm do tôi chạy
trên tài khoản Kaggle của mình; mọi con số truy ngược được tới `logs/` và `predictions/`.
