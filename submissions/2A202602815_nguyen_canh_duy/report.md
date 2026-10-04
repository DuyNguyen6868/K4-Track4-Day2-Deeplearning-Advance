# Báo cáo Lab Day 2: Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Họ tên:** Nguyễn Cảnh Duy · **MSSV:** 2A202602815 · **Notebook Kaggle:** <https://www.kaggle.com/code/cnhduynguyn/notebookea0baa0d83>

Mọi con số trong báo cáo lấy từ `results.xlsx`, `report_tables.md`, `scores/` và `logs/` của lần chạy thật trên Kaggle
(GPU Tesla T4). Số của bài báo gốc được ghi rõ là *trích dẫn*.

---

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp ảnh cỏ dại DeepWeeds (17.509 ảnh 256×256), fold 0 chia sẵn, chỉ số chính là macro-F1.
- **Thí nghiệm:** 7 backbone (`B01`–`B07`), công thức nền `T00` × 3 seed, 13 thí nghiệm một yếu tố (`T01`–`T13`, 7 trục),
  21 dòng suy luận (`I00`–`I08d`, trong đó 2 dòng gộp BN không áp dụng), chung kết `F01` × 3 seed.
- **Cấu hình tốt nhất (`F01`):** ConvNeXt-Tiny (`convnext_tiny.fb_in1k`) + công thức nền `T00` + TTA đa tỉ lệ
  224/256/288 gộp logit (`I03d`) + temperature scaling.
- **Kết quả test (mean ± std, 3 seed):** macro-F1 **0,9649 ± 0,0006**, top-1 **0,9734 ± 0,0002**, ECE 0,0061 ± 0,0016.
- **So với mốc `T00 + I00`** (0,9497 ± 0,0045): Δ macro-F1 = **+0,0152**, lớn hơn s = 0,0045 khoảng 3,4 lần → vượt nhiễu.
- **Kết luận chính:** (1) tiền huấn luyện và lựa chọn backbone quyết định phần lớn kết quả; (2) không yếu tố nào trong
  13 thay đổi công thức huấn luyện vượt được nhiễu giữa các seed; (3) toàn bộ phần cải thiện so với mốc đến từ suy luận
  đa tỉ lệ, không cần huấn luyện thêm.

## 2. Dữ liệu và thiết lập

**Dữ liệu.** DeepWeeds (Olsen và cộng sự, 2019): 17.509 ảnh, 9 lớp (8 loài cỏ + `Negative`). Dùng **fold 0 chia sẵn**
(`train_subset0`, `val_subset0`, `test_subset0`), tải nguyên bản, không sửa, không chia lại.

Kết quả kiểm tra chia dữ liệu (`figures/split_check.json`):

| Kiểm tra | Kết quả |
|---|---|
| Số ảnh train / val / test | 10.501 / 3.501 / 3.507 (59,97% / 20,00% / 20,03%) |
| Giao train∩val, train∩test, val∩test | 0 / 0 / 0 |
| Hợp ba tập | 17.509 |
| File trong CSV không có ảnh | 0 |

Phân bố lớp (`figures/class_counts.csv`); cột "bài báo" là số *trích dẫn* từ Table 1:

| Lớp | train | val | test | tổng | bài báo (Table 1) |
|---|---|---|---|---|---|
| Chinee apple | 675 | 225 | 226 | 1.126 | 1.125 |
| Lantana | 637 | 213 | 213 | 1.063 | 1.064 |
| Parkinsonia | 618 | 206 | 207 | 1.031 | 1.031 |
| Parthenium | 613 | 204 | 205 | 1.022 | 1.022 |
| Prickly acacia | 637 | 212 | 213 | 1.062 | 1.062 |
| Rubber vine | 605 | 202 | 202 | 1.009 | 1.009 |
| Siam weed | 644 | 215 | 215 | 1.074 | 1.074 |
| Snake weed | 609 | 203 | 204 | 1.016 | 1.016 |
| Negative | 5.463 | 1.821 | 1.822 | 9.106 | 9.106 |

![Phân bố lớp](figures/class_distribution.png)

Nhận xét: `Negative` chiếm 52,0% tập train, gấp 9,0 lần lớp ít nhất (Rubber vine), nên top-1 bị lớp này kéo cao và
macro-F1 là chỉ số chính. Số đếm từ `labels.csv` lệch Table 1 đúng 1 ảnh ở hai lớp (Chinee apple +1, Lantana −1),
tổng vẫn là 17.509. Ảnh mẫu mỗi lớp: `figures/samples_per_class.png`.

**Quy ước chống rò rỉ (S1–S6).** Mọi lựa chọn (backbone, công thức, bộ view TTA, nhiệt độ T, cấu hình thời gian thực)
dùng **val**. Test chỉ được mở ở `code/final.py`, **đúng một lần cho mỗi (exp_id, seed)**, có chốt chặn
`test_started.json`; thời điểm mở test ghi trong `logs/F01/seed*/test_done.json`. Nhiệt độ T khớp trên val cho từng seed.
Không gộp val vào train ở bất kỳ lần chạy nào.

**Chỉ số.** macro-F1 (chính), top-1, balanced accuracy, ECE 15 bin, NLL, tính bằng `eval.py` gốc của repo (không sửa);
std lấy mẫu (ddof = 1) qua seed.

**Kiểm tra pipeline trước khi train** (`figures/sanity.json`, ResNet-50):

| Kiểm tra | Kết quả | Kỳ vọng |
|---|---|---|
| Loss ban đầu | 2,191 | ln 9 = 2,197 |
| Overfit một batch nhỏ | loss 2,169 → 0,0001; accuracy ở `eval()` = 1,0 | loss → 0 |
| Ảnh sau augmentation | `figures/augmentation_preview.png` | ảnh hợp lý, nhãn đúng |

**Công thức nền `T00`** (`logs/T00/seed0/config.json`); mọi thí nghiệm `T` chỉ khác `T00` một yếu tố:

| Thành phần | Giá trị |
|---|---|
| Khởi tạo | tiền huấn luyện ImageNet-1k, fine-tune toàn mạng |
| Augmentation | `RandomResizedCrop(224)` + lật ngang |
| Loss | cross-entropy |
| Tối ưu | AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0,05 (không áp dụng cho norm và bias) |
| Lịch LR | warmup 1 epoch rồi cosine theo bước |
| Epoch / batch | 12 / 64 |
| Khác | AMP, `channels_last`, không EMA, không sampler |
| Chọn checkpoint | epoch có macro-F1 val cao nhất |
| Đánh giá | resize 256 + center crop 224, `model.eval()` |

**Môi trường.** Kaggle, GPU Tesla T4 · Python 3.13.15 · torch 2.11.0+cu128 · timm 1.0.29 · seed 0 cho mọi so sánh ở
Bước 1–3; seed 0, 1, 2 cho `T00` và `F01`.
**Ngân sách.** Không hạ số epoch. Không chạy các thí nghiệm mở rộng `T14`–`T19`. Không có `T20` (xem mục 4).

**Quy tắc chọn (đặt trước khi xem kết quả, nằm trong `code/experiments.py` và `code/study.py`):**

| Quyết định | Quy tắc |
|---|---|
| Backbone chính | macro-F1 val ≥ (tốt nhất − 0,003); trong số đó chọn mạng có độ trễ batch-1 thấp nhất |
| Yếu tố thắng | Δ = F1 val (yếu tố) − F1 val (`T00`, seed 0) ≥ max(2·std_T00, 0,003) |
| Kết hợp `T20` | ghép yếu tố thắng tốt nhất mỗi "ô", bỏ cặp xung đột; trục G chỉ phân tích |
| Công thức cuối | trong {`T00`, yếu tố thắng, `T20`}: F1 val cao nhất; chênh nhỏ hơn ngưỡng thì chọn cái ít thay đổi nhất |
| Bộ view TTA | bộ có F1 val cao nhất, chỉ nhận nếu hơn `I00` ≥ 0,002 |
| Thời gian thực | p95 batch-1 ≤ 100 ms và F1 val không kém `I00` quá 0,002; chọn p95 thấp nhất |

Kết quả các quyết định và lý do: `decisions.json`.

## 3. Kết quả so sánh backbone

Bảy mạng thuộc bốn họ, cùng công thức `T00`, cùng seed 0, trọng số ghim tag ImageNet-1k (sheet `Backbones`):

| exp_id | Backbone | Tag trọng số (timm) | Tham số (M) | GMAC | macro-F1 val | top-1 val | Train/epoch (s) | Độ trễ batch-1 p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| B01 | ResNet-50 | `resnet50.a1_in1k` | 23,5 | 4,09 | 0,8104 | 0,8603 | 34,0 | 6,17 |
| B02 | ResNeXt-50 | `resnext50_32x4d.a1h_in1k` | 23,0 | 4,23 | 0,7538 | 0,8112 | 43,6 | 9,39 |
| **B03** | **ConvNeXt-Tiny** | `convnext_tiny.fb_in1k` | 27,8 | 4,45 | **0,9568** | **0,9663** | 51,7 | 5,86 |
| B04 | DeiT-Small | `deit_small_patch16_224.fb_in1k` | 21,7 | 4,24 | 0,9443 | 0,9594 | 34,7 | 5,42 |
| B05 | Swin-Tiny | `swin_tiny_patch4_window7_224.ms_in1k` | 27,5 | 4,49 | 0,9544 | 0,9652 | 65,6 | 9,81 |
| B06 | EfficientNet-B0 | `efficientnet_b0.ra_in1k` | 4,0 | 0,38 | 0,7694 | 0,8298 | 26,2 | 7,92 |
| B07 | MobileNetV3-Large | `mobilenetv3_large_100.ra_in1k` | 4,2 | 0,22 | 0,7121 | 0,7912 | 17,7 | 6,56 |

**Chọn backbone.** Theo quy tắc đặt trước, ứng viên trong 0,003 của tốt nhất là ConvNeXt-Tiny (0,9568) và Swin-Tiny
(0,9544). ConvNeXt-Tiny có độ trễ batch-1 thấp hơn (5,86 ms so với 9,81 ms) và train nhanh hơn (51,7 so với 65,6 giây
mỗi epoch), nên được chọn. Chênh lệch F1 giữa hai mạng (0,0024) nhỏ hơn std_T00 (0,0040): về độ chính xác **không phân
biệt được**; quyết định dựa trên độ trễ. DeiT-Small thấp hơn 0,0125 (khoảng 3 lần std) nhưng mỗi backbone chỉ có 1 seed.

**Nhóm mạng dùng BatchNorm thấp bất thường.** Bốn mạng BatchNorm (ResNet-50, ResNeXt-50, EfficientNet-B0, MobileNetV3)
đạt 0,71–0,81, còn ba mạng LayerNorm đạt 0,94–0,96. Log cho thấy đây là **hội tụ chậm, không phải lỗi đánh giá**:
loss train của `B01` dừng ở 0,377 sau 12 epoch và phẳng từ epoch 7, trong khi `B03` xuống 0,062
(`logs/B01/seed0/history.csv`, `logs/B03/seed0/history.csv`). Nếu là lỗi BatchNorm giữa `train()` và `eval()` thì loss
train phải thấp mà val vẫn kém. Giả thuyết của tôi là LR backbone 1e-4 với AdamW và 12 epoch quá nhẹ cho nhóm này,
nhất là với trọng số `a1`/`a1h` được tiền huấn luyện bằng BCE. Giả thuyết này **chưa được kiểm chứng** (chưa chạy lại
ResNet-50 với LR cao hơn).

Hệ quả: so sánh này công bằng theo nghĩa "cùng công thức", nhưng công thức nền thiên về ConvNeXt và transformer. Không
thể kết luận ResNet-50 kém ConvNeXt 0,146 macro-F1 do kiến trúc; bài báo gốc báo cáo ResNet-50 đạt 95,7% accuracy
(*trích dẫn*, khoảng 100 epoch). Số GMAC không dự báo được độ trễ batch-1 trên GPU: EfficientNet-B0 (0,38 GMAC) chậm hơn
ConvNeXt-Tiny (4,45 GMAC) ở batch 1.

## 4. Kết quả công thức huấn luyện

`T00` chạy 3 seed: macro-F1 val = 0,9568 / 0,9492 / 0,9551, trung bình 0,9537, **std_T00 = 0,0040**.
Ngưỡng thắng = max(2 × 0,0040; 0,003) = **0,0079** (`decisions.json`). Mỗi thí nghiệm dưới đây chạy 1 seed (seed 0) và
chỉ khác `T00` một yếu tố (sheet `Training`):

| exp_id | Trục | Khác `T00` ở điểm nào | macro-F1 val | Δ so với `T00` | Δ / std | F1 Chinee apple | F1 Snake weed | Kết luận |
|---|---|---|---|---|---|---|---|---|
| T00 | — | công thức nền (seed 0) | 0,9568 | 0 | 0 | 0,924 | 0,920 | mốc |
| T01 | A khởi tạo | đóng băng backbone, chỉ train head | 0,6862 | −0,2706 | −68,2 | 0,620 | 0,593 | hại rõ |
| T02 | A khởi tạo | từ đầu, LR toàn mạng 1e-3 | 0,5818 | −0,3750 | −94,5 | 0,444 | 0,500 | hại rõ |
| T03 | B augmentation | + ColorJitter | 0,9507 | −0,0061 | −1,5 | 0,896 | 0,910 | không phân biệt được |
| T04 | B augmentation | + TrivialAugmentWide | 0,9520 | −0,0049 | −1,2 | 0,901 | 0,884 | không phân biệt được |
| T05 | B augmentation | CutMix (α = 1, p = 0,5) | 0,9555 | −0,0013 | −0,3 | 0,906 | 0,905 | không phân biệt được |
| T06 | B augmentation | Mixup (α = 1, p = 0,5) | 0,9534 | −0,0035 | −0,9 | 0,911 | 0,910 | không phân biệt được |
| T07 | C loss | label smoothing 0,1 | 0,9521 | −0,0047 | −1,2 | 0,909 | 0,914 | không phân biệt được |
| T08 | C loss | focal γ = 2 | 0,9503 | −0,0065 | −1,6 | 0,902 | 0,913 | không phân biệt được |
| T09 | C loss | CE có trọng số 1/n_c | 0,9488 | −0,0080 | −2,0 | 0,897 | 0,905 | sát ngưỡng hại |
| T10 | D sampler | sampler cân bằng lớp | 0,9431 | −0,0137 | −3,5 | 0,904 | 0,904 | hại |
| T11 | E LR | LR head = LR backbone = 1e-4 | 0,9492 | −0,0076 | −1,9 | 0,894 | 0,897 | không phân biệt được |
| T12 | F EMA | EMA decay 0,998 | 0,9558 | −0,0010 | −0,2 | 0,924 | 0,919 | không phân biệt được |
| T13 | G độ phân giải | train ở 256 thay vì 224 | 0,9557 | −0,0011 | −0,3 | 0,923 | 0,918 | không phân biệt được |

**Không có yếu tố nào thắng.** Cả 13 thí nghiệm có Δ âm, nên danh sách yếu tố thắng rỗng, công thức cuối là `T00`.
**`T20` (kết hợp) không chạy** vì quy tắc đặt trước chỉ kết hợp các yếu tố thắng; do đó báo cáo này không có bằng chứng
về việc các yếu tố cộng dồn hay triệt tiêu. `F01` dùng lại ba lần chạy `T00` (ghi `alias_of` trong `logs/F01/`).

Nhận xét theo trục:

- **Khởi tạo (A)** là yếu tố lớn nhất trong toàn bộ bài lab. Đóng băng backbone mất 0,27 macro-F1: loss train dừng ở
  0,62 (thiếu khớp, head tuyến tính không đủ). Train từ đầu mất 0,375: đường cong `curves/T02_scratch.png` vẫn đang đi
  lên ở epoch 12, tức 12 epoch với 10.501 ảnh là quá ít để học từ đầu. Chỉ thử trên ConvNeXt, chưa thử ViT từ đầu.
- **Augmentation (B), loss (C), LR (E), EMA (F), độ phân giải (G):** mọi Δ nằm trong ±2·std, nên **không phân biệt được**
  với `T00`. `T00` seed 0 là seed cao nhất trong ba seed (0,9568 so với trung bình 0,9537), vì vậy các Δ đều lệch âm
  khoảng 0,003; so với trung bình ba seed, CutMix, EMA và train ở 256 cao hơn 0,002 và vẫn trong nhiễu.
- **Cân bằng lớp (C `T09`, D `T10`)** đổi macro-F1 lấy balanced accuracy. `T10` nâng balanced accuracy val từ 0,956 lên
  0,960 và recall Snake weed từ 0,911 lên 0,931, nhưng recall `Negative` giảm từ 0,979 xuống 0,947; vì `Negative` có
  1.821 ảnh val, số dương tính giả tăng làm precision các lớp cỏ giảm và macro-F1 giảm 0,0137 (3,5 lần std). `T09` cho
  cùng xu hướng (balanced accuracy 0,961, recall `Negative` 0,958, Δ = −0,0080).
- **CutMix/Mixup** không giúp trong 12 epoch: loss train cao hơn do nhãn trộn (0,35 so với 0,06) và val loss gần như
  bằng `T00` (0,111 so với 0,115). Chưa đủ bằng chứng để nói cắt dán làm mất vật thể nhỏ.

Hạn chế của phần này: mỗi thí nghiệm `T` chỉ có 1 seed. Kết luận "hại" chỉ dùng cho `T01`, `T02`, `T10`.

**Đường cong huấn luyện** (`curves/<exp_id>_*.png`, mỗi lần chạy một ảnh gồm loss train/val, chỉ số val, LR):

- `T00`: loss train (0,06) thấp hơn loss val (0,115) từ epoch 6, loss val phẳng từ epoch 7; quá khớp nhẹ, không phân kỳ.
  Epoch tốt nhất là 11, 9, 10 cho seed 0, 1, 2.
- `B01`, `B02`, `B06`, `B07`: loss train còn cao và phẳng khi LR về 0, tức thiếu khớp.
- `T02`: mọi đường còn dốc ở epoch 12, tức hội tụ chậm.
- F1 val dao động khoảng ±0,005 giữa các epoch cuối, cùng cỡ với nhiễu giữa seed.

## 5. Kết quả suy luận

Mọi số trong mục này đo trên **val** với mô hình `F01` seed 0 (trừ ensemble), sheet `Inference` và `Latency`:

| exp_id | Phương pháp | K | macro-F1 val | top-1 val | ECE val | p50 batch-1 (ms) | Chi phí so với `I00` |
|---|---|---|---|---|---|---|---|
| I00 | 1 view: resize 256 + center crop 224 (mốc) | 1 | 0,9568 | 0,9663 | 0,0079 | 6,07 | 1,00 |
| I01 | TTA lật ngang, gộp xác suất | 2 | 0,9545 | 0,9657 | 0,0095 | 12,09 | 1,99 |
| I02a | TTA 5 crop, gộp xác suất | 5 | 0,9595 | 0,9692 | 0,0068 | 30,46 | 5,01 |
| I02b | TTA 5 crop + lật, gộp xác suất | 10 | 0,9573 | 0,9674 | 0,0073 | 61,53 | 10,13 |
| I02c | TTA đa tỉ lệ 224/256/288, gộp xác suất | 3 | 0,9666 | 0,9746 | 0,0062 | 21,91 | 3,61 |
| **I03d** | **TTA đa tỉ lệ 224/256/288, gộp logit** | 3 | **0,9669** | **0,9749** | 0,0066 | 21,91 | 3,61 |
| I04a | 1 view toàn ảnh 224 | 1 | 0,9548 | 0,9663 | 0,0108 | 6,07 | 1,00 |
| I04b | 1 view toàn ảnh 256 | 1 | 0,9609 | 0,9700 | 0,0047 | 7,09 | 1,17 |
| I04c | 1 view toàn ảnh 288 | 1 | 0,9639 | 0,9726 | 0,0076 | 8,74 | 1,44 |
| I04d | 1 view toàn ảnh 320 | 1 | 0,9653 | 0,9743 | 0,0056 | 10,72 | 1,76 |
| I05a | ensemble 3 seed của `F01` | 3 | 0,9606 | 0,9709 | 0,0108 | 18,22 | 3,00 |
| I05b | ensemble `F01` + EfficientNet-B0 (`B06`) | 2 | 0,9448 | 0,9597 | 0,0836 | 13,79 | 2,27 |
| I06 | trọng số EMA (`T12`) so với trọng số thô cùng lần chạy (0,9542) | 1 | 0,9558 | 0,9660 | 0,0058 | 6,07 | 1,00 |
| I07 | temperature scaling (T = 1,052) | 1 | 0,9568 | 0,9663 | 0,0058 | 6,07 | 1,00 |
| I08b | FP16 (`model.half()`) | 1 | 0,9571 | 0,9666 | 0,0085 | 5,59 | 0,92 |
| I08c | AMP (autocast) | 1 | 0,9568 | 0,9663 | 0,0083 | 7,60 | 1,25 |
| I08a, I08d | gộp BN vào conv (FP32, FP16) | — | không áp dụng | | | | |

Các cặp `I03a`–`I03c` (gộp logit cho lật, 5 crop, 5 crop + lật) lệch cặp gộp xác suất tương ứng dưới 0,001; xem sheet
`Inference`. Gộp BN không áp dụng vì ConvNeXt dùng LayerNorm, không có BatchNorm.

**Độ trễ** (sheet `Latency`; Tesla T4, ảnh 224, 10 lần warmup, 100 lần đo, có `cuda.synchronize`, **chưa gồm** tiền xử
lý và đọc đĩa):

| Cấu hình | dtype | batch | p50 (ms) | p95 (ms) | p99 (ms) | ảnh/s |
|---|---|---|---|---|---|---|
| `F01` 1 view | FP32 | 1 | 6,07 | 9,16 | 9,17 | 165 |
| `F01` 1 view | FP16 | 1 | 5,59 | 6,20 | 6,51 | 179 |
| `F01` 1 view | AMP | 1 | 7,60 | 8,46 | 9,06 | 132 |
| `F01` 1 view | FP32 | 32 | 168,8 | 172,3 | 173,2 | 190 |
| `F01` 1 view | FP16 | 32 | 44,6 | 45,9 | 46,2 | 718 |
| `F01` TTA K=5 (tuần tự) | FP32 | 1 | 30,46 | 31,15 | 31,20 | 33 |
| `F01` TTA K=10 (tuần tự) | FP32 | 1 | 61,53 | 62,39 | 66,08 | 16 |
| `F01` toàn ảnh 320 | FP32 | 1 | 10,72 | 11,17 | 14,57 | 93 |
| `F01` đa tỉ lệ 224/256/288 (`I03d`) | FP32 | 1 | 21,91 | 30,82 | 30,91 | — |
| EfficientNet-B0 (`B06`) | FP32 | 1 | 7,71 | 8,57 | 9,00 | 130 |

Độ trễ của `I03d` là tổng độ trễ đo riêng ở ba độ phân giải, không phải một phép đo liền.

![Đánh đổi F1 và độ trễ](figures/tradeoff_f1_vs_latency.png)
![Độ tin cậy trên val](figures/reliability_val.png)

Nhận xét:

- **Độ phân giải là yếu tố suy luận quan trọng nhất.** F1 val tăng đều theo kích thước ảnh toàn khung: 0,9548 (224),
  0,9609 (256), 0,9639 (288), 0,9653 (320). Điều này phù hợp với hiệu ứng FixRes: train bằng `RandomResizedCrop` làm vật
  thể lúc train to hơn lúc test ở cùng độ phân giải. Gộp ba tỉ lệ (`I03d`) cho +0,0101 so với `I00` (khoảng 2,5 lần
  std_T00) với chi phí 3,6 lần.
- **TTA lật và crop không đáng chi phí.** Lật ngang cho −0,002 (trong nhiễu) với chi phí gấp đôi; 5 crop cho +0,003 với
  chi phí 5 lần; 10 view không hơn 5 view.
- **Gộp xác suất và gộp logit** không phân biệt được (lệch dưới 0,001 ở mọi bộ view).
- **Ensemble 3 seed** (+0,0038, chi phí 3 lần) kém hơn 1 mô hình ở 320 (+0,0085, chi phí 1,76 lần). Ensemble với
  EfficientNet-B0 làm **giảm** F1 0,012 và ECE tăng lên 0,084 vì mô hình yếu kéo trung bình xuống.
- **FP16** giữ nguyên F1 (+0,0003) và giảm p95 batch-1 từ 9,16 xuống 6,20 ms; ở batch 32 nhanh gấp 3,8 lần FP32. AMP
  chậm hơn FP32 ở batch 1 (7,60 so với 6,07 ms).
- **Hiệu chuẩn: temperature scaling không giúp một cách đo được.** Mô hình vốn đã hiệu chuẩn tốt (T khớp trên val =
  1,06–1,09). Trên test, ECE của `F01` trước TS là 0,0062 ± 0,0009 và sau TS là 0,0061 ± 0,0016: chênh 0,0001, nhỏ hơn
  std nhiều lần, và hai trong ba seed có ECE tăng (seed 0: 0,0053 → 0,0061; seed 2: 0,0070 → 0,0077). Ước lượng cross-fit
  2 fold trên val còn cho ECE tăng (0,0092 → 0,0123). Con số ECE val 0,0079 → 0,0058 ở `I07` lạc quan vì khớp và đo
  trên cùng tập.
- **Ngoại tuyến hay thời gian thực:** `I03d` hợp khi cần độ chính xác cao nhất (p95 30,8 ms trên T4); 1 view FP16 hợp khi
  cần độ trễ thấp nhất; 1 view ở 320 là điểm giữa tốt nhất trên val (gần bằng `I03d` với nửa chi phí) nhưng chưa được
  đánh giá trên test.

## 6. Cấu hình tốt nhất

**Mô tả để tái lập (`F01`):** backbone `convnext_tiny.fb_in1k` (timm) · công thức `T00` (bảng ở mục 2) · suy luận: ba
view toàn ảnh 224, 256, 288, gộp trung bình logit, rồi chia cho T khớp trên val theo từng seed (1,082 / 1,061 / 1,089) ·
seed 0, 1, 2 · chi tiết trong `decisions.json` và `logs/F01/seed*/config.json`.

**Kết quả test** (sheet `Final`; mỗi (exp_id, seed) chạy test đúng một lần; tính lại bằng `eval.py score` trên
`predictions/`, kết quả ở `scores/`):

| Cấu hình | seed | macro-F1 val | macro-F1 test | top-1 test | balanced acc test | ECE test | recall Chinee apple | recall Snake weed |
|---|---|---|---|---|---|---|---|---|
| `F01` chung kết | 0 | 0,9669 | 0,9652 | 0,9735 | 0,9598 | 0,0061 | 0,9381 | 0,9216 |
| `F01` chung kết | 1 | 0,9622 | 0,9643 | 0,9735 | 0,9620 | 0,0045 | 0,9513 | 0,8971 |
| `F01` chung kết | 2 | 0,9648 | 0,9653 | 0,9732 | 0,9564 | 0,0077 | 0,9248 | 0,8922 |
| **`F01` chung kết** | **mean ± std** | 0,9646 ± 0,0023 | **0,9649 ± 0,0006** | **0,9734 ± 0,0002** | 0,9594 ± 0,0028 | 0,0061 ± 0,0016 | 0,9381 ± 0,0133 | 0,9036 ± 0,0158 |
| `T00` mốc (`T00 + I00`) | 0 | 0,9568 | 0,9547 | 0,9635 | 0,9575 | 0,0061 | 0,9027 | 0,9314 |
| `T00` mốc | 1 | 0,9492 | 0,9459 | 0,9575 | 0,9533 | 0,0092 | 0,8982 | 0,9118 |
| `T00` mốc | 2 | 0,9551 | 0,9486 | 0,9607 | 0,9441 | 0,0098 | 0,8628 | 0,9020 |
| **`T00` mốc** | **mean ± std** | 0,9537 ± 0,0040 | **0,9497 ± 0,0045** | 0,9606 ± 0,0030 | 0,9516 ± 0,0069 | 0,0083 ± 0,0020 | 0,8879 ± 0,0218 | 0,9150 ± 0,0150 |
| `F01uncal` (chưa TS) | mean ± std | — | 0,9649 ± 0,0006 | 0,9734 ± 0,0002 | 0,9594 ± 0,0028 | 0,0062 ± 0,0009 | 0,9381 ± 0,0133 | 0,9036 ± 0,0158 |
| `F01rt` (1 view FP32) | mean ± std | — | 0,9497 ± 0,0045 | 0,9606 ± 0,0030 | 0,9516 ± 0,0069 | 0,0083 ± 0,0020 | 0,8879 ± 0,0218 | 0,9150 ± 0,0150 |

`F01rt` trùng `T00` từng số vì công thức cuối chính là `T00`: cùng trọng số, cùng 1 view.

**So với mốc:** Δ macro-F1 test = 0,9649 − 0,9497 = **+0,0152**, s = 0,0045 (std lớn hơn trong hai nhóm), Δ/s ≈ 3,4.
Top-1 tăng 1,28 điểm phần trăm. Std giữa các seed giảm từ 0,0045 xuống 0,0006. Chênh macro-F1 giữa val và test của `F01`
là 0,0003.

**Theo lớp (test, mean ± std 3 seed, sheet `PerClass`):**

| Lớp | Số ảnh | `F01` precision | `F01` recall | `F01` F1 | `T00` recall | `T00` F1 |
|---|---|---|---|---|---|---|
| Chinee apple | 226 | 0,948 ± 0,014 | 0,938 ± 0,013 | 0,943 ± 0,001 | 0,888 ± 0,022 | 0,911 ± 0,008 |
| Lantana | 213 | 0,984 ± 0,003 | 0,975 ± 0,011 | 0,980 ± 0,005 | 0,980 ± 0,015 | 0,946 ± 0,004 |
| Parkinsonia | 207 | 0,961 ± 0,020 | 0,990 ± 0,005 | 0,975 ± 0,008 | 0,992 ± 0,003 | 0,956 ± 0,020 |
| Parthenium | 205 | 0,986 ± 0,006 | 0,945 ± 0,012 | 0,965 ± 0,007 | 0,943 ± 0,003 | 0,963 ± 0,007 |
| Prickly acacia | 213 | 0,956 ± 0,007 | 0,953 ± 0,005 | 0,955 ± 0,003 | 0,939 ± 0,016 | 0,936 ± 0,004 |
| Rubber vine | 202 | 0,977 ± 0,015 | 0,957 ± 0,008 | 0,967 ± 0,008 | 0,949 ± 0,008 | 0,952 ± 0,013 |
| Siam weed | 215 | 0,986 ± 0,004 | 0,983 ± 0,007 | 0,984 ± 0,001 | 0,988 ± 0,003 | 0,979 ± 0,006 |
| Snake weed | 204 | 0,964 ± 0,015 | 0,904 ± 0,016 | 0,933 ± 0,003 | 0,915 ± 0,015 | 0,930 ± 0,007 |
| Negative | 1.822 | 0,977 ± 0,005 | 0,990 ± 0,003 | 0,983 ± 0,001 | 0,971 ± 0,009 | 0,974 ± 0,002 |

Hai lớp khó: Chinee apple recall 93,8% và Snake weed recall 90,4% (bài báo, *trích dẫn*: 88,5% và 88,8%). TTA đa tỉ lệ
nâng recall Chinee apple 5,0 điểm; recall Snake weed giảm 1,1 điểm, nằm trong std (1,5–1,6 điểm), tức không phân biệt được.

**Điểm tự chấm phần I** bằng `eval.py grade` (`scores/grade_I.md`, đề xuất, ngưỡng tạm thời): I1 7/7, I2 5/5, I3 4/4,
I4a 1/1, I4b 1/1, I5 2/2. Lưu ý I4a đạt nhờ ECE giảm 0,0001, không có ý nghĩa thống kê (xem mục 5).

![Ma trận nhầm lẫn](figures/confusion_test.png)

**Phân tích lỗi** (ma trận cộng 3 seed; `scores/top_confusions.csv`):

| Nhãn thật → dự đoán | Số ảnh (tổng 3 seed) | % theo hàng |
|---|---|---|
| Snake weed → Negative | 35 | 5,7% |
| Chinee apple → Negative | 31 | 4,6% |
| Rubber vine → Negative | 23 | 3,8% |
| Snake weed → Chinee apple | 18 | 2,9% |
| Parthenium → Prickly acacia | 16 | 2,6% |
| Prickly acacia → Negative | 15 | 2,3% |
| Chinee apple → Snake weed | 8 | 1,2% |

- **Lỗi lớn nhất là bỏ sót cỏ thành `Negative`**, không phải nhầm giữa hai loài: ba dòng đầu đều là "loài cỏ →
  Negative". Với robot diệt cỏ đây là lỗi bỏ sót, cần quan tâm hơn lỗi nhầm loài.
- **Cặp Chinee apple ↔ Snake weed** vẫn là cặp nhầm giữa hai loài nhiều nhất, và bất đối xứng: 2,9% Snake weed bị đoán
  thành Chinee apple, 1,2% chiều ngược lại (bài báo, *trích dẫn*: 4,1% và 3,4%).
- **Nhìn ảnh sai** (`figures/errors_test.png`, 8 ảnh nhầm trong cặp này ở seed 0; danh sách ở `scores/errors_seed0.csv`):
  các ảnh đều là thảm thực vật dày, nhiều loài lẫn trong khung, có bóng đổ gắt hoặc ánh sáng chói, và lá của hai loài
  cùng nhỏ, hình trứng. Độ tin cậy của dự đoán sai từ 0,41 đến 0,81, tức mô hình không chắc ở nhiều ảnh. Giả thuyết: mô
  hình dựa vào kết cấu lá ở tỉ lệ nhỏ, bị nhiễu khi vật thể mục tiêu chỉ chiếm một phần ảnh; điều này cũng giải thích
  vì sao tăng độ phân giải kiểm tra giúp Chinee apple. Một số ảnh có thể chứa cả hai loài (nhãn nhiễu); tôi chưa kiểm
  chứng được điều này.

![Ảnh bị đoán sai](figures/errors_test.png)

Phân tích lỗi chỉ để mô tả; không dùng để sửa cấu hình rồi chạy lại test.

## 7. Kết luận và khuyến nghị

1. **Cấu hình tốt nhất** là ConvNeXt-Tiny + công thức nền `T00` + TTA đa tỉ lệ 224/256/288 gộp logit + temperature
   scaling: macro-F1 test 0,9649 ± 0,0006, top-1 0,9734 ± 0,0002 (3 seed). Hơn mốc `T00 + I00` +0,0152 macro-F1, **vượt
   nhiễu** (s = 0,0045).
2. **Yếu tố đóng góp nhiều nhất**, theo thứ tự độ lớn của Δ:
   - *Khởi tạo tiền huấn luyện:* bỏ đi mất 0,27–0,375 macro-F1 val (`T01`, `T02`).
   - *Backbone:* dưới cùng công thức, ConvNeXt-Tiny hơn ResNet-50 0,146 macro-F1 val, nhưng khoảng cách này lẫn với việc
     công thức nền không hợp nhóm BatchNorm (mục 3). Trong ba mạng tốt nhất, chênh lệch chỉ 0,002–0,012.
   - *Suy luận:* +0,0101 trên val và +0,0152 trên test nhờ đa tỉ lệ, là phần cải thiện duy nhất so với mốc.
   - *Các yếu tố huấn luyện còn lại* (augmentation, loss, LR head, EMA, độ phân giải train): không yếu tố nào phân biệt
     được với nhiễu; sampler cân bằng lớp làm hại (−0,0137).
3. **Triển khai trên robot, ngân sách 30–100 ms mỗi khung:** tôi chọn ConvNeXt-Tiny 1 view. Ở FP32 batch 1 trên Tesla T4,
   p95 = 9,16 ms và macro-F1 test = 0,9497 ± 0,0045 (`F01rt`). Chuyển sang FP16 giảm p95 xuống 6,20 ms với F1 val không
   đổi (0,9571 so với 0,9568); F1 test của bản FP16 chưa được đo riêng. Cấu hình chung kết đa tỉ lệ cũng nằm trong ngân
   sách trên T4 (p95 khoảng 30,8 ms). Hai lưu ý: độ trễ **chưa gồm** tiền xử lý và đọc ảnh, và T4 mạnh hơn phần cứng
   nhúng nhiều (bài báo, *trích dẫn*: ResNet-50 trên Jetson TX2 mất 53,4–180 ms), nên phải đo lại trên thiết bị thật.
   Nếu thiết bị còn dư ngân sách, 1 view ở 320 đáng thử trước (F1 val 0,9653, chi phí 1,76 lần) nhưng cần đánh giá trên
   dữ liệu độc lập.

## 8. Hạn chế và việc tiếp theo

- **Một fold, 3 seed.** Std ước lượng từ 3 seed có sai số lớn; chênh lệch nhỏ hơn nhiễu không kết luận được.
- **Chia ngẫu nhiên, không theo địa điểm.** Ảnh cùng địa điểm, cùng ngày có thể nằm ở cả train và test, nên điểm test có
  thể lạc quan so với khi gặp địa điểm, mùa hay camera mới. Temperature T khớp trên val cũng không còn đáng tin khi
  lệch miền.
- **So sánh backbone thiên về ConvNeXt/transformer.** Nhóm BatchNorm thiếu khớp với công thức nền; chưa chạy lại với LR
  cao hơn hay nhiều epoch hơn, nên không kết luận được về kiến trúc. Mỗi backbone chỉ 1 seed.
- **Ablation 1 seed, trên một backbone.** Tương tác giữa backbone và công thức chưa kiểm chứng. `T00` seed 0 là seed cao
  nhất, làm mọi Δ lệch âm khoảng 0,003.
- **Không có thí nghiệm kết hợp (`T20`)** vì không yếu tố nào thắng theo quy tắc đặt trước; các thí nghiệm mở rộng
  `T14`–`T19` (lật dọc, crop nhẹ, SGD, không warmup, drop path, 20 epoch) không chạy để giữ hạn mức GPU.
- **Hiệu chuẩn:** temperature scaling không cải thiện ECE một cách đo được; I4a đạt ngưỡng chỉ nhờ chênh 0,0001.
- **Độ trễ** đo trên Tesla T4, batch 1, chưa gồm tiền xử lý; độ trễ `I03d` là tổng ba phép đo riêng. Cấu hình thời gian
  thực FP16 có độ trễ đo riêng nhưng F1 test chỉ có cho bản FP32.
- **Ngân sách ngắn hơn bài báo:** 12 epoch so với khoảng 100 epoch (*trích dẫn*), augmentation đơn giản hơn; định nghĩa
  accuracy của bài báo (weighted average) cũng khác top-1 ở đây, nên so sánh với bài báo chỉ mang tính tham khảo.
- **Việc tiếp theo:** chạy lại ResNet-50 với LR 1e-3 để tách kiến trúc khỏi công thức; đánh giá 1 view ở 320 và train ở
  độ phân giải cao hơn; chạy fold 1–4; thử chia theo địa điểm; chưng cất ConvNeXt sang mạng nhẹ cho thiết bị nhúng.

## 9. Phụ lục

- **Danh sách `exp_id` và cấu hình đầy đủ:** `results.xlsx` (sheet `Backbones`, `Training`, `Inference`, `Final`,
  `PerClass`, `Latency`, `Summary`) và `logs/<exp_id>/seed<k>/{config.json, summary.json, history.csv}`.
- **File dự đoán:** `predictions/` gồm `F01`, `F01_uncal`, `F01rt`, `T00` trên test (3 seed) và `F01` trên val (3 seed).
- **Kiểm tra pipeline:** `figures/sanity.json`, `figures/augmentation_preview.png`, `figures/sanity_overfit.png`.
- **Quy trình chạy trên Kaggle:** Version 1 chạy Bước 0–3 (chỉ train và val, 4 giờ 52 phút). Version sau khôi phục
  `runs/` của Version 1, chạy lại Bước 3 trên val rồi mở test một lần ở Bước 4. Số trong bảng suy luận và độ trễ là của
  lần chạy sau.
- **Notebook:** `code/lab_day2.ipynb` · link và cách chạy lại ở `README.md`.
- **Công cụ hỗ trợ:** có dùng trợ lý AI (Claude Code) để viết và rà code, và để soạn thảo báo cáo từ số liệu đã chạy.
  Mọi thí nghiệm do tôi chạy trên tài khoản Kaggle của mình; mọi con số truy ngược được tới `logs/` và `predictions/`.
