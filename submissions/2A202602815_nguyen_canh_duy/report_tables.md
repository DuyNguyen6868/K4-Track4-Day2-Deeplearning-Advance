# Bảng sinh tự động từ số thật (results.py)

## Backbone (B)
| exp_id | backbone | tag trọng số | #tham số (M) | GMAC | macro-F1 val | top-1 val | thời gian train/epoch (s) | độ trễ batch-1 p50 (ms) | ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | resnet50.a1_in1k [timm/resnet50.a1_in1k] | 23.5265 | 4.0872 | 0.8104 | 0.8603 | 34.0421 | 6.1658 |  |
| B02 | resnext50 | resnext50_32x4d.a1h_in1k [timm/resnext50_32x4d.a1h_in1k] | 22.9983 | 4.2285 | 0.7538 | 0.8112 | 43.5890 | 9.3905 |  |
| B03 | convnext_tiny | convnext_tiny.fb_in1k [timm/convnext_tiny.fb_in1k] | 27.8270 | 4.4548 | 0.9568 | 0.9663 | 51.6950 | 5.8607 | được chọn đi tiếp (backbone chính) |
| B04 | deit_small | deit_small_patch16_224.fb_in1k [timm/deit_small_patch16_224.fb_in1k] | 21.6691 | 4.2408 | 0.9443 | 0.9594 | 34.7298 | 5.4181 |  |
| B05 | swin_tiny | swin_tiny_patch4_window7_224.ms_in1k [timm/swin_tiny_patch4_window7_224.ms_in1k] | 27.5263 | 4.4898 | 0.9544 | 0.9652 | 65.6297 | 9.8076 |  |
| B06 | efficientnet_b0 | efficientnet_b0.ra_in1k [timm/efficientnet_b0.ra_in1k] | 4.0191 | 0.3845 | 0.7694 | 0.8298 | 26.1718 | 7.9242 | mạng nhẹ tốt nhất (dùng cho ensemble I05b) |
| B07 | mobilenetv3 | mobilenetv3_large_100.ra_in1k [timm/mobilenetv3_large_100.ra_in1k] | 4.2136 | 0.2153 | 0.7121 | 0.7912 | 17.6570 | 6.5599 |  |

## Công thức huấn luyện (T)
| exp_id | trục thay đổi (A-G) | khác T00 ở điểm nào | seed | macro-F1 val | Δ macro-F1 so với T00 | Δ / std(T00) | F1 val Chinee Apple | F1 val Snake Weed | thắng (Δ >= ngưỡng) |
|---|---|---|---|---|---|---|---|---|---|
| T00 | — | công thức nền (T00) | 0 | 0.9568 | 0.0000 | 0.0000 | 0.9244 | 0.9204 |  |
| T00 | — | công thức nền (T00) | 1 | 0.9492 | -0.0076 | -1.9106 | 0.9206 | 0.9181 |  |
| T00 | — | công thức nền (T00) | 2 | 0.9551 | -0.0018 | -0.4433 | 0.9209 | 0.9082 |  |
| T01 | A | init=frozen: đóng băng backbone, chỉ train head | 0 | 0.6862 | -0.2706 | -68.1837 | 0.6200 | 0.5929 | False |
| T02 | A | init=scratch (không tiền huấn luyện); LR toàn mạng 1e-3 vì 1e-4 là LR của fine-tune | 0 | 0.5818 | -0.3750 | -94.4846 | 0.4444 | 0.5000 | False |
| T03 | B | aug=color (+ ColorJitter) | 0 | 0.9507 | -0.0061 | -1.5496 | 0.8961 | 0.9100 | False |
| T04 | B | aug=trivial (+ TrivialAugmentWide) | 0 | 0.9520 | -0.0049 | -1.2237 | 0.9007 | 0.8844 | False |
| T05 | B | mix=cutmix, alpha=1, xác suất 0,5 mỗi batch | 0 | 0.9555 | -0.0013 | -0.3372 | 0.9062 | 0.9051 | False |
| T06 | B | mix=mixup, alpha=1, xác suất 0,5 mỗi batch | 0 | 0.9534 | -0.0035 | -0.8714 | 0.9107 | 0.9104 | False |
| T07 | C | loss=label smoothing eps=0,1 | 0 | 0.9521 | -0.0047 | -1.1964 | 0.9087 | 0.9136 | False |
| T08 | C | loss=focal gamma=2 | 0 | 0.9503 | -0.0065 | -1.6396 | 0.9022 | 0.9127 | False |
| T09 | C | loss=CE có trọng số 1/n_c (chuẩn hoá trung bình 1) | 0 | 0.9488 | -0.0080 | -2.0274 | 0.8972 | 0.9046 | False |
| T10 | D | sampler cân bằng lớp (oversample) | 0 | 0.9431 | -0.0137 | -3.4613 | 0.9038 | 0.9043 | False |
| T11 | E | LR head = LR backbone = 1e-4 (không gấp 10 lần) | 0 | 0.9492 | -0.0076 | -1.9157 | 0.8939 | 0.8969 | False |
| T12 | F | EMA trọng số, decay 0,998 | 0 | 0.9558 | -0.0010 | -0.2498 | 0.9238 | 0.9185 | False |
| T13 | G | độ phân giải huấn luyện 256 thay vì 224 | 0 | 0.9557 | -0.0011 | -0.2699 | 0.9234 | 0.9177 | False |

## Suy luận (I)
| exp_id | phương pháp | K (số view hoặc số mô hình) | macro-F1 val | top-1 val | ECE val | p50 batch-1 (ms) | chi phí tương đối so với I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view (mốc): resize 256 + center crop | 1 | 0.9568 | 0.9663 | 0.0079 | 6.0739 | 1.0000 |
| I01 | TTA lật ngang (K=2) [gộp xác suất] | 2 | 0.9545 | 0.9657 | 0.0095 | 12.0903 | 1.9905 |
| I02a | TTA 5 crop (K=5) [gộp xác suất] | 5 | 0.9595 | 0.9692 | 0.0068 | 30.4568 | 5.0144 |
| I02b | TTA 5 crop + lật (K=10) [gộp xác suất] | 10 | 0.9573 | 0.9674 | 0.0073 | 61.5303 | 10.1303 |
| I02c | TTA đa tỉ lệ (224, 256, 288) (K=3) [gộp xác suất] | 3 | 0.9666 | 0.9746 | 0.0062 | 21.9060 | 3.6066 |
| I03a | TTA lật ngang (K=2) [gộp logit] | 2 | 0.9553 | 0.9660 | 0.0101 | 12.0903 | 1.9905 |
| I03b | TTA 5 crop (K=5) [gộp logit] | 5 | 0.9592 | 0.9689 | 0.0068 | 30.4568 | 5.0144 |
| I03c | TTA 5 crop + lật (K=10) [gộp logit] | 10 | 0.9566 | 0.9672 | 0.0064 | 61.5303 | 10.1303 |
| I03d | TTA đa tỉ lệ (224, 256, 288) (K=3) [gộp logit] | 3 | 0.9669 | 0.9749 | 0.0066 | 21.9060 | 3.6066 |
| I04a | độ phân giải kiểm tra 224 (toàn ảnh) | 1 | 0.9548 | 0.9663 | 0.0108 | 6.0739 | 1.0000 |
| I04b | độ phân giải kiểm tra 256 (toàn ảnh) | 1 | 0.9609 | 0.9700 | 0.0047 | 7.0889 | 1.1671 |
| I04c | độ phân giải kiểm tra 288 (toàn ảnh) | 1 | 0.9639 | 0.9726 | 0.0076 | 8.7433 | 1.4395 |
| I04d | độ phân giải kiểm tra 320 (toàn ảnh) | 1 | 0.9653 | 0.9743 | 0.0056 | 10.7182 | 1.7646 |
| I05a | ensemble 3 seed của F01 (trung bình xác suất) | 3 | 0.9606 | 0.9709 | 0.0108 | 18.2216 | 3.0000 |
| I05b | ensemble F01 + efficientnet_b0 (B06, công thức nền) | 2 | 0.9448 | 0.9597 | 0.0836 | 13.7910 | 2.2705 |
| I06 | trọng số EMA (so với trọng số thô cùng lần chạy) | 1 | 0.9558 | 0.9660 | 0.0058 | 6.0739 | 1.0000 |
| I07 | temperature scaling (T = 1.052, khớp trên val) | 1 | 0.9568 | 0.9663 | 0.0058 | 6.0739 | 1.0000 |
| I08a | gộp BN vào conv (FP32) | 1 |  |  |  |  |  |
| I08b | FP16 (model.half()) | 1 | 0.9571 | 0.9666 | 0.0085 | 5.5927 | 0.9208 |
| I08c | AMP (autocast) | 1 | 0.9568 | 0.9663 | 0.0083 | 7.5971 | 1.2508 |
| I08d | FP16 + gộp BN | 1 |  |  |  |  |  |

## Chung kết và mốc (test)
| exp_id | cấu hình | seed | macro-F1 val | macro-F1 test | top-1 test | balanced acc test | ECE test | recall Chinee Apple | recall Snake Weed |
|---|---|---|---|---|---|---|---|---|---|
| F01 | cấu hình chung kết (backbone + công thức + suy luận + TS) | 0 | 0.9669 | 0.9652 | 0.9735 | 0.9598 | 0.0061 | 0.9381 | 0.9216 |
| F01 | cấu hình chung kết (backbone + công thức + suy luận + TS) | 1 | 0.9622 | 0.9643 | 0.9735 | 0.9620 | 0.0045 | 0.9513 | 0.8971 |
| F01 | cấu hình chung kết (backbone + công thức + suy luận + TS) | 2 | 0.9648 | 0.9653 | 0.9732 | 0.9564 | 0.0077 | 0.9248 | 0.8922 |
| F01 | cấu hình chung kết (backbone + công thức + suy luận + TS) | mean ± std (3 seed) | 0.9646 ± 0.0023 | 0.9649 ± 0.0006 | 0.9734 ± 0.0002 | 0.9594 ± 0.0028 | 0.0061 ± 0.0016 | 0.9381 ± 0.0133 | 0.9036 ± 0.0158 |
| T00 | mốc: T00 + I00 | 0 | 0.9568 | 0.9547 | 0.9635 | 0.9575 | 0.0061 | 0.9027 | 0.9314 |
| T00 | mốc: T00 + I00 | 1 | 0.9492 | 0.9459 | 0.9575 | 0.9533 | 0.0092 | 0.8982 | 0.9118 |
| T00 | mốc: T00 + I00 | 2 | 0.9551 | 0.9486 | 0.9607 | 0.9441 | 0.0098 | 0.8628 | 0.9020 |
| T00 | mốc: T00 + I00 | mean ± std (3 seed) | 0.9537 ± 0.0040 | 0.9497 ± 0.0045 | 0.9606 ± 0.0030 | 0.9516 ± 0.0069 | 0.0083 ± 0.0020 | 0.8879 ± 0.0218 | 0.9150 ± 0.0150 |
| F01uncal | chung kết, chưa temperature scaling | 0 |  | 0.9652 | 0.9735 | 0.9598 | 0.0053 | 0.9381 | 0.9216 |
| F01uncal | chung kết, chưa temperature scaling | 1 |  | 0.9643 | 0.9735 | 0.9620 | 0.0062 | 0.9513 | 0.8971 |
| F01uncal | chung kết, chưa temperature scaling | 2 |  | 0.9653 | 0.9732 | 0.9564 | 0.0070 | 0.9248 | 0.8922 |
| F01uncal | chung kết, chưa temperature scaling | mean ± std (3 seed) |  | 0.9649 ± 0.0006 | 0.9734 ± 0.0002 | 0.9594 ± 0.0028 | 0.0062 ± 0.0009 | 0.9381 ± 0.0133 | 0.9036 ± 0.0158 |
| F01rt | cấu hình thời gian thực (1 view FP32 của mô hình F01) | 0 |  | 0.9547 | 0.9635 | 0.9575 | 0.0061 | 0.9027 | 0.9314 |
| F01rt | cấu hình thời gian thực (1 view FP32 của mô hình F01) | 1 |  | 0.9459 | 0.9575 | 0.9533 | 0.0092 | 0.8982 | 0.9118 |
| F01rt | cấu hình thời gian thực (1 view FP32 của mô hình F01) | 2 |  | 0.9486 | 0.9607 | 0.9441 | 0.0098 | 0.8628 | 0.9020 |
| F01rt | cấu hình thời gian thực (1 view FP32 của mô hình F01) | mean ± std (3 seed) |  | 0.9497 ± 0.0045 | 0.9606 ± 0.0030 | 0.9516 ± 0.0069 | 0.0083 ± 0.0020 | 0.8879 ± 0.0218 | 0.9150 ± 0.0150 |

## So với mốc
| chỉ số | chung kết (mean) | mốc T00+I00 (mean) | Δ | s = std lớn hơn | Δ > s | điểm I2 đề xuất |
|---|---|---|---|---|---|---|
| macro-F1 test | 0.9649 | 0.9497 | 0.0152 | 0.0045 | True | 5 |

## Theo lớp (test)
| cấu hình | lớp | số ảnh test | precision | recall | F1 |
|---|---|---|---|---|---|
| F01 | Chinee apple | 226 | 0.948 ± 0.014 | 0.938 ± 0.013 | 0.943 ± 0.001 |
| F01 | Lantana | 213 | 0.984 ± 0.003 | 0.975 ± 0.011 | 0.980 ± 0.005 |
| F01 | Parkinsonia | 207 | 0.961 ± 0.020 | 0.990 ± 0.005 | 0.975 ± 0.008 |
| F01 | Parthenium | 205 | 0.986 ± 0.006 | 0.945 ± 0.012 | 0.965 ± 0.007 |
| F01 | Prickly acacia | 213 | 0.956 ± 0.007 | 0.953 ± 0.005 | 0.955 ± 0.003 |
| F01 | Rubber vine | 202 | 0.977 ± 0.015 | 0.957 ± 0.008 | 0.967 ± 0.008 |
| F01 | Siam weed | 215 | 0.986 ± 0.004 | 0.983 ± 0.007 | 0.984 ± 0.001 |
| F01 | Snake weed | 204 | 0.964 ± 0.015 | 0.904 ± 0.016 | 0.933 ± 0.003 |
| F01 | Negative | 1822 | 0.977 ± 0.005 | 0.990 ± 0.003 | 0.983 ± 0.001 |
| T00 | Chinee apple | 226 | 0.937 ± 0.021 | 0.888 ± 0.022 | 0.911 ± 0.008 |
| T00 | Lantana | 213 | 0.915 ± 0.007 | 0.980 ± 0.015 | 0.946 ± 0.004 |
| T00 | Parkinsonia | 207 | 0.923 ± 0.039 | 0.992 ± 0.003 | 0.956 ± 0.020 |
| T00 | Parthenium | 205 | 0.983 ± 0.013 | 0.943 ± 0.003 | 0.963 ± 0.007 |
| T00 | Prickly acacia | 213 | 0.933 ± 0.008 | 0.939 ± 0.016 | 0.936 ± 0.004 |
| T00 | Rubber vine | 202 | 0.955 ± 0.022 | 0.949 ± 0.008 | 0.952 ± 0.013 |
| T00 | Siam weed | 215 | 0.971 ± 0.009 | 0.988 ± 0.003 | 0.979 ± 0.006 |
| T00 | Snake weed | 204 | 0.946 ± 0.002 | 0.915 ± 0.015 | 0.930 ± 0.007 |
| T00 | Negative | 1822 | 0.976 ± 0.006 | 0.971 ± 0.009 | 0.974 ± 0.002 |
