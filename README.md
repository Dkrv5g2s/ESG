# AI CUP VeriPromiseESG 2026 Baseline

本專案整理 AI CUP「ESG 永續承諾驗證競賽 2026」的 baseline 程式碼與繳交檔產生流程。主要程式碼放在倉庫根目錄，競賽相關說明集中在 `docs/`，方便評審或隊友直接查看與執行。

## 專案結構

```text
AICUP/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ baseline_reference.py
├─ ours.py
├─ data/
│  ├─ vpesg4k_train_1000.json
│  ├─ vpesg4k_test_2000.csv
│  ├─ vpesg4k_test_2000.json
│  ├─ vpesg4k_val_1000.csv
│  └─ vpesg4k_val_1000.json
├─ outputs/
│  └─ submission.csv
├─ tests/
│  └─ test_submission_format.py
└─ docs/
   ├─ introduction.txt
   ├─ rule.txt
   ├─ data_explain.txt
   ├─ submit_format.txt
   └─ baseline_reference.ipynb
```


## 程式版本

| 檔案 | 用途 |
| --- | --- |
| `baseline_reference.py` | 官方 baseline notebook 對應的 Python 版，對照 `docs/baseline_reference.ipynb` |
| `ours.py` | 改良版訓練流程與模型架構 |

## 子任務與欄位對照

官方評分分成四個子任務；程式與繳交檔仍須使用官方資料集指定欄位名稱，不能改成子任務英文名。

| 官方子任務 | 程式與繳交欄位 | 權重 | 類別 |
| --- | --- | --- | --- |
| Commitment Classification | `promise_status` | 20% | `Yes`、`No` |
| Evidence Identification | `evidence_status` | 30% | `Yes`、`No`、`N/A` |
| Clarity Classification | `evidence_quality` | 35% | `Clear`、`Not Clear`、`Misleading`、`N/A` |
| Timeline Classification | `verification_timeline` | 15% | `already`、`within_2_years`、`between_2_and_5_years`、`more_than_5_years`、`N/A` |

## 環境安裝

建議使用 Conda 建立獨立環境。PyTorch 官方目前主要提供 pip wheel 安裝指令；因此本專案使用 Conda 管理 Python 與資料科學套件，並在 Conda 環境內用 pip 安裝 CUDA 12.8 版 PyTorch，切換到本專案環境。

```powershell
conda env create -f environment.yml
conda activate aicup-esg
conda activate aicup-esg
```


##離開目前環境時執行：

```powershell
conda deactivate
```

## 產生繳交檔

確認 `data/` 內已有訓練資料與競賽測試集後，直接執行以下指令。程式會優先讀取 `data/vpesg4k_test_2000.json`，並產生 `outputs/submission.csv` 作為測試集預測結果。

```powershell
conda activate aicup-esg
python baseline_reference.py
```

執行時會自動判斷權重檔是否存在：

- 有 `models/baseline_reference.pt` 時，直接讀取模型並產生繳交檔。
- 沒有 `models/baseline_reference.pt` 時，先用 `data/vpesg4k_train_1000.json` 訓練，再產生繳交檔。

`baseline_reference.py` 是官方 baseline notebook 對應的 Python 版，使用 `bert-base-chinese`，權重預設儲存到 `models/baseline_reference.pt`。

## 執行改良版

```powershell
conda activate aicup-esg
python ours.py
```

改良版位於 `ours.py`，預設會使用 `hfl/chinese-roberta-wwm-ext`、類別權重、分層切分、early stopping、weight decay、warmup ratio，以及較強的多層分類頭。權重預設儲存到 `models/ours.pt`。

執行邏輯同樣是：

- 有 `models/ours.pt` 時，直接讀取模型並產生繳交檔。
- 沒有 `models/ours.pt` 時，先用 `data/vpesg4k_train_1000.json` 訓練，再產生繳交檔。

程式預設會自動選擇可用裝置；有 CUDA GPU 時會使用 GPU，否則會退回 CPU。若使用 RTX 4090，可視 GPU 記憶體把 `--batch-size` 從 8 調到 16、32 或更高；若發生記憶體不足，再往下調整。

## 主要參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--target` | `data/vpesg4k_test_2000.json` | 要預測的 CSV 或 JSON 資料；預設為競賽測試集，驗證集僅作備援 |
| `--output` | `outputs/submission.csv` | 繳交檔輸出位置 |
| `--train-data` | `data/vpesg4k_train_1000.json` | 訓練資料位置 |
| `--model-path` | `models/baseline_reference.pt` 或 `models/ours.pt` | 模型權重儲存或讀取位置 |
| `--model-name` | `bert-base-chinese` | Hugging Face 預訓練模型名稱；改良版預設為 `hfl/chinese-roberta-wwm-ext` |
| `--max-len` | `256` | Tokenizer 最大長度 |
| `--batch-size` | `8` | 批次大小 |
| `--epochs` | `10` | 訓練回合數 |
| `--learning-rate` | `2e-5` | 學習率 |
| `--weight-decay` | `0.01` | AdamW 權重衰減 |
| `--warmup-ratio` | `0.1` | 線性 warmup 比例 |
| `--validation-size` | `0.2` | 驗證集比例 |
| `--dropout-rate` | `0.2` | 分類頭 dropout |
| `--head-hidden-size` | `256` | 分類頭隱藏層大小；設為 `0` 可退回單層分類頭 |
| `--early-stopping-patience` | `3` | 驗證分數未改善幾個 epoch 後停止 |
| `--min-delta` | `0.0` | 視為改善所需的最低分數差 |
| `--pooling` | `cls` | 可選 `cls`、`mean` |
| `--no-class-weights` | 關閉 | 關閉類別權重 |
| `--device` | `auto` | 可選 `auto`、`cpu`、`cuda` |

## 繳交格式注意事項

`outputs/submission.csv` 會包含表頭，欄位順序如下：

```text
id,data,esg_type,promise_status,promise_string,verification_timeline,evidence_status,evidence_string,evidence_quality,company,ticker,page_number,pdf_url,company_source
```

程式輸出的 CSV 使用 UTF-8 無 BOM 編碼與 Unix 換行字元，符合競賽規則中對繳交檔的基本要求。

## 測試

```powershell
conda activate aicup-esg
python -m unittest discover -s tests
```

測試會確認輸出欄位順序、UTF-8 無 BOM 編碼，以及換行格式。
