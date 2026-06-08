# AI CUP VeriPromiseESG 2026 Baseline

本專案整理 AI CUP「ESG 永續承諾驗證競賽 2026」的 baseline 程式碼與繳交檔產生流程。主要程式碼放在倉庫根目錄，競賽相關說明集中在 `docs/`，方便評審或隊友直接查看與執行。

## 專案結構

```text
AICUP/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ predict_submission.py
├─ data/
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


## 環境安裝

建議使用 Conda 建立獨立環境。PyTorch 官方目前主要提供 pip wheel 安裝指令；因此本專案使用 Conda 管理 Python 與資料科學套件，並在 Conda 環境內用 pip 安裝 CUDA 12.8 版 PyTorch。

```powershell
cd C:\Users\Ted\Desktop\AICUP
conda env create -f environment.yml
conda activate aicup-esg
```

如果已經建立過同名環境，請用以下指令更新：

```powershell
conda env update -f environment.yml --prune
conda activate aicup-esg
```

## 切換環境

每次開新的 PowerShell 或終端機後，請先切換到本專案環境：

```powershell
conda activate aicup-esg
```

切換成功時，命令提示字元前方通常會出現 `(aicup-esg)`。若想確認目前使用哪個 Conda 環境，可執行：

```powershell
conda info --envs
```

清單中有 `*` 的那一列就是目前環境。要離開目前環境時執行：

```powershell
conda deactivate
```

若 PowerShell 顯示無法使用 `conda activate`，請先執行 `conda init powershell`，關掉並重新開啟 PowerShell 後再試一次。

`requirements.txt` 保留作為 pip 備用安裝檔；正式執行建議以 `environment.yml` 為準。

## 產生繳交檔

確認 `data/` 內已有驗證資料後，直接執行以下指令，會優先讀取 `data/vpesg4k_val_1000.json` 並產生 `outputs/submission.csv`。

```powershell
conda activate aicup-esg
python predict_submission.py
```

未指定模型模式時，程式會沿用輸入資料中已存在的標籤欄位；若欄位缺值或內容不在允許標籤內，會填入保守預設值。這個模式主要用來快速確認輸出格式。

## 重新訓練 Baseline

```powershell
conda activate aicup-esg
python predict_submission.py --train
```

預設會自動下載官方公開訓練資料到 `data/vpesg4k_train_1000.json`，使用 `bert-base-chinese` 訓練多任務分類模型，並把權重儲存到 `models/best_model.pt`。

常用參數：

```powershell
conda activate aicup-esg
python predict_submission.py --train --epochs 3 --batch-size 16
```

程式預設會自動選擇可用裝置；有 CUDA GPU 時會使用 GPU，否則會退回 CPU。若使用 RTX 4090，可視 GPU 記憶體把 `--batch-size` 從 16 調到 32 或更高；若發生記憶體不足，再往下調整。

## 使用既有權重預測

已經有 `models/best_model.pt` 時，可直接讀取權重產生繳交檔。

```powershell
conda activate aicup-esg
python predict_submission.py --predict-with-model
```

也可以指定資料、輸出位置與模型權重：

```powershell
conda activate aicup-esg
python predict_submission.py --predict-with-model --target data/vpesg4k_val_1000.csv --output outputs/submission.csv --model-path models/best_model.pt
```

## 主要參數

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--target` | `data/vpesg4k_val_1000.json` | 要預測的 CSV 或 JSON 資料 |
| `--output` | `outputs/submission.csv` | 繳交檔輸出位置 |
| `--train-data` | `data/vpesg4k_train_1000.json` | 訓練資料位置 |
| `--model-path` | `models/best_model.pt` | 模型權重儲存或讀取位置 |
| `--model-name` | `bert-base-chinese` | Hugging Face 預訓練模型名稱 |
| `--max-len` | `256` | Tokenizer 最大長度 |
| `--batch-size` | `8` | 批次大小 |
| `--epochs` | `10` | 訓練回合數 |
| `--learning-rate` | `2e-5` | 學習率 |
| `--validation-size` | `0.2` | 驗證集比例 |
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
