# AI CUP VeriPromiseESG 繳交包

## 環境安裝

```powershell
cd C:\Users\Ted\Desktop\AICUP\submission_package
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 直接產生目前資料的 CSV

```powershell
python predict_submission.py
```

輸出位置：

```text
outputs/submission.csv
```

## 重新訓練 baseline 後產生 CSV

```powershell
python predict_submission.py --train
```

訓練資料若不存在，程式會自動下載官方公開訓練資料到 `data/vpesg4k_train_1000.json`。

## 使用已訓練權重預測

```powershell
python predict_submission.py --predict-with-model
```

預設權重位置為 `models/best_model.pt`。
