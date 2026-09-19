# Telco Customer Churn Prediction

## Overview
This project provides an end-to-end machine learning solution to predict customer churn for a telecommunications company. By identifying customers who are likely to cancel their service, the retention team can proactively engage them with targeted offers. 

The solution includes Data Preparation, Exploratory Data Analysis (EDA), Feature Engineering, a tuned Decision Tree classification model, and a deployable FastAPI REST endpoint.

## Project Structure

```text
NAGPDSASSIGNMENT/
│
├── data/
│   ├── TelcoCustomerChurn.csv                 # Raw dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv # Data dictionary
│
├── notebook/
│   └── churn_analysis.ipynb                   # Complete EDA, modeling, and evaluation workflow
│
├── model/
│   └── churn_model.pkl                        # Saved Scikit-Learn preprocessing & model pipeline
│
├── app.py                                     # FastAPI application for the prediction endpoint
├── requirements.txt                           # Project dependencies
├── sample_request.json                        # Example JSON payload for API testing
├── sample_request_low_risk.json               # Example JSON payload for API testing
├── .gitignore
└── README.md                                  # Setup and execution instructions
```

## Notebook contents

`notebook/churn_analysis.ipynb` runs top to bottom on a fresh kernel and covers:

| Section | Contents |
|---|---|
| 1 | Data understanding — dtypes, missing values, duplicates, numerical/categorical identification, target distribution, cleaning |
| 2 | Exploratory data analysis — 7 visualisations, each with a business insight |
| 3 | Feature engineering — 4 features plus one tested and rejected |
| 4 | Model development — 4 Decision Tree configurations compared |
| 5 | Evaluation — accuracy, precision, recall, F1, confusion matrix, precision-vs-recall discussion |
| 6 | Interpretation — feature importance, tree visualisation, decision rules |
| 7 | Model saving — pipeline pickled with metadata |

Categorical encoding is handled inside the pipeline (`OneHotEncoder`) rather than
applied to the dataframe up front, so the identical transformation is guaranteed
at both training and serving time.

---

## Setup

Requires Python 3.9 or later.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

### Generate the model

The pickled model is tied to the scikit-learn version that produced it, so
regenerate it locally before starting the API:

```bash
jupyter notebook
```

Open `notebook/churn_analysis.ipynb` and run **Kernel → Restart Kernel and Run
All Cells**. This writes `model/churn_model.pkl` and `model/model_metadata.json`.

### Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Pipeline overview

```
Raw CSV
  → clean TotalCharges (11 blanks, all tenure=0, imputed as 0)
  → drop customerID
  → engineer 4 features
  → 70/30 stratified split (random_state=42)
  → ColumnTransformer: numerical passthrough + one-hot categorical
  → DecisionTreeClassifier
  → joblib pickle (preprocessing + model in one object)
```

Preprocessing lives inside a scikit-learn `Pipeline`, so the encoder is fitted
only on training data and the identical transformation is applied at serving
time. This prevents data leakage and removes any chance of the API and the
notebook disagreeing about preprocessing.

### Engineered features

| Feature | Definition | Rationale |
|---|---|---|
| `Total_Services` | count of the 6 optional services subscribed | proxy for how embedded the customer is in the product |
| `Is_Auto_Payment` | 1 for bank transfer / credit card | automatic payers churn at 16.0% vs 34.7% manual |

All four are computed row-wise from a single customer's own values, using no
statistics pooled across the dataset. They are therefore safe to compute
before the train/test split and reproducible for a single unseen record.

---

## Results

Four Decision Tree configurations were compared on the held-out test set:

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Default (unpruned) | 0.716 | 0.469 | 0.505 | 0.486 |
| max_depth=5, balanced (final) | 0.721 | 0.484 | 0.761 | 0.592 |

Final model on the test set (2,113 customers): accuracy 0.721, precision 0.484,
recall 0.761, F1 0.592.

Confusion matrix:

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 1228 | 321 |
| **Actual Yes** | 278 | 283 |

The model catches 444 of 561 actual churners while flagging 876 customers in
total, reducing the retention team's contact list by 59% versus contacting
everyone.

### Class imbalance

The target is imbalanced at 73.5% / 26.5%. `class_weight='balanced'` reweights
the minority class inversely to its frequency during training, which lifted
recall from 0.319 to 0.781 at equivalent depth. No resampling was applied —
reweighting achieves the same effect without synthesising rows or discarding
data, and keeps the pipeline simple enough to serve directly.

### Why recall over precision

A false negative is a customer lost silently, costing their full remaining
lifetime value. A false positive is a retention offer sent to someone who was
staying, costing only the offer. Telecom churn is largely irreversible once the
customer ports out, so the asymmetry favours recall. The unpruned and
depth-5 models were rejected on this basis despite the latter having the
highest test accuracy.

---

## API

### `POST /predict`

Accepts the 19 raw customer fields. Engineered features are derived
server-side, so callers do not need to know about them.

**Request** (`sample_request.json`):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.40,
  "TotalCharges": 188.80
}
```

**Response** (200):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8697
}
```

A low-risk example is provided in `sample_request_low_risk.json` and returns
`"prediction": "No"` with probability 0.0605.

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info and model load status |
| GET | `/health` | 200 if the model is loaded, 503 otherwise |
| GET | `/docs` | Interactive Swagger UI |

### Calling the API

```bash
# curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```powershell
# PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -InFile sample_request.json
```

```python
# Python
import requests, json
payload = json.load(open("sample_request.json"))
print(requests.post("http://127.0.0.1:8000/predict", json=payload).json())
```

---

## Known limitations

- The dataset is a static snapshot with no timestamps, so the model cannot
  capture trends such as recent bill increases or support-ticket volume.
- Three of the two engineered features received near-zero importance because
  they restate information the tree can already reach through the raw columns.
  They are retained and documented rather than removed.
- No monitoring for data drift; a production deployment would need periodic
  retraining and performance tracking.
  
## Github repo link

https://github.com/115Shubham1992/DSChurnPrediction

## One drive demo video link

https://nagarro-my.sharepoint.com/my?id=/personal/shubham_vijay_nagarro_com/Documents/NAGPDSAssingment&viewid=d14422e4-1c69-4f65-8b3e-53730d53e468

