# Airline Arrival Delay Prediction

## Project objective

Predict **Arrival Delay in Minutes**, a continuous numeric outcome, using the
airline survey dataset. This is a **regression** problem—not a satisfaction
classification problem. The project demonstrates an end-to-end, advanced
baseline: data validation, leakage-aware feature selection, preprocessing,
cross-validated model tuning, held-out evaluation, and business-oriented error
analysis.

## Important limitation and prediction point

The available data is a post-flight passenger survey. Many columns (service
ratings and `satisfaction`) are only known after the flight and may be
influenced by the delay itself. Using them would leak information unavailable
when an operational delay estimate is needed. They are therefore deliberately
excluded.

The default model predicts arrival delay **after departure**, when
`Departure Delay in Minutes` is known. To compare with a prediction made
**before departure**, pass `--prediction-stage pre-departure`; that version
excludes departure delay. The pre-departure version may be substantially less
accurate because the dataset has no route, schedule, weather, aircraft, or
connecting-flight information.

Even the after-departure model is an educational prototype. Passenger age,
gender, and loyalty status are not operational causes of flight delays, may
raise fairness concerns, and should not be used for real-world decisions
without a compelling, reviewed justification. A deployable operational model
should use flight/operations data instead.

## Target and features

**Target:** `Arrival Delay in Minutes` (minutes, continuous and non-negative).

**Candidate predictors**

| Feature | Why it is included |
| --- | --- |
| `Departure Delay in Minutes` | Strong operational predictor when estimating after departure; excluded in the pre-departure scenario. |
| `Flight Distance` | Available trip characteristic that can inform expected flight duration and recovery opportunity. |
| `Age`, `Gender`, `Customer Type`, `Type of Travel`, `Class` | Present in this dataset and available before the flight, but weak/non-causal and potentially sensitive; included only to demonstrate evaluation on the provided columns. Consider dropping them for a production system. |

**Excluded:** all service-quality ratings and `satisfaction`. These are
post-flight survey answers, not reliable pre-outcome operational features.
`satisfaction` is also a separate classification target, not an arrival-delay
feature. IDs are absent from the provided columns.

## Approach

1. Read `data.csv`; validate required columns; coerce numeric fields and drop
   rows with missing target values.
2. Make a reproducible 80/20 random train/test split. The test partition is
   held out from tuning.
3. Impute numeric missing values with the training median and categorical
   missing values with the training most-frequent value. One-hot encode
   categorical features, ignoring unseen categories at inference.
4. Compare a training-set median predictor with a tuned
   `HistGradientBoostingRegressor`. Randomized search uses 3-fold
   cross-validation and mean absolute error (MAE) for model selection.
5. Report test MAE, RMSE, R², median absolute error, and the share of
   predictions within 5 and 15 minutes. Save prediction-level errors,
   delay-band metrics, and permutation importance for interpretation.

Tree boosting can learn nonlinear relationships and feature interactions
without scaling. Arrival delays are typically right-skewed and can have large
outliers; MAE is the primary tuning metric because it is less dominated by
extreme cases than RMSE. RMSE is reported as an additional measure of large
errors. The model is not constrained to non-negative output, so predictions
are clipped at zero before final evaluation and saving.

## Results on the supplied CSV

The file contains 129,880 rows; 129,487 rows had a valid non-negative target
and were used. The target is highly skewed: 56.2% of these flights have zero
arrival delay, the 95th percentile is 78 minutes, the 99th percentile is 182
minutes, and the maximum is 1,584 minutes. This is why both MAE and RMSE are
reported.

On the fixed 20% test split, the default **after-departure** model achieved
MAE **6.06 minutes**, RMSE **17.19 minutes**, R² **0.828**, and **91.5%**
within 15 minutes. The training-median baseline had MAE 15.60 minutes, so the
model reduced absolute error by about 61% on this split. Departure delay
dominates held-out permutation importance, as expected when it is available
at prediction time.

The **pre-departure** variant had test MAE **20.74 minutes**, worse than the
15.60-minute median baseline, with R² approximately 0. This is evidence that
the remaining columns do not provide useful operational delay forecasts in
this dataset; do not deploy that variant.

Performance also degrades for severe disruptions: on delays greater than 60
minutes, test MAE was 22.64 minutes and average signed error was -9.30
minutes, indicating underprediction in that band. Such cases need separate
monitoring and possibly operational escalation rules. These are one random
split's results, not a guarantee of future performance.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
```

## Streamlit application and API

The app is split into two processes:

- `frontend/app.py` — Streamlit UI for entering one flight record
- `backend/main.py` — FastAPI service that validates input and calls the saved
  scikit-learn pipeline

From the project root, install dependencies and ensure the default trained
model is present:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
```

Open **two terminals** in the project root. Start the backend in terminal 1:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then start the frontend in terminal 2:

```bash
.venv/bin/python -m streamlit run frontend/app.py
```

Streamlit prints the local URL (usually `http://localhost:8501`). The backend
health check and interactive API docs are available at
`http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`. Keep both
processes running while using the app. To use a different backend address,
set `AIRLINE_API_URL` before launching Streamlit.

### Run the API with Docker

Build and run the backend API from the project root. The image uses the
after-departure model and metadata already present in `artifacts/`.

```bash
docker compose up --build
```

The API health check and documentation are available at
`http://localhost:8000/health` and `http://localhost:8000/docs`. The Docker
image runs only the API; start the Streamlit frontend separately and set
`AIRLINE_API_URL=http://localhost:8000` so it can reach the container. Stop the
service with `docker compose down`.

If you previously trained the pre-departure variant, rerun the default
`.venv/bin/python train.py` command before starting the API. The API
intentionally checks that the model artifact matches its after-departure input
schema.

The API accepts one flight at `POST /predict`. Its JSON fields are `age`,
`flight_distance`, `departure_delay_minutes`, `gender`, `customer_type`,
`type_of_travel`, and `flight_class`. Input types and ranges are validated by
the API; invalid values return an HTTP 422 response with details.

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"flight_distance":800,"departure_delay_minutes":20,"gender":"Female","customer_type":"Loyal Customer","type_of_travel":"Business travel","flight_class":"Business"}'
```

Compare the pre-departure feature set:

```bash
.venv/bin/python train.py --prediction-stage pre-departure
```

Use a different input file or output folder:

```bash
.venv/bin/python train.py --data path/to/airline.csv --output-dir artifacts
```

## Outputs

The script creates the output directory (default `artifacts/`) and writes:

- `arrival_delay_model.joblib` — fitted preprocessing and regression pipeline
- `metrics.json` — test and cross-validation metrics, row counts, and best
  hyperparameters
- `test_predictions.csv` — actual delay, predicted delay, and signed error
- `permutation_importance.csv` — held-out permutation importances in MAE units
- `delay_band_metrics.csv` — test performance for on-time, short, moderate,
  and long delays

To use the fitted model, pass a pandas DataFrame with the feature columns for
the selected prediction stage to the loaded pipeline's `.predict(...)` method.
For `after-departure`, include `Departure Delay in Minutes`.

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("artifacts/arrival_delay_model.joblib")
flight = pd.DataFrame([{
    "Age": 35,
    "Flight Distance": 800,
    "Departure Delay in Minutes": 20,
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Type of Travel": "Business travel",
    "Class": "Business",
}])
arrival_delay_minutes = np.maximum(0, model.predict(flight))[0]
```

## Reading the results for the business

- **MAE** is the typical absolute miss in minutes. For example, an MAE of 10
  means predictions are off by 10 minutes on average in absolute terms.
- **Within 15 minutes** is an easy-to-communicate operational tolerance rate;
  the acceptable tolerance should be chosen by the airline, not assumed from
  the data.
- **Delay-band metrics** show whether performance degrades on long delays,
  where operational disruption is often greatest.
- **Signed error** is prediction minus actual. Positive values indicate
  overprediction; negative values indicate underprediction. Persistent
  underprediction can lead to missed connection/reaccommodation opportunities.
- **Permutation importance** estimates how much held-out MAE worsens when one
  input column is shuffled. It is predictive association, not causality, and
  correlated features can share or obscure importance.

Random train/test splitting measures performance on similar rows from this
dataset. It does not prove performance on future seasons, unseen routes,
airlines, or changed operations. A real deployment should use a time-based
holdout and operational predictors, check subgroup errors, monitor drift, and
compare model impact against a simple operational baseline.

## Project files

- `train.py` — complete training, evaluation, and artifact-generation script
- `requirements.txt` — runtime dependencies
- `data.csv` — supplied dataset (not distributed by this project)

Backend:
.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Front end:
.venv/bin/python -m streamlit run frontend/app.py