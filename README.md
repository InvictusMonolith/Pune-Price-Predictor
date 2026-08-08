# 🏠 Pune Rental Price Predictor

A machine learning project that predicts monthly rental prices for apartments in Pune — built end-to-end from raw, messy real estate data to a containerized, publicly deployed prediction API with a live interface.

**[Live API](#)** *(https://puneprice-predictor.onrender.com)*


**[Live Demo](#)** *(https://pune-price-predictor.streamlit.app/)*

![App Screenshot](#) *(<img width="867" height="876" alt="image" src="https://github.com/user-attachments/assets/ac5c3710-91d0-495e-8937-ab3d8c34b3bd" />
)*

---

## What This Project Does

Enter a locality, BHK, property type, size, and amenities — get an estimated fair monthly rent, powered by a Random Forest model trained on real Pune rental listings and served through a containerized REST API.

---

## Dataset

- **Source:** Real-world Indian housing listings dataset (Delhi, Mumbai, Pune), filtered to Pune only.
- **Size:** 5,316 raw listings → 3,873 after cleaning.
- **Features used:** house type, house size, location, latitude/longitude, number of bathrooms, number of balconies, BHK count.
- **Target:** monthly rental price (INR).

### Data Quality Issues Found and Handled
Real-world data is messy — here's what was found and how it was handled:

| Issue | Finding | Action |
|---|---|---|
| Empty columns | `priceSqFt`, `isNegotiable` were 100%/86% missing | Dropped |
| Malformed size field | `house_size` stored as text (`"906 sq ft"`), with comma-thousands-separators in larger values | Parsed to numeric |
| Missing values | `numBalconies` ~53% missing | Filled with 0 (assumed "not mentioned") |
| High-cardinality location | 224 unique localities | Grouped to top 20 + "Other" bucket (tested top 40 — regressed accuracy, reverted) |
| Geographic outlier | One listing's coordinates pointed ~1,000km outside Pune | Filtered to Pune's actual lat/long bounding box |
| Size outliers | 43 listings (~1.1%) above 3,000 sq ft | Removed — improved R² from 0.18 to 0.34 on its own |

---

## Modeling

Three models were trained and compared on an 80/20 train/test split:

| Model | RMSE (₹) | R² |
|---|---|---|
| Linear Regression | 16,115.94 | 0.3374 |
| Gradient Boosting | 15,311.49 | 0.4019 |
| **Random Forest (final)** | **15,223.03** | **0.4088** |

### Feature Importance (Random Forest)
```
house_size        54.6%
latitude          14.4%
longitude          9.8%
numBathrooms        4.6%
bhk                 2.4%
```
Property size and geographic coordinates dominate the prediction — individual locality dummy variables and BHK/property-type contribute comparatively little once size and location are accounted for. Expanding location grouping from top-20 to top-40 localities was tested and made accuracy *worse* (0.3987 vs 0.4088) — more categories diluted signal per locality without adding real predictive value. Reverted to top-20.

### Honest Limitations
An R² of ~0.41 means the model explains under half of price variation. This reflects real gaps in the dataset — no information on interior condition, exact building/society reputation, floor number, or furnishing status, all of which meaningfully affect real rent. A more complete dataset, not a different model, would be the biggest lever for further improvement.

---

## Architecture

```
Streamlit UI  →  HTTP POST  →  FastAPI /predict (Dockerized, deployed on Render)  →  Model  →  JSON response
```

- **Per-locality coordinate lookup:** rather than requiring manual lat/long entry, the app auto-fills coordinates based on the selected locality, computed from the training data's per-locality averages, with a Pune-wide fallback for localities outside the top-20 list.
- **Containerized:** packaged with a `Dockerfile` so the API runs identically regardless of host environment.
- **Deployed live on Render.**

---

## Tech Stack

- **Python** — pandas, scikit-learn, joblib
- **FastAPI** — prediction API
- **Docker** — containerized deployment
- **Render** — live hosting
- **Streamlit** — client interface
- **Jupyter / Anaconda** — data cleaning and model development

---

## How to Run

### Option 1 — Docker (recommended, matches production)
```bash
git clone https://github.com/InvictusMonolith/Pune-Price-Predictor.git
cd Pune-Price-Predictor
docker build -t pune-rent-api .
docker run -p 8000:8000 pune-rent-api
```
API docs available at `http://127.0.0.1:8000/docs`

### Option 2 — Local Python environment
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Running the Streamlit client
```bash
streamlit run app.py
```

---

## Project Structure
```
├── pune.ipynb              # Data cleaning, model training, evaluation
├── main.py                  # FastAPI prediction endpoint
├── app.py                   # Streamlit client interface
├── Dockerfile                # Container definition
├── docker-compose.yml
├── requirements.txt
├── rent_model.joblib          # Trained Random Forest model
├── Indian_housing_Pune_data.csv
└── README.md
```

---

## What I Learned

- Cleaning real-world data is most of the work — parsing malformed fields, handling missing values, and catching outliers mattered more to model performance than model choice did. Outlier removal alone nearly doubled R² (0.18 → 0.34); switching models afterward only added a few more points.
- An unexpectedly large accuracy jump is a signal to check for data leakage, not celebrate — a bug elsewhere in this project (accidentally leaving a target-derived column in the feature set) once produced a fake near-doubling of R² before being caught and corrected.
- Writing a fix and actually wiring it into the code path that runs are two different steps — a coordinate lookup table sat unused in the app for a while before being connected to the UI it was meant to serve.
- Feature importance analysis is as valuable as the prediction itself — it explains *why* a model performs the way it does, not just how well.
- Jupyter cells are stateful — re-running a cell that mutates data can silently break things in ways that look like new bugs but are just stale state from a previous run.
- Containerizing exposes assumptions a local environment hides — getting the Dockerfile to correctly resolve the model file's path required being explicit about what gets copied into the image and where.

---

## Roadmap

- [x] Data cleaning and outlier handling
- [x] Model comparison (Linear Regression, Random Forest, Gradient Boosting)
- [x] Feature importance analysis
- [x] Per-locality coordinate lookup
- [x] Streamlit prediction interface
- [x] FastAPI prediction endpoint
- [x] Dockerized deployment
- [x] Live deployment on Render
