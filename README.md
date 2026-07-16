# 🏠 Pune Rental Price Predictor

A machine learning project that predicts monthly rental prices for apartments in Pune, built end-to-end — from raw, messy real estate data to a deployed interactive prediction app.

**[Live Demo](#)** 

![App Screenshot] (<img width="691" height="835" alt="image" src="https://github.com/user-attachments/assets/355dd23f-958c-4769-9096-9461dfc18b36" />
) 

---

## What This Project Does

Enter an apartment's locality, BHK, property type, size, and amenities — get an estimated fair monthly rent, powered by a Random Forest model trained on real Pune rental listings.

---

## Dataset

- **Source:** Real-world Indian housing listings dataset (Delhi, Mumbai, Pune), filtered to Pune only.
- **Size:** 5,316 raw listings → 3,873 after cleaning.
- **Features used:** house type, house size, location, latitude/longitude, number of bathrooms, number of balconies, BHK count.
- **Target:** monthly rental price (INR).

### Data Quality Issues Found and Handled
Real-world data is messy — here's what I found and how I dealt with it:

| Issue | Finding | Action |
|---|---|---|
| Empty columns | `priceSqFt`, `isNegotiable` were 100%/86% missing | Dropped |
| Malformed size field | `house_size` stored as text (`"906 sq ft"`), with comma-thousands-separators in larger values | Parsed to numeric |
| Missing values | `numBalconies` ~53% missing | Filled with 0 (assumed "not mentioned") |
| High-cardinality location | 224 unique localities | Grouped to top 20 + "Other" bucket |
| Geographic outlier | One listing's coordinates pointed ~1,000km outside Pune (likely a data entry error) | Filtered to Pune's actual lat/long bounding box |
| Size outliers | 43 listings (~1.1%) above 3,000 sq ft, including one 6,000 sq ft/₹400,000 listing skewing the fit | Removed — improved R² from 0.18 to 0.34 on its own |

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
Property size and geographic coordinates dominate the prediction — individual locality dummy variables and BHK/property-type contribute comparatively little once size and location are accounted for.

### Honest Limitations
An R² of ~0.41 means the model explains under half of price variation. This isn't a bug — it reflects real gaps in the dataset: no information on interior condition, exact building/society reputation, floor number, or furnishing status, all of which meaningfully affect real rent in practice. A more complete dataset, not a different model, would be the biggest lever for improving accuracy further. (Gradient Boosting and a manually engineered `size_per_bhk` feature were both tested and did not meaningfully outperform Random Forest.)

---

## Tech Stack

- **Python** — pandas, scikit-learn, joblib
- **Streamlit** — interactive prediction interface
- **Jupyter / Anaconda** — data cleaning and model development

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/pune-rental-predictor.git
cd pune-rental-predictor
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn streamlit joblib
```

**3. Run the app**
```bash
streamlit run predictor.py
```

---

## Project Structure
```
├── pune.ipynb          # Data cleaning, model training, evaluation
├── predictor.py         # Streamlit prediction app
├── rent_model.joblib     # Trained Random Forest model
├── cleaned_dataset.csv   # Cleaned, encoded dataset
└── README.md
```

---

## What I Learned

- Cleaning real-world data is most of the work — parsing malformed fields, handling missing values, and catching outliers mattered more to model performance than model choice did.
- Outlier removal alone nearly doubled R² (0.18 → 0.34); switching models afterward only added a few more points.
- Feature importance analysis is as valuable as the prediction itself — it explains *why* a model performs the way it does, not just *how well*.
- Debugging a live app taught me that a silently mismatched dictionary key (`num_Balconies` vs `numBalconies`) doesn't crash — it just quietly feeds wrong data into a model that keeps producing confident-looking, wrong answers. Always verify the exact keys/columns a model expects rather than assuming.
- Jupyter cells are stateful — re-running a cell that mutates data (like `.drop()` or `pd.get_dummies()`) without restarting the kernel can silently break things in ways that look like new bugs but are just stale state.

---

## Roadmap

- [x] Data cleaning and outlier handling
- [x] Model comparison (Linear Regression, Random Forest, Gradient Boosting)
- [x] Feature importance analysis
- [x] Streamlit prediction interface
- [ ] Per-locality coordinate lookup (currently relies on manual lat/long input)
- [ ] Expand location grouping beyond top 20 to reduce "Other" bucket dilution
- [ ] Deploy to Streamlit Community Cloud
