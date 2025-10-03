# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import lightkurve as lk
from lightkurve.periodogram import BoxLeastSquares
from functools import lru_cache

# --- App Initialization ---
app = FastAPI(
    title="AstroVeritas API",
    description="An API for predicting exoplanet candidates from TESS lightcurve data.",
    version="1.0.0",
)

# --- Load Models and Data (Runs once on startup) ---
@lru_cache()
def get_model_and_data():
    try:
        clf_loaded = joblib.load("vetting_lightgbm.pkl")
        df_truth = pd.read_csv("TOI_2025.02.03_06.18.31.csv", skiprows=69)
        df_truth.columns = df_truth.columns.str.strip()
        label_map = {"CP": 1, "PC": 1, "KP": 1, "FP": 0}
        df_truth['label'] = df_truth['tfopwg_disp'].map(label_map)
        return clf_loaded, df_truth
    except Exception as e:
        raise RuntimeError(f"Could not load model or data files: {e}")

clf_loaded, df_truth = get_model_and_data()

# --- Helper Functions ---
def extract_features(time: np.ndarray, flux: np.ndarray):
    """Extracts the 10 features our model expects."""
    try:
        bls = BoxLeastSquares(time, flux)
        results = bls.autopower(0.05)
        idx = np.argmax(results.power)
        # Extract features
        p = results.period[idx]
        depth = results.depth[idx]
        dur = results.duration[idx]
        power = results.power[idx]
        snr = depth / (np.std(flux) + 1e-9)
        mean, std = np.mean(flux), np.std(flux)
        rng = np.max(flux) - np.min(flux)
        skew = np.mean((flux - mean)**3) / (std**3)
        kurt = np.mean((flux - mean)**4) / (std**4)
        return [p, depth, dur, power, snr, mean, std, rng, skew, kurt], p
    except Exception:
        return None, None

# --- API Data Models ---
class PredictionResponse(BaseModel):
    tic_id: int
    planet_probability: float
    ground_truth_label: str
    period: float
    plot_data: dict

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the AstroVeritas API"}

# --- THE CACHING MAGIC ---
# @lru_cache(maxsize=128) tells Python to save the results of this function.
# The next time it's called with the same tic_id, it will return the saved
# result instantly instead of running the code again.
@lru_cache(maxsize=128)
def process_tic_id(tic_id: int):
    """
    This function contains the slow logic and its results will be cached.
    """
    try:
        # 1. Fetch lightcurve data
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", cadence="short")
        lc = search.download().remove_nans().normalize()

        # 2. Extract features
        features, period = extract_features(lc.time.value, lc.flux.value)
        if features is None:
            raise ValueError("Feature extraction failed.")

        # 3. Run prediction
        features_array = np.array([features])
        probability = clf_loaded.predict_proba(features_array)[:, 1][0]

        # 4. Get ground truth label
        label_row = df_truth[df_truth['tid'] == tic_id]
        ground_truth = "Unknown"
        if not label_row.empty:
            ground_truth = "Planet" if label_row['label'].values[0] == 1 else "Not a Planet"

        # 5. Prepare plot data
        folded_lc = lc.fold(period)
        plot_data = {
            "time": folded_lc.time.value.tolist(),
            "flux": folded_lc.flux.value.tolist()
        }

        return {
            "tic_id": tic_id,
            "planet_probability": probability,
            "ground_truth_label": ground_truth,
            "period": period,
            "plot_data": plot_data,
        }
    except Exception as e:
        # We return the error so the main endpoint can handle it
        return {"error": str(e), "tic_id": tic_id}


@app.get("/predict/{tic_id}", response_model=PredictionResponse)
def get_prediction(tic_id: int):
    """
    This is the main API endpoint. It calls the cached function.
    """
    result = process_tic_id(tic_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=f"Could not process TIC ID {result['tic_id']}: {result['error']}")
    return result
