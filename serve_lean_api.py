
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np

# ---- Load model artifact ----
OBJ = joblib.load("models/multi_muscle_rf_lean.joblib")
MODEL     = OBJ["model"]
FEAT_COLS = OBJ["feat_cols"]
MUSCLES   = OBJ["muscles"]
DEFAULT_BANDS = {m: (40, 60) for m in MUSCLES}

app = FastAPI(title="EMG Lean Predictor", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class WindowPayload(BaseModel):
    fs: int = Field(1000, description="Sampling rate Hz")
    data: Dict[str, List[float]] = Field(..., description="{'VM':[...],'RF':[...],'BF':[...],'ST':[...]}")
    bands: Optional[Dict[str, List[float]]] = None

def feats_lean(a: np.ndarray):
    a = np.asarray(a, float)
    if a.size < 5 or not np.isfinite(a).any():
        return {"mean":0.0,"rms":0.0,"var":0.0,"wl":0.0,"iEMG":0.0}
    return {
        "mean": float(np.nanmean(a)),
        "rms":  float(np.sqrt(np.nanmean(a**2))),
        "var":  float(np.nanvar(a)),
        "wl":   float(np.nansum(np.abs(np.diff(a)))),
        "iEMG": float(np.trapz(np.abs(a))),
    }

def build_feature_row(payload: WindowPayload):
    feat = {}
    for m in MUSCLES:
        arr = np.asarray(payload.data.get(m, []), float)
        f = feats_lean(arr)
        for k, v in f.items():
            feat[f"{m}_{k}"] = v
    x = np.array([feat.get(c, 0.0) for c in FEAT_COLS]).reshape(1, -1)
    return x

def flag_val(v: float, band: tuple[float,float], tol: float = 5):
    lo, hi = band
    if v < lo - tol: return "under"
    if v > hi + tol: return "over"
    return "optimal"

@app.get("/health")
def health():
    return {"ok": True, "muscles": MUSCLES, "features": len(FEAT_COLS)}

@app.post("/predict_window")
def predict_window(payload: WindowPayload):
    x = build_feature_row(payload)
    yhat = MODEL.predict(x)[0]
    preds = {m: float(yhat[i]) for i, m in enumerate(MUSCLES)}

    bands = DEFAULT_BANDS.copy()
    if payload.bands:
        for m, b in payload.bands.items():
            if m in bands and isinstance(b, (list, tuple)) and len(b) == 2:
                bands[m] = (float(b[0]), float(b[1]))

    flags = {m: flag_val(preds[m], bands[m]) for m in MUSCLES}
    return {"pred": preds, "flags": flags, "bands_used": bands}
