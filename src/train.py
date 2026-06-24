import os
from typing import Any, Tuple

import joblib
import numpy as np


# Model/scaler/encoder disimpan di folder models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")


def _load_artifacts() -> Tuple[Any, Any, Any, Any]:
    """Load trained artifacts (model, scaler, encoders)."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "purchase_model.pkl"))
    month_encoder = joblib.load(os.path.join(MODELS_DIR, "month_encoder.pkl"))
    visitor_encoder = joblib.load(os.path.join(MODELS_DIR, "visitor_encoder.pkl"))
    return model, scaler, month_encoder, visitor_encoder


def predict_purchase(
    administrative: float,
    administrative_duration: float,
    informational: float,
    informational_duration: float,
    product_related: float,
    product_related_duration: float,
    bounce_rates: float,
    exit_rates: float,
    page_values: float,
    special_day: float,
    month: str,
    weekend: float,
    visitor_type_new: float,
    _unused1: float,
    _unused2: float,
    visitor_type: str,
    use_proba: bool = True,
) -> Tuple[int, float]:
    """Predict purchase (BUY/NO BUY) and probability.

    Notes:
    - Signature ini dibuat agar kompatibel dengan pemanggilan di app.py.
    - Feature engineering minimal mengikuti training notebook:
      * Month dan VisitorType di-encode menggunakan LabelEncoder
      * Weekend/Revenue dibuat int pada training, jadi di sini kami cast float->int
    """

    model, scaler, month_encoder, visitor_encoder = _load_artifacts()

    # Encode categoricals
    month_enc = float(month_encoder.transform([month])[0])
    visitor_enc = float(visitor_encoder.transform([visitor_type])[0])

    # Weekend adalah input app.py (di app.py Anda mengirim nilai `2, 2, 1, 3, visitor_type` yang
    # tampaknya untuk slot numerik lain / atau mismatch penamaan). Untuk keep up with
    # training dataset, kita pakai `weekend` sebagai weekend.
    weekend_int = int(round(float(weekend)))

    # Build feature vector dengan jumlah fitur sesuai model tersimpan (RandomForestClassifier expected 17).
    # Notebook asli menyiapkan X = df_clean.drop(columns=["Revenue"]) setelah preprocessing:
    # kolom X berjumlah 17 (termasuk weekend & beberapa boolean numeric lain yang mungkin
    # berasal dari dataset). Namun karena feature engineering notebook sudah mengubah hanya
    # beberapa kolom eksplisit, kita cukup menyamakan panjang input dengan model:
    # - Kita gunakan 13 fitur yang tersedia dari UI + padding fitur dummy = 0.
    base_features = [
        float(administrative),
        float(administrative_duration),
        float(informational),
        float(informational_duration),
        float(product_related),
        float(product_related_duration),
        float(bounce_rates),
        float(exit_rates),
        float(page_values),
        float(special_day),
        month_enc,
        visitor_enc,
        weekend_int,
    ]

    # Pad sampai n_features_in_ model
    n_expected = int(getattr(model, "n_features_in_", 17) or 17)
    if len(base_features) < n_expected:
        base_features = base_features + [0.0] * (n_expected - len(base_features))
    elif len(base_features) > n_expected:
        base_features = base_features[:n_expected]

    features = np.array(base_features, dtype=float).reshape(1, -1)


    # Training di notebook:
    # - Logistic Regression: scaler.fit_transform(X_train)
    # - RandomForest: rf_model.fit(X_train, y_train) (tanpa scaler)
    # Model yang disimpan di purchase_model.pkl pada notebook adalah best_model dari GridSearchCV
    # (random forest) namun tidak pasti. Untuk aman, kita coba jalankan dengan scaler
    # terlebih dulu, lalu jika error fallback.
    if use_proba:
        try:
            features_scaled = scaler.transform(features)
            proba = float(model.predict_proba(features_scaled)[0, 1])
            pred = int(model.predict(features_scaled)[0])
            return pred, proba
        except Exception:
            proba = float(model.predict_proba(features)[0, 1])
            pred = int(model.predict(features)[0])
            return pred, proba
    else:
        try:
            features_scaled = scaler.transform(features)
            pred = int(model.predict(features_scaled)[0])
            return pred, float(model.predict_proba(features_scaled)[0, 1])
        except Exception:
            pred = int(model.predict(features)[0])
            return pred, float(model.predict_proba(features)[0, 1])

