from pathlib import Path

import joblib
import pandas as pd

# LOAD MODEL & ENCODER

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "purchase_model.pkl"
)

month_encoder = joblib.load(
    BASE_DIR / "models" / "month_encoder.pkl"
)

visitor_encoder = joblib.load(
    BASE_DIR / "models" / "visitor_encoder.pkl"
)


# PREDICTION FUNCTION

def predict_purchase(
    administrative,
    administrative_duration,
    informational,
    informational_duration,
    product_related,
    product_related_duration,
    bounce_rates,
    exit_rates,
    page_values,
    special_day,
    month,
    operating_systems,
    browser,
    region,
    traffic_type,
    visitor_type,
    weekend
):

    month_encoded = month_encoder.transform(
        [month]
    )[0]

    visitor_encoded = visitor_encoder.transform(
        [visitor_type]
    )[0]

    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month_encoded,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_encoded,
        "Weekend": int(weekend)
    }])

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    return prediction, probability


# TEST PREDICTION


if __name__ == "__main__":

    prediction, probability = predict_purchase(
        administrative=2,
        administrative_duration=50,
        informational=1,
        informational_duration=20,
        product_related=40,
        product_related_duration=800,
        bounce_rates=0.01,
        exit_rates=0.03,
        page_values=20,
        special_day=0,
        month="May",
        operating_systems=2,
        browser=2,
        region=1,
        traffic_type=3,
        visitor_type="Returning_Visitor",
        weekend=True
    )

    print("=" * 50)
    print("PURCHASE PREDICTION TEST")
    print("=" * 50)

    print(f"Prediction  : {prediction}")
    print(f"Probability : {probability:.2%}")

    if prediction == 1:
        print("Result      : CUSTOMER WILL PURCHASE")
    else:
        print("Result      : CUSTOMER WILL NOT PURCHASE")