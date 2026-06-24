import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def load_data():
    """
    Load dataset
    """

    df = pd.read_csv(
        "data/online_shoppers_intention.csv"
    )

    return df


def preprocess_data(df):
    """
    Data cleaning and preprocessing
    """

    print("\nChecking Missing Values...")
    print(df.isnull().sum())

    df = df.copy()

    # Encode Month
    month_encoder = LabelEncoder()

    df["Month"] = month_encoder.fit_transform(
        df["Month"]
    )

    # Encode VisitorType
    visitor_encoder = LabelEncoder()

    df["VisitorType"] = visitor_encoder.fit_transform(
        df["VisitorType"]
    )

    # Convert boolean columns
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    return (
        df,
        month_encoder,
        visitor_encoder
    )


def prepare_data(df):
    """
    Split features and target
    """

    X = df.drop(
        columns=["Revenue"]
    )

    y = df["Revenue"]

    return X, y


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate classification model
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )


def main():

    print("=" * 60)
    print("CUSTOMER PURCHASE PREDICTION SYSTEM")
    print("=" * 60)

   

    # LOAD DATA

    df = load_data()

    print("\nDataset Loaded Successfully")
    print(f"Dataset Shape : {df.shape}")



    # PREPROCESSING
    (
        df,
        month_encoder,
        visitor_encoder
    ) = preprocess_data(df)

   

    # FEATURES & TARGET
    
    X, y = prepare_data(df)

    

    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

  

    # STANDARDIZATION

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # LOGISTIC REGRESSION

    print("\nTraining Logistic Regression...")

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(
        X_train_scaled,
        y_train
    )

    (
        lr_acc,
        lr_precision,
        lr_recall,
        lr_f1,
        _
    ) = evaluate_model(
        lr_model,
        X_test_scaled,
        y_test
    )

    print("\nLOGISTIC REGRESSION RESULT")
    print(f"Accuracy  : {lr_acc:.4f}")
    print(f"Precision : {lr_precision:.4f}")
    print(f"Recall    : {lr_recall:.4f}")
    print(f"F1 Score  : {lr_f1:.4f}")

    

    # RANDOM FOREST

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(
        X_train,
        y_train
    )

    (
        rf_acc,
        rf_precision,
        rf_recall,
        rf_f1,
        _
    ) = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    print("\nRANDOM FOREST RESULT")
    print(f"Accuracy  : {rf_acc:.4f}")
    print(f"Precision : {rf_precision:.4f}")
    print(f"Recall    : {rf_recall:.4f}")
    print(f"F1 Score  : {rf_f1:.4f}")

    # =====================
    # HYPERPARAMETER TUNING
    # =====================

    print("\nRunning Hyperparameter Tuning...")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    (
        tuned_acc,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        predictions
    ) = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print("\nBEST PARAMETERS")
    print(grid_search.best_params_)

    print("\nOPTIMIZED MODEL RESULT")
    print(f"Accuracy  : {tuned_acc:.4f}")
    print(f"Precision : {tuned_precision:.4f}")
    print(f"Recall    : {tuned_recall:.4f}")
    print(f"F1 Score  : {tuned_f1:.4f}")

   

    # CLASSIFICATION REPORT

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # FEATURE IMPORTANCE

    print("\nTOP 10 IMPORTANT FEATURES")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance_df.head(10)
    )



    # SAVE FILES

    joblib.dump(
        best_model,
        "models/purchase_model.pkl"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        month_encoder,
        "models/month_encoder.pkl"
    )

    joblib.dump(
        visitor_encoder,
        "models/visitor_encoder.pkl"
    )

    print("\nModel Saved Successfully")
    print("models/purchase_model.pkl")

    print("\nScaler Saved Successfully")
    print("models/scaler.pkl")

    print("\nMonth Encoder Saved Successfully")
    print("models/month_encoder.pkl")

    print("\nVisitor Encoder Saved Successfully")
    print("models/visitor_encoder.pkl")

    print("\nTRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()