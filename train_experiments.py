
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------
# 1. Load datasets
# ------------------------------------------------

X_train = pd.read_csv("har_data/X_train.csv")
X_test = pd.read_csv("har_data/X_test.csv")

y_train = pd.read_csv("har_data/y_train.csv").squeeze("columns")
y_test = pd.read_csv("har_data/y_test.csv").squeeze("columns")

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# ------------------------------------------------
# 2. Define models
# ------------------------------------------------

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Logistic Regression": LogisticRegression(
    max_iter=1000),


    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ------------------------------------------------
# 3. Configure MLflow
# ------------------------------------------------


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HAR_Model_Experiments")

results = []

best_accuracy = 0
best_model = None
best_model_name = None

# ------------------------------------------------
# 4. Train and evaluate models
# ------------------------------------------------

for model_name, classifier in models.items():

    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, average="macro"
        )

        recall = recall_score(
            y_test, predictions, average="macro"
        )

        f1 = f1_score(
            y_test, predictions, average="macro"
        )

        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("scaling", "StandardScaler")

        if model_name == "KNN":
            mlflow.log_param("n_neighbors", 5)

        elif model_name == "Logistic Regression":
            mlflow.log_param("max_iter", 1000)

        elif model_name == "Random Forest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # Log trained model
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print("\nModel:", model_name)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

        # Identify best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_model_name = model_name

# ------------------------------------------------
# 5. Save comparison results
# ------------------------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison)

# ------------------------------------------------
# 6. Save the best model
# ------------------------------------------------

joblib.dump(best_model, "har_model.joblib")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
print("Best model saved successfully!")
