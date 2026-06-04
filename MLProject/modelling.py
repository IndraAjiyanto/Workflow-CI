import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("diabetes-baseline")

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    X_train, X_test, y_train, y_test = load_data('diabetes_preprocessing.csv')
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("Autolog selesai! Artefak tersimpan lokal di folder mlruns/")

if __name__ == "__main__":
    main()
