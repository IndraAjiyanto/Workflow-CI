import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')

dagshub.init(
    repo_owner='IndraAjiyanto',
    repo_name='Eksperimen_SML_Indra-Ajiyanto',
    mlflow=True
)

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_confusion_matrix(y_test, y_pred, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Diabetes', 'Diabetes'],
                yticklabels=['Tidak Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Confusion matrix disimpan: {filename}")
    return filename

def save_feature_importance(model, feature_names, filename='feature_importance.png'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[indices], color='#4C9BE8')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Feature importance disimpan: {filename}")
    return filename

def main():
    X_train, X_test, y_train, y_test = load_data('diabetes_preprocessing.csv')
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

    with mlflow.start_run(run_name="RandomForest_Baseline"):

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        print(f"\n{'='*40}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"{'='*40}\n")

        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "random_forest_model")

        report = classification_report(y_test, y_pred,
                    target_names=['Tidak Diabetes', 'Diabetes'])
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        cm_path = save_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        fi_path = save_feature_importance(model, list(X_train.columns))
        mlflow.log_artifact(fi_path)

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("model.pkl")

        print("Semua log berhasil dikirim ke DagsHub!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
