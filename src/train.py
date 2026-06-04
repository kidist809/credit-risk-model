import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

RANDOM_STATE = 42
mlflow.set_tracking_uri("sqlite:///C:/Users/kidim/credit-risk-model/mlflow.db")
mlflow.set_experiment("credit_risk_rfm")


def load_data(path='data/processed/model_data.csv'):
    df = pd.read_csv(path)
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    return X, y


def train_and_log():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == 'LogisticRegression':
                params = {'C': [0.01, 0.1, 1, 10]}
            elif name == 'RandomForest':
                params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
            else:
                params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}

            grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            mlflow.log_params(grid.best_params_)
            mlflow.log_param("model_type", name)

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, f"model_{name}")
            print(f"{name} - Best params: {grid.best_params_}")
            print(f"{name} - ROC-AUC: {metrics['roc_auc']:.4f}")

    best_run_id = mlflow.last_active_run().info.run_id
    model_uri = f"runs:/{best_run_id}/model_GradientBoosting"
    registered_model = mlflow.register_model(model_uri, "CreditRiskModel")
    print(f"Model registered as version {registered_model.version}")


if __name__ == '__main__':
    train_and_log()
