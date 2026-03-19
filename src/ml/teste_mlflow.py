import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("teste-inicial")

with mlflow.start_run(run_name="minha-primeira-run-v2"):

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"max_iter": 100, "C": 1.0, "solver": "lbfgs"}
    mlflow.log_params(params)

    modelo = LogisticRegression(**params)
    modelo.fit(X_train, y_train)

    acuracia = accuracy_score(y_test, modelo.predict(X_test))
    mlflow.log_metric("acuracia", acuracia)

    # salva o modelo de forma simples, sem a API nova
    import pickle, os
    os.makedirs("mlflow-data/artifacts", exist_ok=True)
    with open("mlflow-data/artifacts/modelo.pkl", "wb") as f:
        pickle.dump(modelo, f)

    mlflow.log_artifact("mlflow-data/artifacts/modelo.pkl")

    print(f"Experimento registrado")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Veja em: http://localhost:5000")