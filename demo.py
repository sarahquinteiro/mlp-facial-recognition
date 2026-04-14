"""
demo.py
-------
Demo rápido: treina a rede no dataset Olivetti e exibe os resultados.
Execute: python demo.py

Paradigma Conexionista — Equipe 4 — Fatec Osasco
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocessing import load_olivetti
from src.model import MLP
from src.train import train, plot_history
from src.utils import precision_recall_f1, print_metrics, visualize_sample_predictions


def main():
    print("=" * 60)
    print("  Reconhecimento Facial com MLP — Fatec Osasco")
    print("  Equipe 4 — Paradigma Conexionista")
    print("=" * 60 + "\n")

    # 1. Carregar dados
    X, y, class_names = load_olivetti()
    print(f"Dataset: {X.shape[0]} imagens | {len(class_names)} pessoas | {X.shape[1]} features\n")

    # 2. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # 3. Criar modelo
    model = MLP(
        n_input=X.shape[1],
        hidden1=128,
        hidden2=64,
        n_classes=len(class_names),
        lr=0.001,
        optimizer="adam",
    )
    print(model)
    print(f"Total de parâmetros: {model.count_params():,}\n")

    # 4. Treinar
    print("Treinando por 50 épocas com Backpropagation + Adam...\n")
    model = train(
        model, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=32, verbose=True
    )

    # 5. Avaliar
    test_acc = model.accuracy(X_test, y_test) * 100
    print(f"\n{'='*40}")
    print(f"  Acurácia no teste: {test_acc:.2f}%")
    print(f"{'='*40}\n")

    y_pred = model.predict(X_test)
    metrics = precision_recall_f1(y_test, y_pred, len(class_names))
    print_metrics(metrics)

    # 6. Salvar modelo
    os.makedirs("models", exist_ok=True)
    model.save("models/mlp_weights.pkl")

    # 7. Visualizações
    os.makedirs("docs", exist_ok=True)

    plot_history(model.history, save_path="docs/curvas_treinamento.png")

    fig = visualize_sample_predictions(model, X_test, y_test, class_names, n_show=8)
    fig.savefig("docs/predicoes_exemplo.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nArquivos gerados:")
    print("  models/mlp_weights.pkl          — pesos do modelo")
    print("  docs/curvas_treinamento.png     — curvas de loss e acurácia")
    print("  docs/predicoes_exemplo.png      — exemplos de predição")
    print("\nDemo concluído!")


if __name__ == "__main__":
    main()
