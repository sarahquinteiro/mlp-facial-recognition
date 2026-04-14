"""
preprocessing.py
----------------
Pré-processamento de imagens faciais para a rede MLP.
Paradigma Conexionista — Equipe 4 — Fatec Osasco
"""

import os
import argparse
import numpy as np
from PIL import Image


TARGET_SIZE = (48, 48)


def load_image(path: str) -> np.ndarray:
    """Carrega uma imagem em escala de cinza e redimensiona para TARGET_SIZE."""
    img = Image.open(path).convert("L")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    """Normaliza os pixels para o intervalo [0, 1]."""
    return img / 255.0


def flatten(img: np.ndarray) -> np.ndarray:
    """Transforma a imagem 2D em vetor 1D (input da MLP)."""
    return img.flatten()


def preprocess_image(path: str) -> np.ndarray:
    """Pipeline completo: carregar → normalizar → achatar."""
    img = load_image(path)
    img = normalize(img)
    return flatten(img)


def load_dataset_from_dir(root: str):
    """
    Carrega dataset de um diretório organizado por classe:
        root/
            pessoa_01/foto1.jpg
            pessoa_02/foto1.jpg

    Retorna
    -------
    X : np.ndarray shape (N, 2304)
    y : np.ndarray shape (N,)
    class_names : list[str]
    """
    class_names = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    X, y = [], []
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(root, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(cls_dir, fname)
                try:
                    vec = preprocess_image(fpath)
                    X.append(vec)
                    y.append(label)
                except Exception as e:
                    print(f"[AVISO] Erro ao processar {fpath}: {e}")

    return np.array(X), np.array(y), class_names


def load_olivetti():
    """
    Carrega o dataset Olivetti Faces via scikit-learn.
    40 pessoas × 10 fotos, 64×64 px → redimensionado para 48×48.

    Retorna
    -------
    X : np.ndarray shape (400, 2304)
    y : np.ndarray shape (400,)
    class_names : list[str]
    """
    from sklearn.datasets import fetch_olivetti_faces
    print("Carregando Olivetti Faces...")
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    images = dataset.images           # (400, 64, 64)
    labels = dataset.target           # (400,)

    X = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        X.append(arr.flatten())

    class_names = [f"Pessoa_{i+1:02d}" for i in range(40)]
    return np.array(X), labels, class_names


def save_processed(X: np.ndarray, y: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Dataset salvo em {output_dir}  |  X: {X.shape}  y: {y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processamento de imagens faciais")
    parser.add_argument("--input", default=None, help="Diretório com imagens (organizado por classe)")
    parser.add_argument("--output", default="data/processed", help="Diretório de saída")
    parser.add_argument("--olivetti", action="store_true", help="Usar dataset Olivetti (scikit-learn)")
    args = parser.parse_args()

    if args.olivetti or args.input is None:
        X, y, names = load_olivetti()
    else:
        X, y, names = load_dataset_from_dir(args.input)

    print(f"Classes encontradas: {names}")
    save_processed(X, y, args.output)
