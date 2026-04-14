"""
inference.py
------------
Inferência com o modelo MLP treinado.
Aceita imagem estática ou leitura via webcam (OpenCV).

Paradigma Conexionista — Equipe 4 — Fatec Osasco
"""

import os
import sys
import argparse
import numpy as np

from model import MLP
from preprocessing import preprocess_image, TARGET_SIZE


def predict_single(model: MLP, image_path: str, class_names: list) -> dict:
    """
    Realiza o reconhecimento facial em uma única imagem.

    Retorna
    -------
    dict com 'predicted_class', 'confidence' e 'all_probs'
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Pré-processamento
    X = preprocess_image(image_path).reshape(1, -1)

    # Forward pass
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def print_result(result: dict, top_k: int = 5):
    """Exibe o resultado da inferência de forma legível."""
    print("\n" + "=" * 45)
    print(f"  Identidade detectada : {result['predicted_class']}")
    print(f"  Confiança            : {result['confidence']*100:.1f}%")
    print("-" * 45)
    print(f"  Top-{top_k} probabilidades:")
    sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)
    for name, prob in sorted_probs[:top_k]:
        bar = "█" * int(prob * 30)
        print(f"    {name:15s} {prob*100:5.1f}%  {bar}")
    print("=" * 45 + "\n")


def run_webcam(model: MLP, class_names: list, cascade_path: str = None):
    """
    Reconhecimento facial em tempo real via webcam.
    Requer OpenCV com suporte a câmera.
    """
    try:
        import cv2
    except ImportError:
        print("[ERRO] OpenCV não instalado. Execute: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    # Detector de faces Haar Cascade
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("Webcam ativa — pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Pré-processar o recorte
            import tempfile
            from PIL import Image
            pil_img = Image.fromarray(face_roi).resize(TARGET_SIZE)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name)
                vec = preprocess_image(tmp.name).reshape(1, -1)
                os.unlink(tmp.name)

            probs = model.predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            label = f"{class_names[pred_idx]} ({probs[pred_idx]*100:.0f}%)"

            # Desenhar no frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (127, 119, 221), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 119, 221), 2)

        cv2.imshow("Reconhecimento Facial — MLP", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inferência MLP — Reconhecimento Facial")
    parser.add_argument("--model", default="models/mlp_weights.pkl", help="Caminho para o modelo salvo")
    parser.add_argument("--image", default=None, help="Caminho para imagem estática")
    parser.add_argument("--webcam", action="store_true", help="Usar webcam em tempo real")
    parser.add_argument("--classes", nargs="+", default=None, help="Lista de nomes das classes")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Carregar modelo
    model = MLP.load(args.model)

    # Nomes das classes
    if args.classes:
        class_names = args.classes
    else:
        class_names = [f"Pessoa_{i+1:02d}" for i in range(model.n_classes)]

    if args.webcam:
        run_webcam(model, class_names)
    elif args.image:
        result = predict_single(model, args.image, class_names)
        print_result(result, top_k=args.top_k)
    else:
        print("Use --image <caminho> ou --webcam para inferência.")
        parser.print_help()


if __name__ == "__main__":
    main()
