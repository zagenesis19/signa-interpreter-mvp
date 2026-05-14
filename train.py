"""
train.py — Herramienta de entrenamiento para Signa Interpreter
================================================================
Uso:
    python train.py

Corre esto ANTES de levantar la app principal. Permite registrar
nuevas señas en gestos.json sin exponer esa funcionalidad al público.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os

GESTOS_FILE = "gestos.json"

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def preprocesar(landmarks):
    bx, by, bz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    puntos = [[l.x - bx, l.y - by, l.z - bz] for l in landmarks]
    dist = np.linalg.norm(np.array(puntos[9]) - np.array(puntos[0]))
    if dist > 0:
        puntos = (np.array(puntos) / dist).tolist()
    return puntos


def cargar_gestos():
    if os.path.exists(GESTOS_FILE):
        try:
            with open(GESTOS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def guardar_gesto(nombre, samples):
    """Guarda el promedio de múltiples muestras para mayor robustez."""
    gestos = cargar_gestos()
    arr = np.array(samples)
    promedio = np.mean(arr, axis=0).tolist()
    gestos[nombre] = promedio
    with open(GESTOS_FILE, "w") as f:
        json.dump(gestos, f, indent=2)
    print(f"\n  ✅ '{nombre}' guardado con {len(samples)} muestra(s).")


def get_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    return None


def main():
    print("=" * 55)
    print("  🤟 SIGNA INTERPRETER — Herramienta de Entrenamiento")
    print("=" * 55)
    print("  Teclas:")
    print("    [ESPACIO] = Capturar muestra")
    print("    [S]       = Guardar gesto con todas las muestras")
    print("    [L]       = Listar gestos guardados")
    print("    [D]       = Eliminar un gesto")
    print("    [Q]       = Salir")
    print("=" * 55)

    gestos_actuales = cargar_gestos()
    print(f"\n  Gestos en base de datos: {list(gestos_actuales.keys()) or 'Ninguno'}\n")

    nombre = input("  Nombre de la seña a registrar: ").strip()
    if not nombre:
        print("Nombre vacío, saliendo.")
        return

    cap = get_camera()
    if cap is None:
        print("❌ No se encontró ninguna cámara.")
        return

    SAMPLES_OBJETIVO = 10
    samples = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        print(f"\n  Registrando: '{nombre}'")
        print(f"  Haz la seña y pulsa [ESPACIO] {SAMPLES_OBJETIVO} veces para capturar.")
        print("  Cambia un poco el ángulo entre capturas para mayor robustez.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for h in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, h, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # Overlay
            status = f"Muestras: {len(samples)}/{SAMPLES_OBJETIVO} | [ESPACIO]=Capturar [S]=Guardar [Q]=Salir"
            cv2.rectangle(image, (0, 0), (640, 40), (15, 23, 42), -1)
            cv2.putText(image, status, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 1)
            cv2.putText(image, f"Seña: {nombre}", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(f"Entrenamiento — {nombre}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n  Saliendo sin guardar.")
                break

            elif key == ord(' '):
                if results.multi_hand_landmarks:
                    pts = preprocesar(results.multi_hand_landmarks[0].landmark)
                    samples.append(pts)
                    print(f"  📸 Muestra {len(samples)}/{SAMPLES_OBJETIVO} capturada")
                    if len(samples) >= SAMPLES_OBJETIVO:
                        guardar_gesto(nombre, samples)
                        print(f"\n  Objetivo alcanzado. Puedes salir con [Q] o seguir con otra seña.")
                else:
                    print("  ⚠️  No se detectó mano. Ponla frente a la cámara.")

            elif key == ord('s'):
                if samples:
                    guardar_gesto(nombre, samples)
                    samples = []
                    nombre = input("\n  Nombre de la siguiente seña (o Enter para salir): ").strip()
                    if not nombre:
                        break
                    print(f"\n  Registrando: '{nombre}'")
                else:
                    print("  ⚠️  Captura al menos una muestra primero.")

            elif key == ord('l'):
                g = cargar_gestos()
                print(f"\n  Gestos en base de datos ({len(g)}): {list(g.keys())}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Entrenamiento finalizado. Reinicia la app para cargar los cambios.")
    print(f"  Gestos guardados: {list(cargar_gestos().keys())}\n")


if __name__ == "__main__":
    main()
