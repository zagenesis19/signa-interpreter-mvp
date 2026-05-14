# 🤟 Signa Interpreter

**Platanus Build Night 26 · Caracas, Venezuela 🇻🇪**
*Built by Genesis Zapata ([@zagenesis19](https://github.com/zagenesis19))*

---

## ¿Qué es Signa?

Signa es un intérprete de **Lenguaje de Señas** en tiempo real diseñado para **taquillas B2B** (Bancos y Gobierno). La cámara web detecta señas de manos y las traduce a texto profesional, eliminando barreras de comunicación para personas con discapacidad auditiva en entornos institucionales.

---

## 🚀 Demo

> Accede en `localhost:8501` tras ejecutar el servidor local.

| Pantalla principal | Detección activa |
|---|---|
| Imagen de espera con UI premium | Landmarks dibujados en tiempo real sobre la cámara |

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io) |
| **Visión por Computadora** | [MediaPipe Tasks API](https://developers.google.com/mediapipe) (HandLandmarker) |
| **Procesamiento de Imagen** | OpenCV 4.x |
| **IA Generativa** | [Anthropic Claude](https://anthropic.com) *(integración futura)* |
| **TTS** | gTTS (Text-to-Speech) |
| **Lenguaje** | Python 3.13 |

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/zagenesis19/platanus-build-night-26-ve-zagenesis19.git
cd platanus-build-night-26-ve-zagenesis19
```

### 2. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar

```bash
streamlit run app.py
```

El modelo de manos (`hand_landmarker.task`) se descarga **automáticamente** la primera vez (~8 MB).

---

## 📁 Estructura del Proyecto

```
signa-interpreter/
├── app.py                    # Aplicación principal Streamlit
├── requirements.txt          # Dependencias Python
├── hand_landmarker.task      # Modelo MediaPipe (auto-descargado)
├── holistic_landmarker.task  # Modelo holístico (referencia)
└── README.md
```

---

## 🧩 Arquitectura Modular

```
app.py
 ├── DetectorDeManos        # MediaPipe HandLandmarker (Tasks API)
 │    ├── procesar()        # Inferencia por frame
 │    ├── dibujar()         # Renderiza landmarks
 │    └── hay_manos()       # Bool de detección
 ├── traducir_senas()       # Stub → aquí va el clasificador/LLM
 └── main()                 # UI Streamlit + bucle de cámara
```

La función `traducir_senas()` está diseñada como punto de extensión: conecta un clasificador de gestos o una llamada a la API de Claude para generar traducciones semánticas.

---

## 🗺️ Roadmap

- [x] Detección de manos en tiempo real (MediaPipe)
- [x] UI B2B premium (modo oscuro, diseño institucional)
- [x] Arquitectura modular para traducción
- [ ] Clasificador de señas básicas (A-Z + números)
- [ ] Integración Claude para frases complejas
- [ ] Salida de voz con gTTS
- [ ] Modo quiosco para taquillas físicas

---

## 🙏 Créditos

- **Google MediaPipe** — Modelos de detección de manos
- **Anthropic** — Patrocinador de la hackathon
- **Platanus** — Organización del Build Night

---

*Made with ❤️ in Caracas, Venezuela during Platanus Build Night 26*
