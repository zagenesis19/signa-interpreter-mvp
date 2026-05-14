import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import Counter
from gtts import gTTS
from io import BytesIO
from PIL import Image

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


@st.cache_resource
def load_model():
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
    )


# ──────────────────────────────────────────────────────────────
#  CLASIFICACIÓN — 10 señas estáticas por heurística
# ──────────────────────────────────────────────────────────────
def clasificar(results) -> tuple[str, str]:
    if not results.multi_hand_landmarks:
        return "", ""

    lm = results.multi_hand_landmarks[0].landmark

    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    thumb_up  = lm[4].y  < lm[3].y and lm[4].y < lm[2].y
    thumb_out = abs(lm[4].x - lm[0].x) > abs(lm[2].x - lm[0].x)

    f = [index_up, middle_up, ring_up, pinky_up]

    if f == [True, True, True, True] and thumb_out:   return "👋 Hola",      "Mano abierta"
    if f == [False,False,False,False] and thumb_up \
       and not thumb_out:                              return "👍 Bien",      "Pulgar arriba"
    if f == [False,False,False,False] and not thumb_up \
       and not thumb_out:                              return "✊ Pare",      "Puño cerrado"
    if f == [True, False,False,True]  and thumb_out:  return "🤟 Te amo",    "Seña ILY"
    if f == [True, True, False,False] and not thumb_out: return "✌️ Paz",    "Dedos en V"
    if f == [True, False,False,False] and not thumb_out: return "☝️ Uno",    "Índice arriba"
    if f == [True, True, False,False] and thumb_out:  return "✌️ Dos",       "Dos dedos"
    if f == [True, True, True, False] and not thumb_out: return "3️⃣ Tres",   "Tres dedos"
    if f == [True, True, True, True]  and not thumb_out: return "4️⃣ Cuatro", "Cuatro dedos"
    if f == [False,False,False,True]  and thumb_out:  return "🤙 Llámame",   "Seña teléfono"

    return "", ""


# ──────────────────────────────────────────────────────────────
#  AUDIO — gTTS
# ──────────────────────────────────────────────────────────────
_audio_cache: dict[str, bytes] = {}

def get_audio(texto: str) -> bytes | None:
    if texto in _audio_cache:
        return _audio_cache[texto]
    try:
        tts = gTTS(text=texto, lang="es", slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        _audio_cache[texto] = fp.getvalue()
        return _audio_cache[texto]
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────
def main():
    if "historial" not in st.session_state:
        st.session_state.historial = []

    st.set_page_config(page_title="Signa Interpreter | B2B", page_icon="🤟", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    .stApp { background:#0F172A; color:#F8FAFC; font-family:'Inter',sans-serif; }
    h1 { background:linear-gradient(90deg,#F59E0B,#FBBF24);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         font-weight:800; font-size:2.6rem !important; margin:0; }
    h2,h3 { color:#E2E8F0 !important; }
    .stButton>button {
        width:100%; height:52px !important; border-radius:10px !important;
        background:#F59E0B !important; color:#0F172A !important;
        font-weight:700 !important; font-size:1rem !important; border:none !important;
        transition:all 0.2s ease; box-shadow:0 4px 12px rgba(245,158,11,0.3);
    }
    .stButton>button:hover {
        transform:translateY(-2px); box-shadow:0 6px 18px rgba(245,158,11,0.5);
        background:#FBBF24 !important;
    }
    .badge-on  { background:#14532d; color:#86efac; border-radius:8px;
                 padding:6px 16px; font-size:.9rem; font-weight:600; display:inline-block; }
    .badge-off { background:#1E293B; color:#64748B; border-radius:8px;
                 padding:6px 16px; font-size:.9rem; display:inline-block; }
    .sena-card {
        background:linear-gradient(135deg,#1E293B,#0F172A);
        border-left:5px solid #F59E0B; border-radius:14px;
        padding:20px 24px; margin-top:8px;
        font-size:2.2rem; font-weight:800; color:#F59E0B; min-height:90px;
    }
    .sena-sub { color:#94A3B8; font-size:.85rem; margin-top:4px; font-weight:400; }
    .stImage img { border-radius:16px; border:3px solid #1E293B; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 9])
    with c1:
        st.markdown("<div style='font-size:3rem;line-height:1'>🤟</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("# Signa Interpreter")
    st.caption("Sistema de Accesibilidad B2B · Platanus Build Night 26 · Caracas 🇻🇪")
    st.divider()

    col_cam, col_ctrl = st.columns([3, 1], gap="large")

    with col_ctrl:
        st.subheader("⚙️ Controles")
        audio_on = st.toggle("🔊 Voz Automática", value=True)
        st.divider()

        st.subheader("🟢 Estado")
        status_ph = st.empty()
        st.divider()

        st.subheader("📝 Traducción")
        transl_ph = st.empty()
        audio_ph  = st.empty()
        st.divider()

        st.subheader("💬 Frase Construida")

        if st.button("🗑️ Limpiar Frase", use_container_width=True):
            st.session_state.historial = []

        historial_ph = st.empty()
        copy_ph = st.empty()

        def update_historial():
            if not st.session_state.historial:
                historial_ph.markdown("<span style='color:#64748B;font-size:0.9rem;'>Empieza a hacer señas...</span>", unsafe_allow_html=True)
                copy_ph.empty()
            else:
                if len(st.session_state.historial) > 200:
                    st.session_state.historial = st.session_state.historial[-200:]
                frase = " ".join(st.session_state.historial)
                html = f"<div style='background:#1E293B; border:1px solid #334155; padding:10px 14px; border-radius:12px; font-size:1.1rem; color:#E2E8F0; margin-bottom:8px;'>{frase}</div>"
                historial_ph.markdown(html, unsafe_allow_html=True)
                copy_ph.code(frase, language=None)

        update_historial()
        st.divider()

        with st.expander("📖 Señas disponibles"):
            st.markdown("""
| Seña | Posición |
|------|---------|
| 👋 Hola | Mano abierta |
| 👍 Bien | Pulgar arriba |
| ✊ Pare | Puño cerrado |
| 🤟 Te amo | ILY sign |
| ✌️ Paz | V con dedos |
| ☝️ Uno | Índice arriba |
| 2️⃣ Dos | Dos dedos |
| 3️⃣ Tres | Tres dedos |
| 4️⃣ Cuatro | Cuatro dedos |
| 🤙 Llámame | Teléfono |
""")
        st.caption("Powered by MediaPipe · Platanus '26")

    # ── Cámara nativa de Streamlit (funciona en CUALQUIER plataforma) ──
    with col_cam:
        st.markdown("##### 📸 Muestra tu seña y presiona el botón de captura")
        photo = st.camera_input("Captura tu seña", key="camera", label_visibility="collapsed")

        if photo is not None:
            # Cargar la imagen capturada
            img = Image.open(photo)
            img_array = np.array(img)

            # Procesar con MediaPipe
            hands_model = load_model()
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            rgb2 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            results = hands_model.process(rgb2)

            if results.multi_hand_landmarks:
                # Dibujar landmarks sobre la imagen
                annotated = img_array.copy()
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, hand, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                st.image(annotated, caption="🔍 Mano detectada", use_container_width=True)

                label, desc = clasificar(results)
                if label:
                    status_ph.markdown('<span class="badge-on">🟢 Seña Detectada</span>', unsafe_allow_html=True)
                    transl_ph.markdown(
                        f'<div class="sena-card">{label}'
                        f'<div class="sena-sub">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )

                    # Audio
                    if audio_on:
                        # Extraer solo el texto sin emoji para TTS
                        clean_label = label.split(" ", 1)[-1] if " " in label else label
                        audio_bytes = get_audio(clean_label)
                        if audio_bytes:
                            audio_ph.audio(audio_bytes, format="audio/mp3", autoplay=True)

                    # Agregar al historial
                    st.session_state.historial.append(label)
                    update_historial()
                else:
                    status_ph.markdown('<span class="badge-on">🟡 Mano visible</span>', unsafe_allow_html=True)
                    transl_ph.markdown(
                        '<div class="sena-card" style="color:#475569;font-size:1.1rem">'
                        'Seña no reconocida. Intenta otra posición.</div>',
                        unsafe_allow_html=True,
                    )
            else:
                status_ph.markdown('<span class="badge-off">🔴 Sin mano</span>', unsafe_allow_html=True)
                transl_ph.markdown(
                    '<div class="sena-card" style="color:#475569;font-size:1.1rem">'
                    'No se detectó mano. Asegúrate de mostrar tu mano claramente.</div>',
                    unsafe_allow_html=True,
                )
        else:
            status_ph.markdown('<span class="badge-off">🔴 Esperando captura</span>', unsafe_allow_html=True)
            transl_ph.markdown(
                '<div class="sena-card" style="color:#475569;font-size:1.1rem">'
                'Presiona el botón 📸 para capturar tu seña.</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()