import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import json
import os
from collections import deque, Counter
from gtts import gTTS
from io import BytesIO
import streamlit.components.v1 as components

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


@st.cache_resource
def load_model():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,          # Solo 1 mano: más rápido
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def get_camera():
    if "camera" in st.session_state and st.session_state.camera is not None:
        if st.session_state.camera.isOpened():
            return st.session_state.camera

    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 15)
            st.session_state.camera = cap
            return cap
        cap.release()
    return None


def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, h, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
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

    if f == [True, True, True, True] and thumb_out:   return "Hola",      "Mano abierta"
    if f == [False,False,False,False] and thumb_up \
       and not thumb_out:                              return "Bien",      "Pulgar arriba"
    if f == [False,False,False,False] and not thumb_up \
       and not thumb_out:                              return "Pare",      "Puño cerrado"
    if f == [True, False,False,True]  and thumb_out:  return "Te amo",    "Seña ILY"
    if f == [True, True, False,False] and not thumb_out: return "Paz",    "Dedos en V"
    if f == [True, False,False,False] and not thumb_out: return "Uno",    "Índice arriba"
    if f == [True, True, False,False] and thumb_out:  return "Dos",       "Dos dedos"
    if f == [True, True, True, False] and not thumb_out: return "Tres",   "Tres dedos"
    if f == [True, True, True, True]  and not thumb_out: return "Cuatro", "Cuatro dedos"
    if f == [False,False,False,True]  and thumb_out:  return "Llámame",   "Seña teléfono"

    return "", ""


# ──────────────────────────────────────────────────────────────
#  AUDIO — gTTS en hilo separado para no bloquear el loop
# ──────────────────────────────────────────────────────────────
_audio_cache: dict[str, bytes] = {}
_audio_lock = threading.Lock()

def _generar_en_hilo(texto: str):
    try:
        tts = gTTS(text=texto, lang="es", slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        with _audio_lock:
            _audio_cache[texto] = fp.getvalue()
    except Exception:
        pass

def pedir_audio(texto: str) -> bytes | None:
    """Retorna bytes si ya está cacheado, si no dispara hilo en background."""
    with _audio_lock:
        if texto in _audio_cache:
            return _audio_cache[texto]
    # Generar en background (no bloquea)
    t = threading.Thread(target=_generar_en_hilo, args=(texto,), daemon=True)
    t.start()
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
        camera_on = st.toggle("Activar Cámara", value=False)
        pausar_on = st.toggle("Pausar Traducción", value=False, help="Congela la IA temporalmente para que no detecte señas por accidente al cambiar de pestaña.")
        
        # HACK MAGICO: Script invisible que detecta cambio de pestaña y presiona el toggle de pausa
        components.html("""
        <script>
        const doc = window.parent.document;
        doc.addEventListener('visibilitychange', function() {
            const labels = doc.querySelectorAll('label');
            for (let label of labels) {
                if (label.innerText.includes('Pausar Traducción')) {
                    const toggleInput = label.querySelector('input');
                    if (toggleInput) {
                        // Si la pestaña se oculta y no está pausado -> Pausar
                        if (doc.hidden && !toggleInput.checked) toggleInput.click();
                        // Si la pestaña vuelve y estaba pausado -> Despausar
                        else if (!doc.hidden && toggleInput.checked) toggleInput.click();
                    }
                }
            }
        });
        </script>
        """, height=0, width=0)

        audio_on  = st.toggle("Voz Automática", value=True)
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
        
        # Función para pintar el historial como una frase
        def update_historial():
            if not st.session_state.historial:
                historial_ph.markdown("<span style='color:#64748B;font-size:0.9rem;'>Empieza a hacer señas...</span>", unsafe_allow_html=True)
                copy_ph.empty()
            else:
                # Límite interno alto (200) solo por seguridad de RAM
                if len(st.session_state.historial) > 200:
                    st.session_state.historial = st.session_state.historial[-200:]
                
                frase = " ".join(st.session_state.historial)
                html = f"<div style='background:#1E293B; border:1px solid #334155; padding:10px 14px; border-radius:12px; font-size:1.1rem; color:#E2E8F0; margin-bottom:8px;'>{frase}</div>"
                historial_ph.markdown(html, unsafe_allow_html=True)
                
                # st.code incluye un botón nativo de "Copiar al portapapeles" en la esquina
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

    with col_cam:
        video_ph = st.empty()

    if not camera_on:
        video_ph.image(
            "https://images.unsplash.com/photo-1557804506-669a67965ba0?w=900",
            use_container_width=True,
        )
        status_ph.markdown('<span class="badge-off">🔴 Inactivo</span>', unsafe_allow_html=True)
        transl_ph.markdown(
            '<div class="sena-card" style="color:#475569;font-size:1.1rem">'
            'Activa la cámara para comenzar.</div>',
            unsafe_allow_html=True,
        )
        return

    hands_model = load_model()
    cap = get_camera()

    if cap is None:
        status_ph.markdown('<span class="badge-off">❌ Sin cámara</span>', unsafe_allow_html=True)
        st.error("No se detectó cámara. Asegúrate de haber instalado opencv-python (no headless).")
        return

    status_ph.markdown('<span class="badge-on">🟢 IA Activa</span>', unsafe_allow_html=True)

    last_text   = ""
    frame_count = 0
    last_result = None
    
    # 🧠 Memoria Caché para estabilizar las señas y evitar errores/flickering
    sign_buffer = deque(maxlen=5) 
    
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)

                if frame_count % 2 == 0:
                    image.flags.writeable = False
                    last_result = hands_model.process(image)
                    image.flags.writeable = True

                if last_result:
                    draw_landmarks(image, last_result)

                # Enviar video a la UI
                video_ph.image(image, channels="RGB", use_container_width=True)

                if pausar_on:
                    transl_ph.markdown(
                        '<div class="sena-card" style="color:#64748b;font-size:1.1rem; border-left:5px solid #64748b;">'
                        '⏸ Traducción pausada</div>', unsafe_allow_html=True)
                    last_text = "Pausado"
                    time.sleep(0.05)
                    continue

                if last_result:
                    etiqueta, descripcion = clasificar(last_result)
                    
                    # Añadir a la memoria caché
                    if etiqueta:
                        sign_buffer.append((etiqueta, descripcion))
                    else:
                        sign_buffer.append(("", ""))

                    # Solo actuar si tenemos la memoria llena para estar seguros
                    if len(sign_buffer) == 5:
                        # Obtener la seña más común en los últimos 5 frames
                        etiquetas = [x[0] for x in sign_buffer if x[0]]
                        if etiquetas:
                            conteo = Counter(etiquetas)
                            seña_fuerte, apariciones = conteo.most_common(1)[0]
                            
                            # Si es dominante (aparece al menos 3 de 5 veces) y es nueva
                            if apariciones >= 3 and seña_fuerte != last_text:
                                # Buscar la descripción correspondiente
                                desc_fuerte = next(x[1] for x in sign_buffer if x[0] == seña_fuerte)
                                
                                transl_ph.markdown(
                                    f'<div class="sena-card">{seña_fuerte}'
                                    f'<div class="sena-sub">{desc_fuerte}</div></div>',
                                    unsafe_allow_html=True,
                                )
                                
                                if audio_on:
                                    audio_bytes = pedir_audio(seña_fuerte)
                                    if audio_bytes:
                                        unique_bytes = audio_bytes + str(time.time()).encode()
                                        audio_ph.empty()
                                        audio_ph.audio(unique_bytes, format="audio/mp3", autoplay=True)
                                
                                last_text = seña_fuerte
                                st.session_state.historial.append(seña_fuerte)
                                update_historial()
                        else:
                            # Si no hay señas en el buffer, mostrar estado de análisis
                            if last_result.multi_hand_landmarks and last_text != "Analizando...":
                                transl_ph.markdown(
                                    '<div class="sena-card" style="color:#64748b;font-size:1.2rem">'
                                    'Analizando...</div>', unsafe_allow_html=True)
                                last_text = "Analizando..."
                            elif not last_result.multi_hand_landmarks and last_text != "Esperando":
                                transl_ph.markdown(
                                    '<div class="sena-card" style="color:#475569;font-size:1.1rem">'
                                    'Muestra tu mano 🖐</div>', unsafe_allow_html=True)
                                last_text = "Esperando"

                time.sleep(0.04) 
                
            except Exception as e:
                # Si Streamlit interrumpe el script (ej: clic en botón), salir limpio
                if type(e).__name__ in ['ScriptControlException', 'StopException']:
                    break
                print(f"Error en loop de cámara: {e}")
                time.sleep(0.1) # Pausa breve para recuperarse de errores
                
    finally:
        pass # NO cerramos la cámara aquí para que sobreviva a los clics de botones


if __name__ == "__main__":
    main()