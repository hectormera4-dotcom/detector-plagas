import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Plagas en Hojas")
st.markdown("### Modelo YOLO11s - mAP50: 82.7%")

# Configuración Telegram (SOLO para enviar alertas)
TELEGRAM_BOT_TOKEN = "8725129241:AAGBYwVLnmVfbBUa9RVjIdQD2AaOswKjinc"
TELEGRAM_CHAT_ID = "7700414080"
ecuador_tz = timezone(timedelta(hours=-5))

def enviar_alerta_telegram(clase, conf, imagen_bytes):
    """Envía alerta SOLO si es crítico o nada saludable"""
    if clase not in ['Crítico', 'Nada Saludable']:
        return
    
    ahora = datetime.now(ecuador_tz)
    mensaje = f"""
🚨 *ALERTA DE PLAGA DETECTADA*

🍃 *Clase:* {clase}
📊 *Confianza:* {conf:.2f}%
 *Hora:* {ahora.strftime('%H:%M:%S')}
📅 *Fecha:* {ahora.strftime('%d/%m/%Y')}

️ *Acción recomendada:* Revisar planta inmediatamente
    """
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        url_foto = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {'photo': imagen_bytes}
        requests.post(url_foto, files=files, 
                     data={"chat_id": TELEGRAM_CHAT_ID}, timeout=10)
    except:
        pass

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        model_url = "https://huggingface.co/EAMB2001/detector-plagas-modelo/resolve/main/modelo.pt"
        response = requests.get(model_url, timeout=60)
        model_path = "/tmp/modelo.pt"
        with open(model_path, "wb") as f:
            f.write(response.content)
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

model = load_model()
CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']

if model is None:
    st.error("❌ Error cargando el modelo")
    st.stop()

# Interfaz
st.sidebar.info("""
**Funcionalidades:**
- ✅ Subir imágenes desde la web
- ✅ Análisis con YOLO11s (82.7% mAP)
- ✅ Alertas automáticas a Telegram cuando se detectan casos críticos
""")

uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original")
    
    if st.button("🔍 Analizar Hoja"):
        with st.spinner("Analizando imagen..."):
            try:
                image = Image.open(uploaded_file)
                temp_path = "/tmp/temp.jpg"
                image.save(temp_path)
                
                results = model(temp_path, verbose=False)
                boxes = results[0].boxes
                
                with col2:
                    if len(boxes) > 0:
                        mejor = max(boxes, key=lambda b: float(b.conf[0]))
                        clase = CLASSES[int(mejor.cls[0])]
                        conf = float(mejor.conf[0]) * 100
                        
                        st.success(f"✅ **{clase}**")
                        st.metric("Confianza", f"{conf:.2f}%")
                        
                        result_img = results[0].plot()
                        st.image(result_img, caption="Resultado")
                        
                        # Enviar alerta si es crítico
                        if clase in ['Crítico', 'Nada Saludable']:
                            img_bytes = BytesIO()
                            image.save(img_bytes, format='JPEG')
                            enviar_alerta_telegram(clase, conf, img_bytes)
                            st.warning("⚠️ **Alerta enviada a Telegram**")
                    else:
                        st.warning("No se detectó ninguna hoja")
                        
            except Exception as e:
                st.error(f"Error procesando: {e}")
