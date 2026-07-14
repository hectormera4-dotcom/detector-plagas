import gradio as gr
from ultralytics import YOLO
from PIL import Image
import requests
import os
import threading
from datetime import datetime

# ==========================================
# CONFIGURACIÓN (Se carga desde Secrets)
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ==========================================
# FUNCIÓN TELEGRAM
# ==========================================
def enviar_telegram(mensaje, imagen_path=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML"
    }
    
    try:
        requests.post(url, data=data, timeout=10)
        
        if imagen_path and os.path.exists(imagen_path):
            url_foto = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(imagen_path, 'rb') as img:
                files = {'photo': img}
                data_foto = {"chat_id": TELEGRAM_CHAT_ID}
                requests.post(url_foto, data=data_foto, files=files, timeout=10)
        
        print("✅ Enviado a Telegram")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")

# ==========================================
# CARGAR MODELO
# ==========================================
print("🚀 Cargando modelo YOLO...")
model = YOLO('modelo.pt')  # El modelo está en la misma carpeta
CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']
print("✅ Modelo cargado")

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================
def detectar_plaga(image):
    if image is None:
        return None, "️ Sube una imagen válida"
    
    # Guardar temporal
    temp_path = '/tmp/temp_image.jpg'
    image.save(temp_path)
    
    # Predecir
    results = model(temp_path, verbose=False)
    
    # Procesar resultados
    boxes = results[0].boxes
    detecciones = []
    
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detecciones.append(f"• {CLASSES[cls]}: {conf*100:.2f}%")
        
        clase_dom = max(detecciones, key=lambda x: float(x.split(': ')[1].replace('%', '')))
    else:
        clase_dom = "Sin detecciones"
        detecciones = ["  No se detectó ninguna hoja"]
    
    # Imagen con bounding boxes
    result_image = results[0].plot()
    result_path = '/tmp/result_image.jpg'
    Image.fromarray(result_image).save(result_path)
    
    # Texto de resultado
    texto = f"{'='*50}\n"
    texto += f"🎯 CLASE DOMINANTE: {clase_dom}\n"
    texto += f"{'='*50}\n\n"
    texto += f"📊 DETECCIONES:\n"
    for det in detecciones:
        texto += f"  {det}\n"
    
    # Enviar a Telegram (en segundo plano)
    if TELEGRAM_BOT_TOKEN:
        mensaje = f"""
🍃 <b>DETECTOR DE PLAGAS</b>

🎯 <b>Clase:</b> {clase_dom}

📊 <b>Detecciones:</b>
{chr(10).join(detecciones)}

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title(" Detector de Plagas en Hojas")
st.markdown("### Modelo YOLO11s - mAP50: 82.7%")

# Cargar modelo (se descarga automáticamente la primera vez)
@st.cache_resource
def load_model():
    # URL de tu modelo en Hugging Face o Google Drive
    # Opción A: Si lo subes a Hugging Face Hub
    model_url = "https://huggingface.co/spaces/EAMB2001/DetectorPlagas/tree/main"
    
    # Descargar modelo
    response = requests.get(model_url)
    model_path = "/tmp/modelo.pt"
    with open(model_path, "wb") as f:
        f.write(response.content)
    
    return YOLO(model_path)

try:
    model = load_model()
    CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']
except:
    st.error("Error cargando el modelo")
    model = None

# Subir imagen
uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original", use_column_width=True)
    
    if st.button("🔍 Analizar"):
        with st.spinner("Analizando imagen..."):
            # Guardar temporal
            image = Image.open(uploaded_file)
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
            # Predecir
            results = model(temp_path, verbose=False)
            boxes = results[0].boxes
            
            with col2:
                if len(boxes) > 0:
                    # Mejor detección
                    mejor = max(boxes, key=lambda b: float(b.conf[0]))
                    clase = CLASSES[int(mejor.cls[0])]
                    conf = float(mejor.conf[0]) * 100
                    
                    st.success(f"✅ **{clase}**")
                    st.metric("Confianza", f"{conf:.2f}%")
                    
                    # Imagen con detecciones
                    result_img = results[0].plot()
                    st.image(result_img, caption="Resultado", use_column_width=True)
                    
                    # Todas las detecciones
                    st.write("### 📊 Todas las detecciones:")
                    for box in boxes:
                        cls = CLASSES[int(box.cls[0])]
                        confidence = float(box.conf[0]) * 100
                        st.write(f"• **{cls}**: {confidence:.2f}%")
                else:
                    st.warning("No se detectó ninguna hoja")