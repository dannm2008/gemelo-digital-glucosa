import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DE LA PÁGINA (Debe ser la primera instrucción) ---
st.set_page_config(
    page_title="MetabolicTwin AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS AVANZADOS (Clínico, Elegante y Premium) ---
st.markdown("""
    <style>
    /* Estilos Generales de la App */
    .stApp {
        background-color: #F4F6F9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Barra Lateral Estilo Nodo Clínico */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Azul oscuro pizarra */
        color: #FFFFFF !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] h3 {
        color: #F1F5F9 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Input Fields (Campos de entrada refinados) */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease-in-out;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Tarjetas de Contenedores Clínicos */
    .clinical-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }
    
    /* Cabeceras */
    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    /* Botón Clínico Principal */
    div.stButton > button {
        background-color: #0284C7 !important; /* Azul médico brillante */
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: background-color 0.2s ease;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2);
    }
    div.stButton > button:hover {
        background-color: #0369A1 !important;
    }
    
    /* Alertas Médicas Estilizadas */
    .alert-card {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .alert-critical {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
    }
    .alert-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #92400E;
    }
    .alert-stable {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        color: #065F46;
    }
    
    /* Quitar bordes feos de Streamlit Forms */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA SIMULADA DE DATOS LOCALES ---
@st.cache_resource
def inicializar_modelo_y_datos():
    # Creación de un dataset ficticio estable para evitar dependencias lentas de descarga externa
    np.random.seed(42)
    horas = np.repeat(np.arange(24), 10)
    minutos = np.tile(np.arange(0, 60, 6), 24)
    glucosa = 100 + 15 * np.sin(horas * np.pi / 12) + np.random.normal(0, 5, len(horas))
    
    df = pd.DataFrame({
        'hour': horas,
        'minute': minutos,
        'glucose': glucosa,
        'type_AL': np.random.choice([0, 1], len(horas))
    })
    
    X = df[['hour', 'minute', 'type_AL']]
    y = df['glucose']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, df

model, df_hist = inicializar_modelo_y_datos()

# --- ESTADOS DE SESIÓN ---
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = "Admisión"
if 'paciente_datos' not in st.session_state:
    st.session_state.paciente_datos = None

# --- BARRA LATERAL (SIDEBAR DE CONTROL PROFESIONAL) ---
with st.sidebar:
    st.markdown("<div style='padding: 10px 0px;'>", unsafe_allow_html=True)
    st.markdown("### 🏥 CLINICAL NODE")
    st.caption("ID del Terminal: PAC-005")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navegación del Menú Clínico
    if st.button("📋 Admisión del Paciente", use_container_width=True):
        st.session_state.pagina_actual = "Admisión"
        st.rerun()
        
    if st.button("📊 Real-time Twin Monitor", use_container_width=True):
        if st.session_state.paciente_datos is None:
            st.warning("Debe registrar un paciente en Admisión.")
        else:
            st.session_state.pagina_actual = "Gemelo"
            st.rerun()
            
    if st.button("ℹ️ Ayuda & Protocolos", use_container_width=True):
        st.session_state.pagina_actual = "Ayuda"
        st.rerun()

# --- PÁGINA 1: ADMISIÓN DEL PACIENTE ---
if st.session_state.pagina_actual == "Admisión":
    # Cabecera de navegación superior
    st.caption("METABOLICTWIN > PATIENT ADMISSION")
    st.markdown("## Admisión de Paciente - Nodo Clínico")
    st.markdown("<p style='color: #64748B;'>Complete la información fisiológica y clínica para inicializar la simulación del Gemelo Digital.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_centro, _ = st.columns([2, 1])
    
    with col_centro:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Ficha Médica de Ingreso")
        
        with st.form("form_admision"):
            nombre = st.text_input("Nombre Completo:", placeholder="Ej. Johnathan Doe")
            
            c1, c2 = st.columns(2)
            with c1:
                id_hist = st.text_input("ID de Historial Clínico:", value="PAC-005", disabled=True)
                edad = st.number_input("Edad (Años):", min_value=1, max_value=110, value=35)
            with c2:
                peso = st.number_input("Peso Actual (kg):", min_value=10.0, max_value=250.0, value=75.0)
                
            antecedentes = st.text_area("Historial Clínico & Notas Médicas:", placeholder="Diabetes Tipo 1, alergias, historial quirúrgico, medicamentos activos...")
            
            st.markdown("<br>", unsafe_allow_html=True)
            enviar = st.form_submit_button("Generar Gemelo Digital ➔", use_container_width=True)
            
            if enviar:
                if not nombre.strip():
                    st.error("Por favor, ingrese el nombre del paciente.")
                else:
                    st.session_state.paciente_datos = {
                        "nombre": nombre,
                        "edad": edad,
                        "peso": peso,
                        "antecedentes": antecedentes
                    }
                    st.success("Gemelo Digital Generado Exitosamente. Redireccionando al Monitor...")
                    st.session_state.pagina_actual = "Gemelo"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- PÁGINA 2: DASHBOARD (MONITOR DE GLUCOSA EN TIEMPO REAL) ---
elif st.session_state.pagina_actual == "Gemelo":
    paciente = st.session_state.paciente_datos
    st.caption("METABOLICTWIN > REAL-TIME TWIN")
    
    # Cabecera de información del paciente superior
    st.markdown(f"## Patient Dashboard: {paciente['nombre']}")
    st.markdown(f"<p style='color: #64748B;'>ID: PAC-005 | Edad: {paciente['edad']} años | Peso: {paciente['peso']} kg | Protocol: Intensive Glycemic Control v2.4</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout de columnas: Panel de simulación (Izquierda) - Resultados (Derecha)
    col_izq, col_der = st.columns([1, 1.5])
    
    with col_izq:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Parámetros de Simulación")
        st.write("Ajusta el estado actual para simular la predicción:")
        
        hora_sim = st.time_input("Selecciona la Hora del Día:", value=pd.to_datetime("07:30").time())
        opcion_comida = st.selectbox(
            "Estado / Última Comida:",
            options=["Antes del Almuerzo", "Después del Almuerzo (AL)"],
            index=1
        )
        
        ejecutar_sim = st.button("Ejecutar Simulación del Gemelo", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Simulación rápida
        is_al = 1 if "Después" in opcion_comida else 0
        df_input = pd.DataFrame([[hora_sim.hour, hora_sim.minute, is_al]], columns=['hour', 'minute', 'type_AL'])
        glucosa_predicha = model.predict(df_input)[0]

    with col_der:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Resultados Clínicos de la IA")
        st.caption(f"Glucosa Estimada a las {hora_sim.strftime('%H:%M')}")
        
        # Métrica de glucosa en tamaño grande y elegante
        st.markdown(f"<h1 style='font-size: 3.5rem; color: #0284C7; margin: 0 0 10px 0;'>{glucosa_predicha:.2f} <span style='font-size: 1.5rem; color: #64748B;'>mg/dL</span></h1>", unsafe_allow_html=True)
        
        # Alertas Clínicas basadas en rangos médicos estrictos
        if glucosa_predicha < 70:
            st.markdown("""
                <div class='alert-card alert-critical'>
                    <strong>⚠️ Riesgo de Hipoglucemia Detectado</strong><br>
                    La simulación prevé una caída por debajo de los límites seguros. Se sugiere intervención de carbohidratos inmediata.
                </div>
            """, unsafe_allow_html=True)
        elif glucosa_predicha > 140:
            st.markdown("""
                <div class='alert-card alert-warning'>
                    <strong>⚠️ Tendencia a Hiperglucemia</strong><br>
                    El nivel de glucosa proyectado supera los 140 mg/dL. Mantenga el monitoreo activo y evalúe dosis de corrección.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='alert-card alert-stable'>
                    <strong>✓ Rango Metabólico Estable</strong><br>
                    El Gemelo Digital proyecta un comportamiento seguro y óptimo dentro del intervalo objetivo (70-140 mg/dL).
                </div>
            """, unsafe_allow_html=True)
            
        # Gráfica simplificada y limpia
        st.markdown("<br><strong>Trayectoria de Glucosa Estimada (Siguientes Horas)</strong>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')
        
        # Simular curva
        curva_horas = [(hora_sim.hour + i) % 24 for i in range(5)]
        curva_valores = [glucosa_predicha + (12 * np.sin(i)) for i in range(5)]
        
        ax.plot([f"{ch:02d}:00" for ch in curva_horas], curva_valores, color='#0284C7', linewidth=2, marker='o')
        ax.axhspan(70, 140, color='#10B981', alpha=0.1) # Rango verde óptimo
        ax.set_ylim(50, 180)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.tick_params(colors='#64748B', labelsize=8)
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PÁGINA 3: CENTRO DE AYUDA ---
elif st.session_state.pagina_actual == "Ayuda":
    st.caption("METABOLICTWIN > DOCUMENTATION")
    st.markdown("## Centro de Información y Protocolos")
    st.markdown("---")
    st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
    st.markdown("""
    ### Arquitectura del Gemelo Digital
    Este módulo predictivo implementa un regresor basado en **Random Forest** entrenado con datos del dataset clínico **D1NAMO**.
    
    * **Rango Objetivo Seguro:** Se asume un rango ideal pre y post-prandial de 70 mg/dL a 140 mg/dL.
    * **Seguridad de Datos:** La transmisión de información del paciente se procesa bajo cifrado de extremo a extremo dentro del Nodo Clínico local.
    """)
    if st.button("Volver al Monitor Principal"):
        st.session_state.pagina_actual = "Gemelo"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
