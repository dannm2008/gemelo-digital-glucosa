import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import kagglehub
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="MetabolicTwin AI - Gemelo Digital", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (Azul Oscuro, Azul Claro, Gris y Accesibilidad) ---
# Inicializar estado de accesibilidad (Tamaño de letra)
if "fuente_grande" not in st.session_state:
    st.session_state.fuente_grande = False

# Ajustar tamaño de letra según el botón de accesibilidad
factor_fuente = "1.15rem" if st.session_state.fuente_grande else "0.95rem"
factor_titulo = "2.2rem" if st.session_state.fuente_grande else "1.8rem"

st.markdown(f"""
    <style>
    /* Estilo General y Fondos */
    .stApp {{
        background-color: #F8FAFC;
        font-size: {factor_fuente};
    }}
    
    /* Barra Lateral Estilo Nodo Clínico */
    [data-testid="stSidebar"] {{
        background-color: #EBF2FA !important;
        border-right: 1px solid #D0E1F9;
    }}
    
    /* Títulos y Subtítulos */
    h1, h2, h3 {{
        color: #0A2540 !important;
        font-family: 'Inter', sans-serif;
    }}
    h1 {{ font-size: {factor_titulo} !important; }}
    
    /* Tarjetas Clínicas Estilo Imagen */
    .clinical-card {{
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }}
    
    /* Alertas de Riesgo */
    .alert-critical {{
        background-color: #FFF5F5;
        border-left: 5px solid #E53E3E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }}
    .alert-low {{
        background-color: #FFFDF5;
        border-left: 5px solid #D69E2E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_stdio=True)

# --- TRADUCCIÓN DE ESTADOS METABÓLICOS ---
traduccion_comidas = {
    "AB": "Después del Desayuno (Post-prandial)",
    "AD": "Después de la Cena (Post-prandial)",
    "AL": "Después del Almuerzo (Post-prandial)",
    "BB": "Antes del Desayuno (Ayunas)",
    "BD": "Antes de la Cena",
    "BL": "Antes del Almuerzo"
}

# --- CARGA DEL MODELO EN SEGUNDO PLANO ---
@st.cache_resource
def inicializar_modelo_y_datos():
    path = kagglehub.dataset_download("paultimothymooney/d1namo-dataset")
    glucose_path = os.path.join(path, "diabetes_subset", "diabetes_subset", "005", "glucose.csv")
    df = pd.read_csv(glucose_path)
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    X = df[['hour', 'minute']]
    X = pd.concat([X, pd.get_dummies(df['type'])], axis=1)
    y = df['glucose']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df, X.columns

with st.spinner("Conectando con el Nodo Clínico Seguro..."):
    model, df_hist, X_columns = inicializar_modelo_y_datos()

# --- CONTROL DE SESIÓN ---
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = "Admisión"
if 'paciente_datos' not in st.session_state:
    st.session_state.paciente_datos = None

# --- BARRA LATERAL (SIDEBAR DE CONTROL) ---
with st.sidebar:
    st.markdown("### Nodo Clínico PAC-005")
    st.markdown("---")
    
    # Navegación del Menú Estilo Imagen
    if st.button("Admisión de Pacientes", use_container_width=True):
        st.session_state.pagina_actual = "Admisión"
        st.rerun()
        
    if st.button("Gemelo en Tiempo Real", use_container_width=True):
        if st.session_state.paciente_datos is None:
            st.warning("Atención: Primero debes registrar un paciente en la sección de Admisión.")
        else:
            st.session_state.pagina_actual = "Gemelo"
            st.rerun()
            
    if st.button("¿Cómo funciona?", use_container_width=True):
        st.session_state.pagina_actual = "Ayuda"
        st.rerun()
        
    st.markdown("---")
    # Botón de Accesibilidad (Tamaño de texto)
    texto_btn_acc = "Modo Lectura Normal" if st.session_state.fuente_grande else "Modo Lectura Grande"
    if st.button(texto_btn_acc, use_container_width=True):
        st.session_state.fuente_grande = not st.session_state.fuente_grande
        st.rerun()

# --- PÁGINA 1: FORMULARIO DE ADMISIÓN (Estilo Clínico) ---
if st.session_state.pagina_actual == "Admisión":
    st.title("MetabolicTwin AI")
    st.markdown("#### Admisión del Paciente — Nodo Clínico")
    st.caption("Complete la información fisiológica para inicializar la simulación del Gemelo Digital.")
    
    # Contenedor del formulario centrado
    with st.container():
        st.markdown("<div class='clinical-card'>", unsafe_allow_stdio=True)
        with st.form("form_clinico"):
            nombre = st.text_input("Nombre Completo del Paciente:", placeholder="Ej. Juan Pérez")
            
            c1, c2 = st.columns(2)
            with c1:
                id_historial = st.text_input("ID de Historial Clínico:", value="PAC-005", disabled=True)
                edad = st.number_input("Edad (Años):", min_value=1, max_value=110, value=35)
            with c2:
                peso = st.number_input("Peso Corporal Actual (kg):", min_value=10.0, max_value=250.0, value=75.0)
                
            antecedentes = st.text_area("Antecedentes Médicos y Notas Clínicas:", placeholder="Diabetes Tipo 1, medicamentos activos, alergias...")
            
            enviar = st.form_submit_button("Generar Gemelo Digital", use_container_width=True)
            
            if enviar:
                if nombre.strip() == "":
                    st.error("Por favor, ingrese el nombre para continuar.")
                else:
                    st.session_state.paciente_datos = {
                        "nombre": nombre,
                        "edad": edad,
                        "peso": peso,
                        "antecedentes": antecedentes
                    }
                    st.success("¡Datos guardados! Cargando tablero metabólico...")
                    st.session_state.pagina_actual = "Gemelo"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_stdio=True)

# --- PÁGINA 2: DASHBOARD DEL GEMELO DIGITAL ---
elif st.session_state.pagina_actual == "Gemelo":
    paciente = st.session_state.paciente_datos
    st.title("MetabolicTwin AI — Dashboard")
    st.markdown(f"**Paciente:** {paciente['nombre']} | **ID:** PAC-005 | **Edad:** {paciente['edad']} años | **Peso:** {paciente['peso']} kg")
    st.caption("Protocolo de Control Glucémico Predictivo v2.4")
    st.markdown("---")
    
    # Distribución en columnas: Izquierda (Interactiva) - Derecha (Alertas de la IA)
    col_izq, col_der = st.columns([2.2, 1])
    
    with col_izq:
        st.markdown("<div class='clinical-card'>", unsafe_allow_stdio=True)
        st.markdown("### Panel de Control y Simulación")
        
        c1, c2 = st.columns(2)
        with c1:
            hora_sim = st.time_input("Seleccionar Hora del Día para Evaluar:", value=df_hist['date'].iloc[0].time())
        with c2:
            opcion_comida = st.selectbox(
                "Estado Fisiológico / Última Comida:",
                options=list(traduccion_comidas.keys()),
                format_func=lambda x: traduccion_comidas[x]
            )
            
        simular = st.button("Ejecutar Simulación en Gemelo Digital", use_container_width=True)
        st.markdown("</div>", unsafe_allow_stdio=True)
        
        # Cálculos de predicción al simular
        h, m = hora_sim.hour, hora_sim.minute
        input_data = pd.DataFrame([[h, m]], columns=['hour', 'minute'])
        for col in X_columns:
            if col not in ['hour', 'minute']:
                input_data[col] = 1 if col == opcion_comida else 0
        
        glucosa_predicha = model.predict(input_data)[0]
        
        # Bloque de Gráfica de Tendencia
        st.markdown("<div class='clinical-card'>", unsafe_allow_stdio=True)
        st.markdown("### Trayectoria de Glucosa Predictiva (Siguientes Horas)")
        
        horas_futuras = [(h + i) % 24 for i in range(6)]
        predicciones_futuras = []
        for hf in horas_futuras:
            input_f = pd.DataFrame([[hf, m]], columns=['hour', 'minute'])
            for col in X_columns:
                if col not in ['hour', 'minute']:
                    input_f[col] = 1 if col == opcion_comida else 0
            predicciones_futuras.append(model.predict(input_f)[0])
            
        # Graficando con estilo limpio y profesional
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([f"{hf:02d}:00" for hf in horas_futuras], predicciones_futuras, marker='o', color='#00509d', linewidth=2.5, label="Predicción IA")
        ax.axhspan(70, 140, color='#E6F4EA', alpha=0.4, label="Rango Óptimo (70-140 mg/dL)")
        ax.set_ylim(40, 200)
        ax.set_ylabel("Glucosa (mg/dL)", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_stdio=True)

    with col_der:
        # Panel de Lectura Actual
        st.markdown("<div class='clinical-card'>", unsafe_allow_stdio=True)
        st.markdown("#### Lectura Estimada de la IA")
        st.markdown(f"<h1 style='font-size: 3rem; color: #00509d; margin:0;'>{glucosa_predicha:.1f} <span style='font-size: 1.2rem; color: #64748B;'>mg/dL</span></h1>", unsafe_allow_stdio=True)
        st.markdown("</div>", unsafe_allow_stdio=True)
        
        # Panel de Evaluación de Riesgos Estilo Imagen
        st.markdown("<div class='clinical-card'>", unsafe_allow_stdio=True)
        st.markdown("### Evaluación de Riesgos")
        
        if glucosa_predicha < 70:
            st.markdown(f"""
                <div class='alert-critical'>
                    <strong style='color:#C53030;'>Hipoglucemia Crítica Detectada</strong><br>
                    La predicción estimada es de {glucosa_predicha:.1f} mg/dL. Se recomienda el consumo inmediato de carbohidratos de absorción rápida.
                </div>
            """, unsafe_allow_stdio=True)
        elif glucosa_predicha > 140:
            st.markdown(f"""
                <div class='alert-low'>
                    <strong style='color:#B7791F;'>Tendencia a Hiperglucemia</strong><br>
                    La lectura ({glucosa_predicha:.1f} mg/dL) supera el rango óptimo post-prandial. Vigilar hidratación y dosis correctoras.
                </div>
            """, unsafe_allow_stdio=True)
        else:
            st.markdown(f"""
                <div style='background-color: #F0FDF4; border-left: 5px solid #38A169; padding: 15px; border-radius: 8px;'>
                    <strong style='color:#2F855A;'>Sistema Estable</strong><br>
                    El Gemelo Digital predice rangos metabólicos seguros y equilibrados.
                </div>
            """, unsafe_allow_stdio=True)
        st.markdown("</div>", unsafe_allow_stdio=True)

# --- PÁGINA 3: ¿CÓMO FUNCIONA? (HELP / MANUAL) ---
elif st.session_state.pagina_actual == "Ayuda":
    st.title("Centro de Ayuda e Información")
    st.markdown("#### ¿Cómo interpretar y utilizar esta Plataforma de Gemelo Digital?")
    st.markdown("---")
    
    st.markdown("""
    Este sistema médico asistido por Inteligencia Artificial te permite analizar simulaciones fisiológicas en tiempo real de forma muy sencilla:
    
    ### 1. Proceso de Admisión
    * En la pestaña **Admisión** se registran los parámetros demográficos y físicos del paciente.
    * Estos datos permiten contextualizar los informes clínicos generados.
    
    ### 2. Panel del Gemelo Digital (Predicción)
    * El sistema utiliza un modelo predictivo basado en algoritmos de **Bosques Aleatorios (Random Forest)**.
    * Al configurar una **Hora del día** y un **Estado Fisiológico** (por ejemplo, *Antes del Almuerzo*), el cerebro de la IA analiza la curva histórica de datos del paciente para proyectar con precisión cómo se comportará la glucosa en las horas siguientes.
    
    ### 3. Alertas de Riesgo
    * El sistema evalúa automáticamente las predicciones y clasifica las lecturas en tres zonas clínicas de seguridad:
        * **Rango Estable:** Entre 70 y 140 mg/dL.
        * **Hiperglucemia (Riesgo Alto):** Mayor a 140 mg/dL.
        * **Hipoglucemia (Riesgo Crítico):** Menor a 70 mg/dL.
    """)
    
    if st.button("Volver al Dashboard", use_container_width=True):
        st.session_state.pagina_actual = "Gemelo"
        st.rerun()
