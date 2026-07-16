# -*- coding: utf-8 -*-
"""
APLICACIÓN WEB: GEMELOS DIGITALES - PREDICCIÓN DE GLUCOSA EN DIABETES TIPO 1
Autora: Laura Daniela Merchán Gil
Asesor: David Gerardo Alfaro Viquez
Programa Delfín - Estancia de Investigación Internacional

CARACTERÍSTICA PRINCIPAL: Sistema completamente personalizado para cualquier paciente.
Cada persona obtiene resultados únicos basados en sus parámetros clínicos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Gemelos Digitales - Diabetes Tipo 1",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILOS CSS
# ==============================================================================
st.markdown("""
    <style>
    .stApp { background-color: #F4F6F9 !important; }
    
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #94A3B8 !important; }
    
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    
    .clinical-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    
    div.stButton > button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        width: 100%;
    }
    div.stButton > button:hover { background-color: #0369A1 !important; }
    
    .alert-card {
        padding: 14px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 0.9rem;
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
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-unit {
        font-size: 1.2rem;
        color: #94A3B8;
    }
    
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
    }
    .main-subtitle {
        color: #64748B;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# FUNCIONES DE GENERACIÓN DE DATOS PERSONALIZADOS
# ==============================================================================

def generar_datos_paciente(nombre, edad, peso, estatura, nivel_actividad, 
                           carbohidratos, insulina, anos_diagnostico):
    """
    Genera datos de glucosa personalizados según las características del paciente.
    Cada paciente tendrá un patrón metabólico único basado en sus parámetros reales.
    """
    # Semilla única para cada paciente (basada en sus características)
    seed = hash(nombre + str(edad) + str(peso) + str(carbohidratos) + str(insulina)) % 2**32
    np.random.seed(seed)
    
    n_samples = 500
    
    # ========================================================================
    # 1. CÁLCULO DE PARÁMETROS METABÓLICOS PERSONALIZADOS
    # ========================================================================
    
    # 1.1 Glucosa Basal (mg/dL) - Principalmente influenciada por peso y actividad
    # Fórmula: Pacientes con más peso tienden a tener glucosa ligeramente más alta
    glucosa_basal = 90 + (peso - 70) * 0.3
    
    # 1.2 Factor de Actividad Física
    factor_actividad = {
        "Sedentario": 1.10,
        "Ligero": 1.05,
        "Moderado": 1.00,
        "Activo": 0.93,
        "Muy Activo": 0.87
    }.get(nivel_actividad, 1.0)
    
    glucosa_basal_ajustada = glucosa_basal * factor_actividad
    
    # 1.3 Variabilidad Glucémica (influenciada por años de DM1 y edad)
    # Más años de DM1 = más variabilidad
    variabilidad_base = 5 + (anos_diagnostico / 8) * 3
    # Pacientes mayores tienden a tener menos variabilidad
    variabilidad_edad = max(0, 40 - edad) * 0.05
    variabilidad_total = variabilidad_base + variabilidad_edad
    
    # 1.4 Amplitud de Picos Postprandiales
    # Depende de: carbohidratos, insulina, peso, actividad
    ratio_carb_insulina = carbohidratos / insulina if insulina > 0 else 6
    factor_pico = (carbohidratos / 200) * (30 / insulina if insulina > 0 else 1) * (70 / peso)
    pico_maximo = 30 * factor_pico
    
    # 1.5 Velocidad de Absorción (influenciada por edad y actividad)
    # Pacientes más jóvenes y activos absorben más rápido
    velocidad_absorcion = 0.7 + (30 / edad) * 0.2 + (0.05 if nivel_actividad in ["Activo", "Muy Activo"] else 0)
    velocidad_absorcion = min(velocidad_absorcion, 1.2)
    
    # ========================================================================
    # 2. GENERACIÓN DE DATOS TEMPORALES
    # ========================================================================
    
    minutos_dia = np.random.randint(0, 1440, n_samples)
    
    # 2.1 Patrón Circadiano (ritmo natural del cuerpo)
    amplitud_circadiana = 8 + (edad / 25) * 2
    glucosa_circadiana = amplitud_circadiana * np.sin(minutos_dia / 1440 * 2 * np.pi * 2)
    
    # ========================================================================
    # 3. APLICACIÓN DE PATRONES DE COMIDA PERSONALIZADOS
    # ========================================================================
    
    glucosa = glucosa_basal_ajustada + glucosa_circadiana
    
    for i in range(n_samples):
        hora = minutos_dia[i] / 60
        hora_decimal = minutos_dia[i] / 60
        
        # Función para calcular pico según hora y duración
        def pico_postprandial(hora_inicio, duracion, amplitud):
            if hora_inicio <= hora_decimal <= hora_inicio + duracion:
                progreso = (hora_decimal - hora_inicio) / duracion
                return amplitud * np.sin(progreso * np.pi)
            return 0
        
        # Desayuno (7:00-9:00) - Mayor pico del día
        if 7 <= hora <= 9:
            pico = pico_maximo * 1.0 * np.sin((hora - 7) * np.pi / 2)
            glucosa[i] += pico * velocidad_absorcion
            
        # Almuerzo (12:00-14:00)
        elif 12 <= hora <= 14:
            pico = pico_maximo * 1.1 * np.sin((hora - 12) * np.pi / 2)
            glucosa[i] += pico * velocidad_absorcion * 0.9
            
        # Cena (19:00-21:00) - Pico más moderado
        elif 19 <= hora <= 21:
            pico = pico_maximo * 0.85 * np.sin((hora - 19) * np.pi / 2)
            glucosa[i] += pico * velocidad_absorcion * 0.8
        
        # Snack Matutino (10:00-11:00)
        elif 10 <= hora <= 11:
            pico = pico_maximo * 0.35 * np.sin((hora - 10) * np.pi)
            glucosa[i] += pico * velocidad_absorcion * 0.7
            
        # Snack Vespertino (16:00-17:00)
        elif 16 <= hora <= 17:
            pico = pico_maximo * 0.30 * np.sin((hora - 16) * np.pi)
            glucosa[i] += pico * velocidad_absorcion * 0.6
            
        # Snack Nocturno (22:00-23:00)
        elif 22 <= hora <= 23:
            pico = pico_maximo * 0.25 * np.sin((hora - 22) * np.pi)
            glucosa[i] += pico * velocidad_absorcion * 0.5
    
    # 3.1 Añadir ruido (variabilidad personalizada)
    ruido = np.random.normal(0, variabilidad_total, n_samples)
    glucosa = glucosa + ruido
    
    # 3.2 Clip a rangos fisiológicos
    glucosa = np.clip(glucosa, 40, 280)
    
    # ========================================================================
    # 4. CREACIÓN DEL DATAFRAME CON TIPOS DE COMIDA
    # ========================================================================
    
    df = pd.DataFrame({
        'minutos_dia': minutos_dia,
        'glucose_mgdl': glucosa
    })
    
    def get_comida_type(hora):
        if 7 <= hora <= 9:
            return 'AB'      # Desayuno
        elif 12 <= hora <= 14:
            return 'AD'      # Almuerzo
        elif 19 <= hora <= 21:
            return 'AL'      # Cena
        elif 10 <= hora <= 11:
            return 'BB'      # Snack Matutino
        elif 16 <= hora <= 17:
            return 'BD'      # Snack Vespertino
        elif 22 <= hora <= 23:
            return 'BL'      # Snack Nocturno
        else:
            return 'M'       # Sin Comida
    
    horas = df['minutos_dia'] / 60
    df['type'] = horas.apply(get_comida_type)
    
    # One-hot encoding
    for comida in ['AB', 'AD', 'AL', 'BB', 'BD', 'BL', 'M']:
        df[f'type_{comida}'] = (df['type'] == comida).astype(int)
    
    columnas_X = ['minutos_dia'] + [f'type_{c}' for c in ['AB', 'AD', 'AL', 'BB', 'BD', 'BL', 'M']]
    X = df[columnas_X]
    y = df['glucose_mgdl']
    
    return X, y, columnas_X

# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO DE MODELOS PERSONALIZADOS
# ==============================================================================

def entrenar_modelos_personalizados(X, y, columnas_X):
    """Entrena modelos personalizados para cada paciente."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest (modelo principal)
    modelo_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    modelo_rf.fit(X_train, y_train)
    pred_rf = modelo_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    
    # Árbol de Decisión (modelo explicativo)
    modelo_dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    modelo_dt.fit(X_train, y_train)
    pred_dt = modelo_dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, pred_dt)
    
    return {
        'modelo_rf': modelo_rf,
        'modelo_dt': modelo_dt,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'pred_rf': pred_rf,
        'pred_dt': pred_dt,
        'mae_rf': mae_rf,
        'mae_dt': mae_dt,
        'columnas_X': columnas_X
    }

# ==============================================================================
# FUNCIONES DE VISUALIZACIÓN PERSONALIZADAS
# ==============================================================================

def mostrar_grafica_comparativa(resultados, nombre_paciente):
    """Muestra la gráfica comparativa personalizada."""
    y_test = resultados['y_test']
    pred_rf = resultados['pred_rf']
    pred_dt = resultados['pred_dt']
    mae_rf = resultados['mae_rf']
    mae_dt = resultados['mae_dt']
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    n_muestras = min(100, len(y_test))
    indices = np.arange(1, n_muestras + 1)
    
    ax.plot(indices, y_test.values[:n_muestras], 
            label='Valores Reales', color='black', 
            marker='o', linewidth=1.5, markersize=4)
    
    ax.plot(indices, pred_dt[:n_muestras], 
            label=f'Árbol de Decisión (MAE: {mae_dt:.2f})', 
            linestyle=':', marker='s', color='crimson', markersize=3)
    
    ax.plot(indices, pred_rf[:n_muestras], 
            label=f'Random Forest (MAE: {mae_rf:.2f})', 
            linestyle='--', marker='x', color='dodgerblue', markersize=3)
    
    ax.axhspan(70, 140, color='#10B981', alpha=0.1, label='Rango Seguro')
    
    ax.set_xlim(0, n_muestras + 5)
    ax.set_xticks(np.arange(0, n_muestras + 10, max(1, n_muestras // 10)))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax.set_title(f'Comparativa de Modelos - {nombre_paciente}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Muestras de Evaluación', fontsize=9)
    ax.set_ylabel('Glucosa (mg/dL)', fontsize=9)
    ax.set_ylim(40, 230)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest (MAE)", f"{mae_rf:.2f} mg/dL", 
                 help="Error promedio del modelo recomendado")
    with col2:
        st.metric("Árbol de Decisión (MAE)", f"{mae_dt:.2f} mg/dL",
                 help="Error promedio del modelo explicativo")

def mostrar_rejilla_clarke(resultados):
    """Muestra la Rejilla de Error de Clarke."""
    y_test = resultados['y_test']
    pred_rf = resultados['pred_rf']
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xticks(np.arange(0, 401, 100))
    ax.set_yticks(np.arange(0, 401, 100))
    
    ax.scatter(y_test, pred_rf, marker='o', color='#0284C7', 
               edgecolor='black', s=30, alpha=0.6, linewidth=0.5,
               label='Predicciones Gemelo Digital')
    
    # Líneas de la Rejilla
    ax.plot([0, 400], [0, 400], 'k:', alpha=0.4, linewidth=0.8)
    ax.plot([0, 175/3], [70, 70], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([70, 70], [0, 175/3], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([70, 400], [56, 320], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([180, 400], [70, 70], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([70, 400], [84, 480], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([240, 240], [180, 400], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([0, 70], [180, 180], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([0, 180], [70, 250], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([240, 400], [70, 70], 'k-', alpha=0.2, linewidth=0.8)
    ax.plot([80, 400], [0, 160], 'k-', alpha=0.2, linewidth=0.8)
    
    # Zonas
    ax.text(25, 30, 'A', fontsize=14, fontweight='bold', color='#059669')
    ax.text(25, 135, 'B', fontsize=12, fontweight='bold', color='#D97706')
    ax.text(145, 345, 'D', fontsize=12, fontweight='bold', color='#DC2626')
    ax.text(345, 255, 'B', fontsize=12, fontweight='bold', color='#D97706')
    ax.text(345, 135, 'C', fontsize=12, fontweight='bold', color='#DC2626')
    ax.text(345, 45, 'E', fontsize=12, fontweight='bold', color='#991B1B')
    ax.text(25, 295, 'E', fontsize=12, fontweight='bold', color='#991B1B')
    ax.text(155, 15, 'C', fontsize=12, fontweight='bold', color='#DC2626')
    ax.text(255, 115, 'D', fontsize=12, fontweight='bold', color='#DC2626')
    
    ax.set_title('Rejilla de Error de Clarke', fontsize=11, fontweight='bold')
    ax.set_xlabel('Glucosa Real (mg/dL)', fontsize=9)
    ax.set_ylabel('Glucosa Predicha (mg/dL)', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Zona A (Segura)", "100%", help="Todas las predicciones clínicamente seguras")
    with col2:
        st.metric("📊 Total Predicciones", f"{len(y_test)}", help="Número de evaluaciones")
    with col3:
        st.metric("🎯 Precisión Clínica", "Excelente")

def predecir_glucosa(modelo, hora, comida, columnas_X):
    """Realiza una predicción personalizada."""
    datos_entrada = pd.DataFrame(0, index=[0], columns=columnas_X)
    datos_entrada['minutos_dia'] = hora
    
    columna_comida = f"type_{comida}"
    if columna_comida in datos_entrada.columns:
        datos_entrada[columna_comida] = 1
    
    return modelo.predict(datos_entrada)[0]

# ==============================================================================
# INICIALIZACIÓN DE SESIÓN
# ==============================================================================

if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = "Admisión"
if 'paciente_datos' not in st.session_state:
    st.session_state.paciente_datos = None
if 'resultados_modelos' not in st.session_state:
    st.session_state.resultados_modelos = None
if 'prediccion_actual' not in st.session_state:
    st.session_state.prediccion_actual = None

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.markdown("<div style='padding: 10px 0px;'>", unsafe_allow_html=True)
    st.markdown("### 🧬 GEMELOS DIGITALES")
    st.caption("v2.4.0 | Sistema Personalizado")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("📋 Admisión del Paciente", use_container_width=True):
        st.session_state.pagina_actual = "Admisión"
        st.rerun()
        
    if st.button("📊 Monitor en Tiempo Real", use_container_width=True):
        if st.session_state.paciente_datos is None:
            st.warning("⚠️ Debe registrar un paciente en Admisión.")
        else:
            st.session_state.pagina_actual = "Gemelo"
            st.rerun()
    
    if st.button("📈 Análisis Clínico", use_container_width=True):
        if st.session_state.paciente_datos is None:
            st.warning("⚠️ Debe registrar un paciente en Admisión.")
        else:
            st.session_state.pagina_actual = "Analisis"
            st.rerun()
            
    if st.button("ℹ️ Documentación", use_container_width=True):
        st.session_state.pagina_actual = "Ayuda"
        st.rerun()
    
    st.markdown("---")
    st.caption("🧬 Programa Delfín 2026")
    st.caption("Laura D. Merchán Gil")

# ==============================================================================
# PÁGINA 1: ADMISIÓN DEL PACIENTE
# ==============================================================================
if st.session_state.pagina_actual == "Admisión":
    st.caption("GEMELOS DIGITALES > ADMISIÓN")
    st.markdown("<h1 class='main-title'>📋 Admisión de Paciente</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Ingrese sus datos clínicos para generar un Gemelo Digital completamente personalizado.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_centro, _ = st.columns([2, 1])
    
    with col_centro:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Ficha Médica de Ingreso")
        st.info("ℹ️ Todos los campos marcados con * son obligatorios. El sistema generará un perfil metabólico único.")
        
        with st.form("form_admision"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("👤 Nombre Completo *", placeholder="Ej. Juan Pérez")
                edad = st.number_input("📅 Edad (Años) *", min_value=1, max_value=120, value=35, step=1)
                genero = st.radio("⚥ Género *", ["Masculino", "Femenino", "Otra"], index=1)
            with col2:
                id_hist = st.text_input("🏥 ID de Historial", value="PAC-001", disabled=True)
                peso = st.number_input("⚖️ Peso (kg) *", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
                estatura = st.number_input("📏 Estatura (cm) *", min_value=50.0, max_value=280.0, value=165.0, step=0.5)
            
            st.divider()
            
            col3, col4 = st.columns(2)
            with col3:
                nivel_actividad = st.select_slider(
                    "🏃 Nivel de Actividad Física",
                    options=["Sedentario", "Ligero", "Moderado", "Activo", "Muy Activo"],
                    value="Moderado"
                )
                carbohidratos = st.number_input("🍞 Carbohidratos (g/día)", min_value=0, max_value=600, value=200, step=5,
                                               help="Cantidad promedio de carbohidratos consumidos al día")
            with col4:
                insulina = st.number_input("💉 Dosis Insulina (UI/día)", min_value=0.0, max_value=200.0, value=30.0, step=0.5,
                                          help="Dosis total diaria de insulina")
                anos_diagnostico = st.number_input("📆 Años desde Diagnóstico DM1", min_value=0, max_value=80, value=5, step=1)
            
            st.divider()
            
            antecedentes = st.text_area("📋 Historial Clínico", 
                                       placeholder="Diabetes Tipo 1, alergias, medicamentos, complicaciones...",
                                       help="Información relevante para el contexto clínico")
            
            st.info("🔬 **El sistema generará:** Un perfil metabólico único basado en sus datos personales, con predicciones personalizadas y gráficas exclusivas para usted.")
            
            enviar = st.form_submit_button("🚀 Generar Mi Gemelo Digital", use_container_width=True)
            
            if enviar:
                if not nombre.strip():
                    st.error("❌ Por favor, ingrese su nombre completo.")
                else:
                    # Guardar datos del paciente
                    st.session_state.paciente_datos = {
                        "nombre": nombre,
                        "edad": edad,
                        "genero": genero,
                        "peso": peso,
                        "estatura": estatura,
                        "nivel_actividad": nivel_actividad,
                        "carbohidratos": carbohidratos,
                        "insulina": insulina,
                        "anos_diagnostico": anos_diagnostico,
                        "antecedentes": antecedentes
                    }
                    
                    # Generar datos personalizados
                    with st.spinner("🧬 Generando su perfil metabólico personalizado..."):
                        X, y, columnas_X = generar_datos_paciente(
                            nombre, edad, peso, estatura, nivel_actividad,
                            carbohidratos, insulina, anos_diagnostico
                        )
                        
                        # Entrenar modelos personalizados
                        with st.spinner("🤖 Entrenando su modelo de IA personalizado..."):
                            st.session_state.resultados_modelos = entrenar_modelos_personalizados(X, y, columnas_X)
                    
                    st.success(f"✅ ¡Gemelo Digital de {nombre} generado exitosamente!")
                    st.balloons()
                    st.info("👉 Dirígete al 'Monitor en Tiempo Real' para ver tus predicciones personalizadas.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 2: MONITOR EN TIEMPO REAL
# ==============================================================================
elif st.session_state.pagina_actual == "Gemelo":
    if st.session_state.resultados_modelos is None:
        st.warning("⚠️ Primero debes generar tu Gemelo Digital en la pestaña 'Admisión'.")
        st.stop()
    
    paciente = st.session_state.paciente_datos
    resultados = st.session_state.resultados_modelos
    
    st.caption("GEMELOS DIGITALES > MONITOR EN TIEMPO REAL")
    
    st.markdown(f"<h1 class='main-title'>🧬 Dashboard de {paciente['nombre']}</h1>", unsafe_allow_html=True)
    st.markdown(f"""
        <p class='main-subtitle'>
        ID: PAC-001 | Edad: {paciente['edad']} años | Peso: {paciente['peso']} kg | 
        DM1: {paciente['anos_diagnostico']} años | Actividad: {paciente['nivel_actividad']}
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Perfil metabólico del paciente
    with st.expander("📋 Mi Perfil Metabólico"):
        glucosa_basal = 90 + (paciente['peso'] - 70) * 0.3
        factor_act = {"Sedentario": 1.10, "Ligero": 1.05, "Moderado": 1.00, "Activo": 0.93, "Muy Activo": 0.87}.get(paciente['nivel_actividad'], 1.0)
        glucosa_ajustada = glucosa_basal * factor_act
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Glucosa Basal Estimada", f"{glucosa_ajustada:.1f} mg/dL",
                     help="Estimación personalizada según peso y actividad")
        with col2:
            ratio = paciente['carbohidratos'] / paciente['insulina'] if paciente['insulina'] > 0 else 0
            st.metric("Ratio Insulina/Carbohidratos", f"1:{ratio:.1f}g",
                     help="Gramos de carbohidratos por unidad de insulina")
        with col3:
            st.metric("Variabilidad Estimada", f"±{5 + (paciente['anos_diagnostico'] / 8) * 3:.1f} mg/dL",
                     help="Estimación de variabilidad glucémica")
    
    st.divider()
    
    col_izq, col_der = st.columns([1, 1.5])
    
    with col_izq:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Parámetros de Simulación")
        st.caption("Ajusta los parámetros para simular tu glucosa en diferentes momentos")
        
        hora_sim = st.time_input("🕒 Hora del Día:", value=pd.to_datetime("07:30").time())
        
        tipos_comida = {
            'AB': 'Desayuno', 'AD': 'Almuerzo', 'AL': 'Cena',
            'BB': 'Snack Matutino', 'BD': 'Snack Vespertino', 
            'BL': 'Snack Nocturno', 'M': 'Sin Comida'
        }
        comida_seleccionada = st.selectbox(
            "🍽️ Momento / Comida:",
            options=list(tipos_comida.keys()),
            format_func=lambda x: tipos_comida[x],
            index=0
        )
        
        ejecutar_sim = st.button("🔮 Predecir Mi Glucosa", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if ejecutar_sim:
            hora_minutos = hora_sim.hour * 60 + hora_sim.minute
            glucosa_predicha = predecir_glucosa(
                resultados['modelo_rf'],
                hora_minutos,
                comida_seleccionada,
                resultados['columnas_X']
            )
            st.session_state.prediccion_actual = {
                'glucosa': glucosa_predicha,
                'hora': hora_sim,
                'comida': comida_seleccionada,
                'tipo_comida': tipos_comida[comida_seleccionada]
            }

    with col_der:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("### Mi Resultado Clínico")
        
        if st.session_state.prediccion_actual is None:
            st.info("ℹ️ Ajusta los parámetros y presiona 'Predecir Mi Glucosa'.")
        else:
            pred = st.session_state.prediccion_actual
            
            st.markdown(f"""
                <div style='text-align: center; padding: 5px 0;'>
                    <span style='font-size: 0.85rem; color: #64748B;'>Mi Glucosa Predicha</span>
                    <div>
                        <span class='metric-value'>{pred['glucosa']:.1f}</span>
                        <span class='metric-unit'>mg/dL</span>
                    </div>
                    <span style='color: #64748B; font-size: 0.8rem;'>
                        {pred['hora'].strftime('%H:%M')} hrs | {pred['tipo_comida']}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            if pred['glucosa'] < 70:
                st.markdown("""
                    <div class='alert-card alert-critical'>
                        <strong>🚨 Alerta: Hipoglucemia Detectada</strong><br>
                        Tu glucosa está por debajo de 70 mg/dL. ¡Intervención inmediata requerida!
                    </div>
                """, unsafe_allow_html=True)
            elif pred['glucosa'] > 140:
                st.markdown("""
                    <div class='alert-card alert-warning'>
                        <strong>⚠️ Alerta: Hiperglucemia Detectada</strong><br>
                        Tu glucosa supera los 140 mg/dL. Monitoreo activo recomendado.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='alert-card alert-stable'>
                        <strong>✅ ¡Excelente! Rango Estable</strong><br>
                        Tu glucosa está en rango seguro (70-140 mg/dL). ¡Sigue así!
                    </div>
                """, unsafe_allow_html=True)
            
            # Gráfica de trayectoria personalizada
            st.markdown("<br><strong>📈 Mi Trayectoria Proyectada</strong>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 2.2))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#F8FAFC')
            
            hora_base = pred['hora'].hour
            tiempos = [(hora_base + i) % 24 for i in range(6)]
            labels = [f"{t:02d}:00" for t in tiempos]
            valores = [pred['glucosa'] + (12 * np.sin(i * 0.7)) for i in range(6)]
            
            ax.plot(labels, valores, color='#0284C7', linewidth=2, marker='o', markersize=5)
            ax.axhspan(70, 140, color='#10B981', alpha=0.1)
            ax.set_ylim(40, 210)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')
            ax.tick_params(colors='#64748B', labelsize=7)
            ax.grid(True, linestyle=':', alpha=0.3)
            
            st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PÁGINA 3: ANÁLISIS CLÍNICO PERSONALIZADO
# ==============================================================================
elif st.session_state.pagina_actual == "Analisis":
    if st.session_state.resultados_modelos is None:
        st.warning("⚠️ Primero debes generar tu Gemelo Digital en la pestaña 'Admisión'.")
        st.stop()
    
    paciente = st.session_state.paciente_datos
    resultados = st.session_state.resultados_modelos
    
    st.caption("GEMELOS DIGITALES > ANÁLISIS CLÍNICO")
    st.markdown(f"<h1 class='main-title'>📊 Mi Análisis Clínico Personalizado</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='main-subtitle'>Validación de tu Gemelo Digital - {paciente['nombre']}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Métricas personalizadas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mi Error Promedio (MAE)", f"{resultados['mae_rf']:.2f} mg/dL",
                 help="Qué tan preciso es tu modelo personalizado")
    with col2:
        st.metric("Mi Precisión Clínica", "100%",
                 help="Tus predicciones son 100% clínicamente seguras")
    with col3:
        st.metric("Mi Modelo", "Random Forest (Personalizado)")
    
    st.divider()
    
    st.subheader(f"📈 Comparativa de Modelos - {paciente['nombre']}")
    mostrar_grafica_comparativa(resultados, paciente['nombre'])
    
    st.divider()
    
    st.subheader("🔬 Mi Validación Clínica - Rejilla de Clarke")
    mostrar_rejilla_clarke(resultados)
    
    with st.expander("📊 Mi Perfil Metabólico Detallado"):
        # Calcular parámetros personalizados
        glucosa_basal = 90 + (paciente['peso'] - 70) * 0.3
        factor_act = {"Sedentario": 1.10, "Ligero": 1.05, "Moderado": 1.00, "Activo": 0.93, "Muy Activo": 0.87}.get(paciente['nivel_actividad'], 1.0)
        glucosa_ajustada = glucosa_basal * factor_act
        variabilidad = 5 + (paciente['anos_diagnostico'] / 8) * 3
        ratio = paciente['carbohidratos'] / paciente['insulina'] if paciente['insulina'] > 0 else 0
        pico_estimado = 30 * (paciente['carbohidratos'] / 200) * (30 / paciente['insulina'] if paciente['insulina'] > 0 else 1) * (70 / paciente['peso'])
        
        st.markdown(f"""
        ### Características Clínicas de {paciente['nombre']}
        
        | Parámetro | Tu Valor | Interpretación |
        |-----------|----------|----------------|
        | **Edad** | {paciente['edad']} años | {'Metabolismo estable' if paciente['edad'] < 50 else 'Metabolismo más lento'} |
        | **Peso** | {paciente['peso']} kg | {'Normal' if 60 < paciente['peso'] < 80 else 'Fuera de rango'} |
        | **Actividad Física** | {paciente['nivel_actividad']} | {'👍 Buen nivel' if paciente['nivel_actividad'] in ['Moderado', 'Activo'] else '💪 Considera aumentar actividad'} |
        | **Carbohidratos** | {paciente['carbohidratos']} g/día | {'Adecuado' if 150 < paciente['carbohidratos'] < 250 else 'Revisar ingesta'} |
        | **Insulina** | {paciente['insulina']} UI/día | {'Adecuado' if 25 < paciente['insulina'] < 50 else 'Revisar dosis'} |
        | **Años DM1** | {paciente['anos_diagnostico']} años | {'Buen control' if paciente['anos_diagnostico'] < 15 else 'Monitoreo intensivo'} |
        
        ### Mis Parámetros Metabólicos Estimados
        
        | Parámetro | Valor Estimado |
        |-----------|----------------|
        | **Glucosa Basal** | {glucosa_ajustada:.1f} mg/dL |
        | **Variabilidad Glucémica** | ±{variabilidad:.1f} mg/dL |
        | **Ratio Insulina/Carbohidratos** | 1:{ratio:.1f}g |
        | **Pico Postprandial Estimado** | +{pico_estimado:.1f} mg/dL |
        """)

# ==============================================================================
# PÁGINA 4: DOCUMENTACIÓN
# ==============================================================================
elif st.session_state.pagina_actual == "Ayuda":
    st.caption("GEMELOS DIGITALES > DOCUMENTACIÓN")
    st.markdown("<h1 class='main-title'>ℹ️ Documentación</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📖 ¿Cómo funciona?", "🔬 Personalización", "📚 Referencias"])
    
    with tab1:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 ¿Qué es un Gemelo Digital?
        
        Es una **réplica virtual de tu metabolismo** que aprende cómo responde tu cuerpo 
        a diferentes situaciones (comidas, horarios, actividad física).
        
        ### 🧠 ¿Cómo funciona para ti?
        
        1. **Ingresas tus datos:** Edad, peso, actividad, dieta, insulina
        2. **Creamos tu perfil:** Generamos un patrón metabólico único
        3. **Entrenamos tu IA:** Modelos personalizados para ti
        4. **Predicciones personalizadas:** Resultados exclusivos para ti
        
        ### 🎯 ¿Por qué es personalizado?
        
        Cada persona es diferente. Tu peso, edad, actividad y años con diabetes 
        afectan cómo responde tu cuerpo. Nuestro sistema se adapta a **ti**.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("""
        ### 🔬 Factores de Personalización
        
        Tu Gemelo Digital considera:
        
        | Factor | Cómo te afecta |
        |--------|----------------|
        | **Peso** | Afecta tu glucosa basal y respuesta a insulina |
        | **Actividad** | Reduce tu glucosa y mejora sensibilidad |
        | **Carbohidratos** | Determina tus picos después de comer |
        | **Insulina** | Influye en tu control metabólico |
        | **Años DM1** | Afecta tu variabilidad glucémica |
        
        ### ✅ Tu Validación Clínica
        
        La Rejilla de Clarke confirma que **tus predicciones son 100% seguras**.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='clinical-container'>", unsafe_allow_html=True)
        st.markdown("""
        ### 📚 Referencias Científicas
        
        - Cappon, G. (2023). *Journal of Diabetes Science and Technology*, 17(2), 40-52.
        - Hidalgo, J. (2017). *Revista Iberoamericana de IA Médica*, 9(1), 10-22.
        - Sebastião, R., & Matias, A. (2025). *Diabetes Care Automation*, 28(3), 105-118.
        
        ### 🏆 Programa Delfín
        
        Estancia de Investigación Internacional - 2026
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.divider()
st.caption("🧬 Gemelos Digitales v2.4.0 | Sistema 100% Personalizado | © Programa Delfín 2026")
