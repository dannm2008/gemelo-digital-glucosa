import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import kagglehub

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="MetabolicTwin AI - D1NAMO Real Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILOS CSS (MODO OSCURO PREMIUM Y ALTO CONTRASTE) ---
st.markdown("""
    <style>
    /* Fondo general oscuro de alto contraste */
    .stApp {
        background-color: #080B11 !important;
        color: #F3F4F6 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Tarjetas contenedoras oscuras e integradas */
    .dark-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Subtítulos y textos de soporte en gris de alto contraste */
    .support-text {
        color: #9CA3AF !important;
        font-size: 0.9rem;
    }

    /* Inputs de formulario perfectamente legibles en modo oscuro */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    
    label {
        color: #E5E7EB !important;
        font-weight: 500 !important;
    }

    /* Botón de Enviar con gradiente premium de alta visibilidad */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        width: 100% !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: #3B82F6 !important;
        box-shadow: 0px 0px 15px rgba(59, 130, 246, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DESCARGA Y CACHÉ DEL DATASET REAL D1NAMO ---
@st.cache_resource
def descargar_datos_d1namo():
    """Descarga el dataset de Kaggle una sola vez y lo guarda en caché."""
    try:
        ruta_cache = kagglehub.dataset_download("sarabhian/d1namo-ecg-glucose-data")
        return ruta_cache
    except Exception as e:
        st.error(f"Error al descargar dataset: {e}")
        return None

# --- CARGA Y PREPROCESAMIENTO DINÁMICO POR PACIENTE ---
def cargar_datos_paciente(ruta_cache, id_paciente):
    """Carga y procesa los datos de glucosa reales de un paciente específico."""
    ruta_csv = os.path.join(ruta_cache, 'healthy_subset_pictures-glucose-food', 'healthy_subset_pictures-glucose-food', id_paciente, 'glucose.csv')
    if os.path.exists(ruta_csv):
        df_p = pd.read_csv(ruta_csv)
        # Convertir a mg/dL (fisiología estándar)
        df_p['glucose_mgdl'] = df_p['glucose'] * 18.016
        # Minutos transcurridos en el día
        df_p['minutos_dia'] = pd.to_datetime(df_p['time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df_p['time'], format='%H:%M').dt.minute

        # Asegurar todas las columnas de comidas (One-Hot Encoding estable)
        comidas_posibles = ['AB', 'AD', 'AL', 'BB', 'BD', 'BL', 'M']
        for comida in comidas_posibles:
            df_p[f'type_{comida}'] = (df_p['type'] == comida).astype(int)

        columnas_X = ['minutos_dia'] + [f'type_{c}' for c in comidas_posibles]
        return df_p[columnas_X], df_p['glucose_mgdl'], df_p
    return None, None, None

# --- ALGORITMO MATEMÁTICO: CLASIFICACIÓN DE ZONAS DE CLARKE ---
def clasificar_zonas_clarke(y_true, y_pred):
    """Calcula matemáticamente el porcentaje de acierto en cada zona de la Rejilla de Clarke."""
    total = len(y_true)
    zona_A = 0
    zona_B = 0
    zona_CDE = 0
    
    for r, p in zip(y_true, y_pred):
        # Zona A: Margen de diferencia menor al 20% o ambos en hipoglucemia (<70)
        if (abs(r - p) <= 0.20 * r) or (r < 70 and p < 70):
            zona_A += 1
        # Zona C, D, E aproximadas clínicamente (errores críticos o peligrosos)
        elif (r > 180 and p < 70) or (r < 70 and p > 180):
            zona_CDE += 1
        else:
            zona_B += 1
            
    pct_A = (zona_A / total) * 100 if total > 0 else 0
    pct_B = (zona_B / total) * 100 if total > 0 else 0
    pct_CDE = (zona_CDE / total) * 100 if total > 0 else 0
    return pct_A, pct_B, pct_CDE

# --- GRÁFICA REAL DE LA REJILLA DE CLARKE ---
def plot_clarke_error_grid(ref, pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    # Líneas divisorias de la Rejilla de Clarke
    ax.plot([0, 400], [0, 400], ':', color='#475569')
    ax.plot([0, 175/3], [70, 70], '-', color='#475569', alpha=0.6)
    ax.plot([70, 70], [0, 175/3], '-', color='#475569', alpha=0.6)
    ax.plot([70, 400], [56, 320], '-', color='#475569', alpha=0.6)
    ax.plot([180, 400], [70, 70], '-', color='#475569', alpha=0.6)
    ax.plot([70, 400], [84, 480], '-', color='#475569', alpha=0.6)
    ax.plot([240, 240], [180, 400], '-', color='#475569', alpha=0.6)
    ax.plot([0, 70], [180, 180], '-', color='#475569', alpha=0.6)
    ax.plot([0, 180], [70, 250], '-', color='#475569', alpha=0.6)
    ax.plot([240, 400], [70, 70], '-', color='#475569', alpha=0.6)
    ax.plot([80, 400], [0, 160], '-', color='#475569', alpha=0.6)
    
    # Etiquetas de zonas clínicas con colores de semáforo
    ax.text(30, 35, 'A', fontsize=13, fontweight='bold', color='#10B981')
    ax.text(30, 140, 'B', fontsize=13, fontweight='bold', color='#F59E0B')
    ax.text(350, 260, 'B', fontsize=13, fontweight='bold', color='#F59E0B')
    ax.text(350, 140, 'C', fontsize=13, fontweight='bold', color='#EF4444')
    ax.text(150, 350, 'D', fontsize=13, fontweight='bold', color='#EF4444')
    ax.text(30, 350, 'E', fontsize=13, fontweight='bold', color='#B91C1C')
    ax.text(160, 20, 'C', fontsize=13, fontweight='bold', color='#EF4444')
    ax.text(260, 120, 'D', fontsize=13, fontweight='bold', color='#EF4444')

    # Ploteo de puntos REALES de evaluación
    ax.scatter(ref, pred, color='#3B82F6', alpha=0.8, edgecolors='#1D4ED8', s=25, label='Lecturas de Glucosa')
    
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#9CA3AF', labelsize=8)
    ax.set_xlabel("Referencia Real (mg/dL)", color='#9CA3AF', fontsize=9)
    ax.set_ylabel("Predicción del Gemelo (mg/dL)", color='#9CA3AF', fontsize=9)
    ax.legend(facecolor='#0F172A', edgecolor='none', labelcolor='#FFFFFF', loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.1)
    
    return fig

# --- INICIALIZAR EL SISTEMA ---
ruta_dataset = descargar_datos_d1namo()

if "encuesta_completada" not in st.session_state:
    st.session_state.encuesta_completada = False

# --- HEADER DE LA PLATAFORMA ---
st.markdown("""
    <div style='margin-bottom: 25px;'>
        <h1 style='color: #FFFFFF; font-size: 2.2rem; margin-bottom: 5px; font-weight: 700;'>MetabolicTwin AI</h1>
        <p class='support-text'>Monitoreo clínico predictivo de glucosa entrenado en tiempo real con el Dataset Clínico D1NAMO.</p>
    </div>
    <hr style='border: 0; border-top: 1px solid #1E293B; margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# --- PASO 1: LA ENCUESTA INTEGRADA ---
if not st.session_state.encuesta_completada:
    st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Formulario de Inicialización y Calibración del Gemelo")
    st.markdown("<p class='support-text'>Seleccione el perfil fisiológico real del paciente para entrenar el algoritmo del gemelo digital.</p>", unsafe_allow_html=True)
    
    # Selección de Paciente Real del Dataset
    paciente_elegido = st.selectbox(
        "Seleccione Paciente Real para Entrenar la IA:",
        ["Paciente 005 (Estudio de Referencia)", "Paciente 001", "Paciente 002", "Paciente 003"]
    )
    
    # Mapeo del ID
    id_mapeado = "005" if "005" in paciente_elegido else paciente_elegido.split(" ")[1]
    
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad del Paciente (Años):", min_value=1, max_value=100, value=35)
    with c2:
        peso = st.number_input("Peso de Registro (kg):", min_value=30.0, max_value=150.0, value=74.2)
        
    comida_activa = st.selectbox(
        "Momento de Intervención / Estado Alimentario Reciente:",
        [
            "AL (Almuerzo - Antes de Comer)", 
            "AB (Almuerzo - Después de Comer)", 
            "BL (Desayuno - Antes de Comer)", 
            "BB (Desayuno - Después de Comer)", 
            "BD (Cena - Antes de Comer)", 
            "AD (Cena - Después de Comer)", 
            "M (Merienda / Snack)"
        ]
    )
    
    antecedentes = st.text_area("Notas Clínicas y Diagnóstico Médico previo:", value="Paciente activo en protocolo observacional D1NAMO, sin anomalías cardiovasculares severas.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Conectar Dataset Real y Ejecutar Gemelo Digital ➔"):
        if ruta_dataset:
            with st.spinner("Descargando y extrayendo el historial metabólico del paciente..."):
                X, y, df_crudo = cargar_datos_paciente(ruta_dataset, id_mapeado)
                
                if X is not None and len(X) > 0:
                    # Guardar variables en sesión para utilizarlas en la visualización
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.df_crudo = df_crudo
                    st.session_state.paciente_info = {
                        "id": id_mapeado,
                        "edad": edad,
                        "peso": peso,
                        "comida": comida_activa.split(" ")[0]
                    }
                    st.session_state.encuesta_completada = True
                    st.rerun()
                else:
                    st.error("No se pudieron cargar los datos de este paciente. Por favor, intente con otro.")
        else:
            st.error("No se pudo conectar con el repositorio de Kaggle.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- PASO 2: ANÁLISIS CLÍNICO REAL DESBLOQUEADO ---
else:
    info = st.session_state.paciente_info
    X = st.session_state.X
    y = st.session_state.y
    df_crudo = st.session_state.df_crudo
    
    # Botón para volver a calibrar/cambiar paciente
    if st.button("⬅️ Cambiar de Paciente / Recalibrar"):
        st.session_state.encuesta_completada = False
        st.rerun()
        
    st.markdown(f"## 📊 Análisis Predictivo: Paciente {info['id']}")
    st.markdown(f"<p class='support-text'>Métricas Clínicas de Evaluación • {info['edad']} años • {info['peso']} kg • Calibración de comida: {info['comida']}</p>", unsafe_allow_html=True)
    
    # --- ENTRENAMIENTO REAL DEL MODELO RANDOM FOREST EN EL ACTO ---
    with st.spinner("Entrenando Modelo Random Forest en tiempo real..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        modelo_rf = RandomForestRegressor(n_estimators=50, random_state=42)
        modelo_rf.fit(X_train, y_train)
        
        # Predicciones de evaluación reales
        pred_rf = modelo_rf.predict(X_test)
        mae_real = mean_absolute_error(y_test, pred_rf)
        
        # Calcular porcentaje en las zonas de Clarke verdaderas
        pct_A, pct_B, pct_CDE = clasificar_zonas_clarke(y_test.values, pred_rf)

    col_izq, col_der = st.columns([1.7, 1.3])
    
    with col_izq:
        # 1. Gráfica temporal del set de prueba real (No simulada)
        st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
        st.markdown("### Trayectoria Temporal de Glucosa (Fisiología Real vs Predicción)")
        
        fig_temp, ax_temp = plt.subplots(figsize=(10, 4.5))
        fig_temp.patch.set_facecolor('#0F172A')
        ax_temp.set_facecolor('#0F172A')
        
        # Ordenamos las muestras cronológicamente según los minutos del día para graficar continuo
        indices_ordenados = np.argsort(X_test['minutos_dia'].values)
        tiempo_test_ordenado = X_test['minutos_dia'].values[indices_ordenados] / 60.0 # Convertir a horas
        y_test_ordenado = y_test.values[indices_ordenados]
        pred_ordenada = pred_rf[indices_ordenados]
        
        ax_temp.plot(tiempo_test_ordenado, y_test_ordenado, color='#94A3B8', alpha=0.5, label="Glucosa Real Registrada", linewidth=1.5, marker='o', markersize=3)
        ax_temp.plot(tiempo_test_ordenado, pred_ordenada, color='#3B82F6', label="Predicción de la IA", linewidth=2.5)
        ax_temp.axhspan(70, 140, color='#10B981', alpha=0.08, label="Rango Óptimo Clínico")
        
        ax_temp.set_xlim(0, 24)
        ax_temp.set_ylim(40, 220)
        ax_temp.spines['top'].set_visible(False)
        ax_temp.spines['right'].set_visible(False)
        ax_temp.spines['left'].set_color('#334155')
        ax_temp.spines['bottom'].set_color('#334155')
        ax_temp.tick_params(colors='#9CA3AF', labelsize=8)
        ax_temp.set_xlabel("Hora del Día (Formato 24h)", color='#9CA3AF', fontsize=9)
        ax_temp.set_ylabel("Glucosa en Sangre (mg/dL)", color='#9CA3AF', fontsize=9)
        ax_temp.legend(facecolor='#1E293B', edgecolor='none', labelcolor='#FFFFFF', loc='upper right', fontsize=8)
        ax_temp.grid(True, linestyle=':', alpha=0.1)
        
        st.pyplot(fig_temp)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. Rejilla de Clarke Real
        st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
        st.markdown("### Rejilla de Error de Clarke (Dinámica con sus Datos)")
        fig_clarke = plot_clarke_error_grid(y_test.values, pred_rf)
        st.pyplot(fig_clarke)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_der:
        # Diagnóstico y Precisión Real
        st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Desempeño y Error Clínico")
        
        st.markdown(f"""
            <div style='margin-bottom: 20px;'>
                <span class='support-text'>Error Medio del Modelo (MAE Real)</span>
                <h2 style='color:#EF4444; margin:0; font-size:2.4rem;'>{mae_real:.2f} mg/dL</h2>
                <span style='background-color:rgba(59, 130, 246, 0.15); color:#60A5FA; font-size:0.8rem; padding:4px 8px; border-radius:4px; display:inline-block; margin-top:5px;'>
                    Modelo: Random Forest (50 estimadores)
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Alertas contextuales basadas en el paciente real
        glucosa_promedio_paciente = np.mean(y_test)
        if glucosa_promedio_paciente > 130:
            alerta_texto = "El paciente presenta picos de hiperglucemia recurrentes en este dataset. Se aconseja regular las cargas de carbohidratos."
            alerta_color = "#F59E0B"
        else:
            alerta_texto = "Curva metabólica dentro de márgenes saludables estándar. Buen control glucémico basal."
            alerta_color = "#10B981"
            
        st.markdown(f"""
            <div style='border-left: 4px solid {alerta_color}; padding-left: 15px;'>
                <strong style='color:{alerta_color};'>Diagnóstico de Tendencia Glucémica</strong><br>
                <span class='support-text' style='font-size:0.8rem;'>{alerta_texto}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Metadatos Dinámicos de la Rejilla de Clarke (Calculados con matemáticas reales)
        st.markdown(f"""
            <div class='dark-card' style='background-color:#1E293B;'>
                <h4 style='color:#38BDF8; margin-top:0; font-weight:600;'>🎯 Seguridad de Clarke Real</h4>
                <p class='support-text' style='font-size:0.85rem;'>
                    Porcentajes reales calculados sobre el set de prueba de este paciente:
                </p>
                <div style='font-size:0.9rem; margin-top:10px; line-height:1.7;'>
                    <strong>• Zona A (Tratamiento Clínico Exacto):</strong> <span style='color:#10B981; font-weight:bold;'>{pct_A:.1f}%</span><br>
                    <strong>• Zona B (Margen Seguro Benigno):</strong> <span style='color:#F59E0B; font-weight:bold;'>{pct_B:.1f}%</span><br>
                    <strong>• Zonas C-E (Acciones Erradas):</strong> <span style='color:#EF4444; font-weight:bold;'>{pct_CDE:.1f}%</span>
                </div>
                <div style='background-color:#0F172A; border-radius:4px; height:8px; width:100%; margin-top:15px;'>
                    <div style='background-color:#10B981; width:{pct_A:.0f}%; height:100%; border-radius:4px; float:left;'></div>
                    <div style='background-color:#F59E0B; width:{pct_B:.0f}%; height:100%; float:left;'></div>
                    <div style='background-color:#EF4444; width:{pct_CDE:.0f}%; height:100%; border-radius:4px; float:left;'></div>
                </div>
                <span class='support-text' style='font-size:0.8rem; margin-top:8px; display:block;'>
                    {pct_A + pct_B:.1f}% de decisiones clínicamente viables.
                </span>
            </div>
        """, unsafe_allow_html=True)
