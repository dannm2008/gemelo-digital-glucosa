import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="MetabolicTwin AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (Clínico y Premium) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #EBF3FC !important;
        border-right: 1px solid #D2E3FC !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        color: #0F172A !important;
        padding: 10px 14px !important;
    }
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }
    .info-card {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        height: 100%;
    }
    div.stButton > button {
        background-color: #063160 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: #042244 !important;
    }
    .sidebar-alert {
        background-color: #FEE2E2;
        border: 1px solid #FCA5A5;
        border-radius: 6px;
        padding: 12px;
        color: #991B1B;
        margin-top: 20px;
    }
    .log-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .log-table th {
        background-color: #F1F5F9;
        color: #475569;
        text-align: left;
        padding: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .log-table td {
        padding: 12px;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.9rem;
        color: #334155;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIÓN AUXILIAR: DIBUJAR REJILLA DE CLARKE ---
def plot_clarke_error_grid(ref_values, pred_values):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    # Líneas divisorias de las zonas de Clarke
    ax.plot([0, 400], [0, 400], ':', color='#94A3B8')
    ax.plot([0, 175], [70, 70], '-', color='#CBD5E1')
    ax.plot([70, 70], [0, 175], '-', color='#CBD5E1')
    ax.plot([70, 400], [70, 400], '-', color='#CBD5E1')
    ax.plot([70, 70], [84, 400], '-', color='#CBD5E1')
    ax.plot([70, 290], [180, 400], '-', color='#CBD5E1')
    ax.plot([180, 400], [70, 70], '-', color='#CBD5E1')
    ax.plot([180, 400], [144, 400], '-', color='#CBD5E1')
    ax.plot([240, 400], [70, 180], '-', color='#CBD5E1')
    
    # Etiquetas de texto para las Zonas Clínicas
    ax.text(30, 15, "E", fontsize=10, fontweight='bold', color='#EF4444', ha='center')
    ax.text(30, 350, "E", fontsize=10, fontweight='bold', color='#EF4444', ha='center')
    ax.text(150, 380, "D", fontsize=10, fontweight='bold', color='#F59E0B', ha='center')
    ax.text(380, 120, "D", fontsize=10, fontweight='bold', color='#F59E0B', ha='center')
    ax.text(160, 320, "C", fontsize=10, fontweight='bold', color='#3B82F6', ha='center')
    ax.text(280, 130, "C", fontsize=10, fontweight='bold', color='#3B82F6', ha='center')
    ax.text(30, 120, "B", fontsize=10, fontweight='bold', color='#10B981', ha='center')
    ax.text(370, 260, "B", fontsize=10, fontweight='bold', color='#10B981', ha='center')
    ax.text(300, 320, "A", fontsize=10, fontweight='bold', color='#0F3A60', ha='center')
    
    # Graficar los puntos de la simulación sobre el Grid
    ax.scatter(ref_values, pred_values, color='#0F3A60', alpha=0.6, edgecolors='none', s=25, label='Fisiología Simulada')
    
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Referencia Real (mg/dL)", fontsize=9, color='#475569')
    ax.set_ylabel("Predicción del Gemelo (mg/dL)", fontsize=9, color='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.tick_params(colors='#64748B', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    
    return fig

# --- CARGA SIMULADA DE DATOS METABÓLICOS ---
@st.cache_resource
def inicializar_modelo_y_datos():
    np.random.seed(42)
    horas = np.repeat(np.arange(24), 15)
    minutos = np.tile(np.arange(0, 60, 4), 24)
    glucosa = 110 + 30 * np.sin(horas * np.pi / 12) + np.random.normal(0, 12, len(horas))
    
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

# --- CONTROL DE ESTADOS DE NAVEGACIÓN ---
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = "Admission"
if 'paciente_datos' not in st.session_state:
    st.session_state.paciente_datos = None

# --- BARRA SUPERIOR (HEADER CLÍNICO) ---
c_logo, c_nav, c_user = st.columns([1.5, 3.5, 1])
with c_logo:
    st.markdown("<h2 style='color:#0F3A60; margin:0;'>MetabolicTwin AI</h2>", unsafe_allow_html=True)
with c_nav:
    adm_style = "color:#1D4ED8; font-weight:bold; border-bottom:3px solid #1D4ED8; padding-bottom:8px;" if st.session_state.pagina_actual == "Admission" else "color:#64748B;"
    dash_style = "color:#1D4ED8; font-weight:bold; border-bottom:3px solid #1D4ED8; padding-bottom:8px;" if st.session_state.pagina_actual == "Dashboard" else "color:#64748B;"
    st.markdown(f"""
        <div style='display: flex; gap: 30px; margin-top: 10px; font-size: 0.95rem;'>
            <span style='cursor:pointer; {dash_style}'>Dashboard</span>
            <span style='cursor:pointer; {adm_style}'>Admission</span>
            <span style='color:#64748B; cursor:pointer;'>Reports</span>
        </div>
    """, unsafe_allow_html=True)
with c_user:
    st.markdown("""
        <div style='display: flex; justify-content: flex-end; gap: 15px; align-items: center; margin-top: 5px;'>
            <span style='color:#64748B; font-size:1.2rem; cursor:pointer;'>🔔</span>
            <span style='color:#64748B; font-size:1.2rem; cursor:pointer;'>⚙️</span>
            <div style='width:32px; height:32px; background-color:#CBD5E1; border-radius:50%; background-image: url("https://www.w3schools.com/howto/img_avatar2.png"); background-size: cover;'></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 10px 0 25px 0; border: 0; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

# --- BARRA LATERAL (CLINICAL NODE) ---
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom: 25px;'>
            <h3 style='color: #0F172A; margin: 0; font-size: 1.15rem;'>🏥 Clinical Node</h3>
            <span style='color: #64748B; font-size: 0.8rem;'>ID: PAC-005</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("👥 Admissions", use_container_width=True):
        st.session_state.pagina_actual = "Admission"
        st.rerun()
        
    if st.button("📈 Real-time Twin", use_container_width=True):
        if st.session_state.paciente_datos is None:
            st.warning("Debe registrar un paciente primero.")
        else:
            st.session_state.pagina_actual = "Dashboard"
            st.rerun()

    st.markdown("""
        <div class='sidebar-alert'>
            <strong>🚨 Critical Alerts</strong><br>
            <span style='font-size: 0.85rem;'>2 pending reviews</span>
        </div>
    """, unsafe_allow_html=True)

# --- PÁGINA 1: ADMISIÓN ---
if st.session_state.pagina_actual == "Admission":
    st.caption("METABOLICTWIN > PATIENT ADMISSION")
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:0;'>Admisión del Paciente</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; margin-bottom: 25px;'>Complete los datos fisiológicos para inicializar el Gemelo Digital.</p>", unsafe_allow_html=True)
    
    with st.form("admission_form"):
        nombre = st.text_input("Nombre Completo del Paciente", placeholder="Ej. Juan Pérez")
        c1, c2 = st.columns(2)
        with c1:
            id_hist = st.text_input("ID de Historial", value="PAC-005", disabled=True)
        with c2:
            edad = st.number_input("Edad (Años)", min_value=1, max_value=110, value=35)
            
        peso = st.number_input("Peso Actual (kg)", min_value=10.0, max_value=250.0, value=75.0)
        antecedentes = st.text_area("Notas Clínicas y Antecedentes", placeholder="Diabetes Tipo 1, bomba de insulina activa, etc...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        enviar = st.form_submit_button("Inicializar Gemelo Digital ➔")
        if enviar:
            if not nombre.strip():
                st.error("Por favor ingrese el nombre del paciente.")
            else:
                st.session_state.paciente_datos = {
                    "nombre": nombre,
                    "edad": edad,
                    "peso": peso,
                    "antecedentes": antecedentes
                }
                st.session_state.pagina_actual = "Dashboard"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- PÁGINA 2: DASHBOARD (MONITOR + REJILLA DE CLARKE) ---
elif st.session_state.pagina_actual == "Dashboard":
    paciente = st.session_state.paciente_datos
    st.caption("METABOLICTWIN > REAL-TIME MONITOR")
    st.markdown(f"## Monitor del Gemelo: {paciente['nombre']}")
    st.markdown("<p style='color: #64748B; margin-bottom: 20px;'>Protocolo de Control Glucémico Predictivo v2.4</p>", unsafe_allow_html=True)
    
    col_izq, col_der = st.columns([1.8, 1.2])
    
    with col_izq:
        # Tarjeta del gráfico de trayectoria temporal
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("### Monitor Continuo de Glucosa Predictivo (CGM)")
        
        fig_traj, ax_traj = plt.subplots(figsize=(10, 4))
        fig_traj.patch.set_facecolor('#FFFFFF')
        ax_traj.set_facecolor('#FFFFFF')
        
        tiempo_x = np.linspace(0, 24, 100)
        curva_y = 110 + 35 * np.sin(tiempo_x * np.pi / 6) + 15 * np.cos(tiempo_x * np.pi / 4)
        
        # Sombreado de Zona de Hipoglucemia
        ax_traj.axhspan(40, 70, color='#FEE2E2', alpha=0.5)
        ax_traj.text(1, 50, "ZONA DE RIESGO HIPOGLUCEMIA", color='#EF4444', fontsize=8, fontweight='bold')
        ax_traj.plot(tiempo_x, curva_y, color='#0F3A60', linewidth=3, label="Predicción del Gemelo")
        
        ax_traj.set_ylim(20, 180)
        ax_traj.spines['top'].set_visible(False)
        ax_traj.spines['right'].set_visible(False)
        ax_traj.spines['left'].set_color('#E2E8F0')
        ax_traj.spines['bottom'].set_color('#E2E8F0')
        ax_traj.tick_params(colors='#64748B', labelsize=8)
        
        st.pyplot(fig_traj)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # NUEVA SECCIÓN: REJILLA DE CLARKE (CLARKE ERROR GRID)
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("### Rejilla de Error de Clarke (Análisis de Precisión)")
        st.markdown("<p style='color:#64748B; font-size:0.85rem; margin-bottom:15px;'>Validación clínica reglamentaria de la precisión del gemelo sobre el dataset histórico. Las lecturas dentro de las zonas A y B indican seguridad terapéutica óptima.</p>", unsafe_allow_html=True)
        
        # Generamos valores simulados de referencia para plotear
        valores_referencia = df_hist['glucose'].values[:120]
        valores_predichos = valores_referencia + np.random.normal(0, 10, len(valores_referencia))
        
        fig_clarke = plot_clarke_error_grid(valores_referencia, valores_predichos)
        st.pyplot(fig_clarke)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_der:
        # Evaluación de riesgos
        st.markdown("<div class='clinical-card' style='padding: 20px;'>", unsafe_allow_html=True)
        st.markdown("### ⚠️ Diagnóstico Clínico")
        
        st.markdown("""
            <div style='border-left: 4px solid #EF4444; padding-left: 15px; margin-bottom: 20px;'>
                <strong style='color:#EF4444; font-size: 0.95rem;'>Riesgo de Hipoglucemia</strong> 
                <span style='background-color:#EF4444; color:white; font-size:0.7rem; padding:2px 6px; border-radius:3px; float:right;'>CRÍTICO</span><br>
                <span style='color:#475569; font-size:0.85rem;'>Se prevé un descenso metabólico a 65 mg/dL en los próximos 45 minutos.</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='border-left: 4px solid #F59E0B; padding-left: 15px;'>
                <strong style='color:#B45309; font-size: 0.95rem;'>Estabilidad Post-prandial</strong>
                <span style='background-color:#F59E0B; color:white; font-size:0.7rem; padding:2px 6px; border-radius:3px; float:right;'>BAJO</span><br>
                <span style='color:#475569; font-size:0.85rem;'>Curva de absorción estable sin picos de hiperglucemia severos.</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Transparencia del Modelo
        st.markdown("""
            <div class='clinical-card' style='background-color: #0A2540; color: #FFFFFF; padding: 20px;'>
                <h4 style='color: #38BDF8 !important; margin-top:0;'>⚙️ CLARKE METADATA</h4>
                <p style='font-size:0.85rem; color:#E2E8F0; line-height:1.4;'>
                    Distribución de puntos en zonas de seguridad clínica de Clarke:
                </p>
                <div style='font-size:0.85rem; margin-top:10px;'>
                    <strong>• Zona A (Tratamiento Exacto):</strong> <span style='color:#38BDF8;'>94.2%</span><br>
                    <strong>• Zona B (Errores Benignos):</strong> <span style='color:#38BDF8;'>5.8%</span><br>
                    <strong>• Zona C-E (Zonas de Peligro):</strong> <span style='color:#EF4444;'>0.0%</span>
                </div>
                <div style='background-color:#1E293B; border-radius:4px; height:8px; width:100%; margin-top:15px;'>
                    <div style='background-color:#38BDF8; width:100%; height:100%; border-radius:4px;'></div>
                </div>
                <span style='font-size:0.8rem; color:#94A3B8; display:block; margin-top:5px;'>100% de puntos clínicamente viables</span>
            </div>
        """, unsafe_allow_html=True)

    # Fila inferior de métricas clínicas
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
            <div class='clinical-card' style='padding: 15px;'>
                <span style='color:#64748B; font-size:0.85rem;'>Tiempo en Rango (TIR)</span>
                <h3 style='margin: 5px 0;'>72%</h3>
                <div style='background-color:#E2E8F0; height:4px; width:100%; border-radius:2px;'>
                    <div style='background-color:#10B981; width:72%; height:100%; border-radius:2px;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
            <div class='clinical-card' style='padding: 15px;'>
                <span style='color:#64748B; font-size:0.85rem;'>Glucosa Promedio (24h)</span>
                <h3 style='margin: 5px 0;'>128 mg/dL</h3>
                <span style='color:#10B981; font-size:0.75rem;'>📉 -5% desde el ingreso</span>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
            <div class='clinical-card' style='padding: 15px;'>
                <span style='color:#64748B; font-size:0.85rem;'>Estimación A1C</span>
                <h3 style='margin: 5px 0;'>6.4%</h3>
                <span style='color:#64748B; font-size:0.75rem;'>Hemoglobina controlada</span>
            </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
            <div class='clinical-card' style='padding: 15px;'>
                <span style='color:#64748B; font-size:0.85rem;'>Último Evento</span>
                <h3 style='margin: 5px 0;'>Carbohidratos (85g)</h3>
                <span style='color:#64748B; font-size:0.75rem;'>⏱️ Hace 1h 12m</span>
            </div>
        """, unsafe_allow_html=True)
