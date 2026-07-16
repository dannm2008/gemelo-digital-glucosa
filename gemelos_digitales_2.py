import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import kagglehub
import shutil

# --- 1. CONFIGURACIÓN DE LA PLATAFORMA ---
st.set_page_config(
    page_title="Gemelo Digital - D1NAMO",
    layout="wide"
)

st.title("Plataforma del Gemelo Digital: Monitoreo de Glucosa (Dataset D1NAMO)")
st.markdown("Esta aplicación web ejecuta el modelo de IA entrenado para predecir y prevenir desequilibrios glucémicos.")
st.markdown("---")

# --- 2. ENTRENAMIENTO DEL MODELO EN SEGUNDO PLANO ---
# Usamos un decorador para que solo entrene la primera vez que se abre la app y no se vuelva lento
@st.cache_resource
def inicializar_modelo_y_datos():
    ruta_final = 'datos_d1namo/glucose.csv'
    
    if not os.path.exists(ruta_final):
        # Descarga rápida del dataset D1NAMO
        ruta_cache = kagglehub.dataset_download("sarabhian/d1namo-ecg-glucose-data")
        ruta_paciente_005 = os.path.join(ruta_cache, 'healthy_subset_pictures-glucose-food', 'healthy_subset_pictures-glucose-food', '005')
        archivo_origen = os.path.join(ruta_paciente_005, 'glucose.csv')
        os.makedirs('datos_d1namo', exist_ok=True)
        shutil.copy(archivo_origen, ruta_final)
        
    # Procesar Datos
    df = pd.read_csv(ruta_final)
    df['glucose_mgdl'] = df['glucose'] * 18.016
    df['minutos_dia'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df['time'], format='%H:%M').dt.minute
    df_ia = pd.get_dummies(df, columns=['type'], drop_first=False)
    
    columnas_predictoras = ['minutos_dia'] + [col for col in df_ia.columns if 'type_' in col]
    X = df_ia[columnas_predictoras]
    y = df_ia['glucose_mgdl']
    
    # Entrenar el Random Forest (Su mejor modelo en Colab)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    # Extraer la lista de comidas disponibles
    lista_comidas = [col.replace('type_', '') for col in columnas_predictoras if 'type_' in col]
    
    return modelo_rf, columnas_predictoras, lista_comidas

# Ejecutar la inicialización
with st.spinner("Cargando el cerebro metabólico del Gemelo Digital..."):
    modelo_rf, columnas_predictoras, lista_comidas = inicializar_modelo_y_datos()

# --- 3. DISEÑO DE LA INTERFAZ DE USUARIO (COLUMNAS) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Parámetros Clínicos")
    st.write("Ajusta el estado actual del Paciente 005 para simular la predicción:")
    
    # Formulario dinámico
    hora_simulada = st.time_input("Selecciona la Hora del Día:", value=pd.to_datetime("08:00").time())
    minutos_totales = hora_simulada.hour * 60 + hora_simulada.minute
    
    comida_seleccionada = st.selectbox("Estado / Última Comida:", lista_comidas)
    
    calcular = st.button("Ejecutar Simulación del Gemelo")

with col2:
    st.header("Resultados Clínicos de la IA")
    
    if calcular:
        # Construir el vector exacto con Pandas para el modelo
        datos_entrada = pd.DataFrame(0, index=[0], columns=columnas_predictoras)
        datos_entrada['minutos_dia'] = minutos_totales
        
        columna_comida_activa = f"type_{comida_seleccionada}"
        if columna_comida_activa in datos_entrada.columns:
            datos_entrada[columna_comida_activa] = 1
            
        # Predicción real utilizando el modelo de su Colab
        resultado_glucosa = modelo_rf.predict(datos_entrada)[0]
        
        # Métrica Visual
        st.metric(
            label=f"Glucosa Estimada a las {hora_simulada.strftime('%H:%M')}",
            value=f"{resultado_glucosa:.2f} mg/dL"
        )
        
        # Alerta Clínica basada en los umbrales
        if resultado_glucosa < 70:
            st.error("Alerta Clínica: Riesgo inminente de Hipoglucemia detectado.")
        elif resultado_glucosa > 140:
            st.warning("Alerta Clínica: Riesgo de Hiperglucemia postprandial detectado.")
        else:
            st.success("Estado Metabólico: Valores estables en rango seguro (70-140 mg/dL).")
            
        # Crear simulación de la curva de glucosa en el tiempo para la gráfica
        st.subheader("Trayectoria de Glucosa Estimada (Próximas Horas)")
        tiempos = []
        valores_proyectados = []
        
        for offset in range(-60, 121, 15):  # Analiza de una hora antes a dos horas después
            t_simulado = (minutos_totales + offset) % 1440
            entrada_temp = pd.DataFrame(0, index=[0], columns=columnas_predictoras)
            entrada_temp['minutos_dia'] = t_simulado
            if columna_comida_activa in entrada_temp.columns:
                entrada_temp[columna_comida_activa] = 1
            tiempos.append(f"{(t_simulado//60):02d}:{(t_simulado%60):02d}")
            valores_proyectados.append(modelo_rf.predict(entrada_temp)[0])
            
        df_grafica = pd.DataFrame({"Hora": tiempos, "Glucosa (mg/dL)": valores_proyectados})
        st.line_chart(df_grafica.set_index("Hora"))
        
    else:
        st.info("Presiona el botón 'Ejecutar Simulación del Gemelo' para proyectar el comportamiento metabólico.")