
# Gemelo Digital para la Predicción de Glucosa en Pacientes con Diabetes Tipo 1 (DM1)

## 1. Autora y Afiliación
* **Investigadora:** Laura Daniela Merchán Gil (Colombia)
* **Asesor Principal:** Prof. David Gerardo Alfaro Viquez (Costa Rica)
* **Programa:** Programa Delfín – Estancia de Investigación Internacional
* **Línea de Investigación:** Inteligencia Artificial Aplicada a la Salud Humana

---

## 2. Descripción del Proyecto
Este repositorio contiene el código fuente, la metodología y los experimentos desarrollados para la construcción de un **Gemelo Digital Personalizado** enfocado en el **Paciente 005** de la base de datos médica **D1NAMO**. 

El objetivo principal es predecir los niveles de glucosa en la sangre con un horizonte temporal de **30 y 60 minutos** ($G_{t+k}$), permitiendo al paciente anticipar crisis de hipoglucemia o hiperglucemia de forma segura.

---

## 3. Formulación Matemática Formal

El sistema modela el metabolismo glucémico como un problema de regresión secuencial basado en series de tiempo:

$$G_{t+k} = f(G_t, G_{t-1}, G_{t-2}, \dots)$$

### 4. Algoritmo de Optimización (Árboles de Decisión)
Para evitar el sobreajuste (*overfitting*) característico de los modelos poblacionales neuronales (como LSTM) en muestras individuales, el Gemelo Digital implementa **Árboles de Decisión para Regresión**. El criterio de división del espacio de datos busca minimizar la varianza de los errores cuadráticos residuales en cada nodo:

$$\text{Minimizar } J(j, s) = \sum_{i \in R_{\text{izq}}(j,s)} (y_i - \hat{y}_{\text{izq}})^2 + \sum_{i \in R_{\text{der}}(j,s)} (y_i - \hat{y}_{\text{der}})^2$$

Donde:
* $J(j, s)$: Función de costo o error total medido en la partición.
* $j$: Variable o rezago temporal seleccionado para realizar el corte.
* $s$: Umbral o punto de división numérico aplicado sobre la variable.
* $y_i$: Valor real de glucosa registrado por el sensor continuo (CGM).
* $\hat{y}_{\text{izq}} / \hat{y}_{\text{der}}$: Media matemática de la predicción en las ramas correspondientes.

---

## 5. Resultados Clínicos Destacados (Paciente 005)

Tras la fase de pruebas comparativas, la aproximación personalizada mediante el Gemelo Digital demostró una superioridad crítica frente a modelos globales:

| Métrica / Criterio | Modelo Poblacional (LSTM General) | Gemelo Digital (Árboles Personalizados) |
| :--- | :---: | :---: |
| **Error Absoluto Medio (MAE)** | > 20.0 mg/dL | **9.31 mg/dL** |
| **Zona A de la Rejilla de Clarke** | < 85.0% | **100.0% (Éxito Clínico Absoluto)** |
| **Zonas de Riesgo Fisiológico (C, D, E)**| Presencia de alertas erróneas | **0.0% (Seguridad Total)** |

---

## 6. Estructura del Repositorio
* `/data/`: Scripts para la descarga y el preprocesamiento de las señales de la base de datos D1NAMO (Paciente 005).
* `/models/`: Implementación de los cuadernos de código (`.ipynb`) correspondientes a las redes LSTM poblacionales y los modelos personalizados de Árboles de Decisión.
* `/docs/`: Reporte Técnico Oficial maquetado en formato APA v9 y recursos visuales para el Cartel Científico.

---

## 7. Cómo Ejecutar los Experimentos

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/gemelo-digital-glucosa.git](https://github.com/TU_USUARIO/gemelo-digital-glucosa.git)
   cd gemelo-digital-glucosa
