📊 Dashboard de Análisis de Ventas, Cartera, RFM y Agente – AG Medical Solutions

Dashboard interactivo en Streamlit con analítica de Ventas, Cartera, RFM, modelos predictivos y un Agente que responde preguntas exclusivamente con base en un Reporte Consolidado generado por la app.

⸻

🧭 Índice
	•	Características
	•	Arquitectura & Flujo
	•	Estructura del repositorio
	•	Requisitos
	•	Configuración de API (ROG)
	•	Ejecución local
	•	Despliegue en Streamlit Cloud
	•	Uso de la aplicación
	•	Preguntas al Agente (ejemplos)
	•	Métricas ML y evaluación
	•	Solución de problemas (FAQ)
	•	Futuras mejoras
	•	Créditos

⸻

✨ Características
	1.	Carga y preprocesamiento de datos
	•	Fuente: DB_AGMS.xlsx con hojas Ventas, Lista Medicos, Metadatos, CarteraAgosto.
	•	Limpieza de fechas, columnas monetarias y normalización de nombres.
	2.	Análisis de Ventas (EDA)
	•	KPIs: ventas totales, transacciones, clientes únicos, ticket promedio.
	•	Series temporales por Mes / Semana / Día con SMA.
	•	Top-N por Productos, Clientes y Comerciales.
	•	Pareto 80/20 y Heatmap Día x Mes.
	•	Generador de Reporte Consolidado (texto) para el Agente.
	3.	Gestión de Cartera
	•	Saldos: total pendiente, vencido, por vencer.
	•	Clasificación por estado (Pagada, Vencida, Por Vencer).
	•	Tabla con coloreado condicional y gráfico de antigüedad por rangos.
	4.	Análisis RFM + Recomendador ML
	•	Cálculo de Recencia, Frecuencia, Monetario y segmentos (Champions, Loyal, etc.).
	•	Recomendador: genera Top-N clientes a contactar, filtrables por día y segmentos.
	•	Modelos: Regresión Logística, Random Forest, XGBoost/GradientBoosting, MLP (según entorno).
	5.	Modelo Predictivo de Compradores por Producto
	•	Optimización de hiperparámetros vía RandomizedSearchCV.
	•	Comparación con Balanced Accuracy, MCC y F1-macro.
	•	Exportación de Top 10 candidatos a CSV.
	6.	Agente de Análisis (LLM)
	•	Responde preguntas solo con base en el Reporte Consolidado.
	•	Motor de intents (reglas) + búsqueda semántica (TF-IDF).
	•	Integración opcional con ROG (API OpenAI-compatible).

⸻

🏗️ Arquitectura & Flujo

Diagrama de flujo (Mermaid)

flowchart TD
  A[Inicio: Usuario abre app] --> B[Carga de Excel DB_AGMS.xlsx]
  B --> C[Preprocesamiento: fechas, moneda, columnas]
  C --> D[Tab 1: EDA Ventas]
  C --> E[Tab 2: Cartera]
  C --> F[Tab 3: RFM + Recomendador]
  C --> G[Tab 4: Predictivo por Producto]
  C --> H[Tab 5: Agente]

  D --> D1[KPIs + Series + Top + Pareto + Heatmap]
  D --> D2[Generar Reporte Consolidado (texto)]
  D2 -->|Guarda| S[(st.session_state.AGMS_REPORT)]

  E --> E1[KPIs de Cartera + Estilos + Antigüedad]

  F --> F1[Tabla RFM y Segmentos]
  F --> F2[Features comportamiento (día/hora/producto)]
  F --> F3[Comparación de Modelos (CV)]
  F --> F4[Top-N candidatos (día, segmento)]

  G --> G1[Construcción dataset por producto objetivo]
  G --> G2[RandomSearch: RF / XGB-GB / MLP]
  G --> G3[Mejor modelo y Top 10 clientes]

  H --> H1[Intents (reglas) sobre Reporte]
  H --> H2[Semántica TF-IDF sobre frases del Reporte]
  S --> H

Diagrama de secuencia: “Pregunta al Agente”

sequenceDiagram
  participant U as Usuario
  participant A as Agente (tab 5)
  participant R as Reporte (AGMS_REPORT)
  participant L as LLM (ROG opcional)

  U->>A: Pregunta (ej. "saldo vencido" / "top 5 productos")
  A->>R: Lee Reporte de st.session_state
  alt hay intent que coincide
    A->>A: Extrae dato estructurado del reporte
    A-->>U: Respuesta específica
  else no hay intent claro
    A->>R: Extrae frases relevantes (TF-IDF)
    A->>L: (Opcional) Envía contexto seleccionado
    L-->>A: Respuesta natural (si disponible)
    A-->>U: Resumen basado en frases relevantes
  end


⸻

📁 Estructura del repositorio

.
├── app.py                # Código principal de Streamlit
├── DB_AGMS.xlsx          # Datos de entrada (local / privado)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este documento

⚠️ No es recomendable versionar DB_AGMS.xlsx en público (datos sensibles).

⸻

🧩 Requisitos
	•	Python 3.10+
	•	Paquetes (ver requirements.txt):
	•	streamlit, pandas, numpy, plotly
	•	scikit-learn
	•	xgboost o lightgbm
	•	openai (si usas ROG)

⸻

🔑 Configuración de API (ROG)

En Streamlit Cloud → App → Settings → Secrets (o ~/.streamlit/secrets.toml en local):

ROG_API_KEY = "tu_api_key"
ROG_API_BASE = "https://api.rog.ai/v1" # opcional
ROG_MODEL = "gpt-4o-mini"

El agente funciona sin ROG, pero con ROG las respuestas son más naturales.

⸻

💻 Ejecución local

# 1) Crear y activar entorno
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Ejecutar la app
streamlit run app.py


⸻

☁️ Despliegue en Streamlit Cloud
	1.	Subir repositorio a GitHub.
	2.	En share.streamlit.io, crear la App y apuntar a app.py.
	3.	Definir secrets (si usas ROG).
	4.	Deploy.

⸻

🧑‍🏫 Uso de la aplicación
	•	Tab 1 – Análisis de Ventas: KPIs, Top-N, Pareto, Heatmap, genera Reporte Consolidado.
	•	Tab 2 – Cartera: KPIs de cartera, clasificación, vencimientos.
	•	Tab 3 – RFM: Segmentos, recomendador ML, top candidatos.
	•	Tab 4 – Predictivo por Producto: ML optimizado por producto objetivo.
	•	Tab 5 – Agente: Preguntas al reporte (saldo vencido, top productos, cuántos por segmento).

⸻

💡 Preguntas al Agente (ejemplos)
	•	Cartera
	•	“¿Cuál es el saldo vencido?”
	•	“Saldo entre 31 y 60 días de mora”
	•	“Clientes con menos de 6 días de mora”
	•	Ventas
	•	“Ventas totales”
	•	“Mejor mes de ventas”
	•	“Top 5 productos”
	•	RFM
	•	“¿Cuántos clientes hay en cada segmento?”
	•	“Recencia media”
	•	“Top 10 clientes por monetario”

⸻

📐 Métricas ML y evaluación
	•	Balanced Accuracy: balancea exactitud en clases desbalanceadas.
	•	MCC: robusto ante desbalance.
	•	F1-macro: promedia F1 sin ponderación.

⸻

🛠️ Solución de problemas (FAQ)
	1.	ModuleNotFoundError: instalar dependencias con requirements.txt.
	2.	SyntaxError: revisar indentación de bloques.
	3.	AUC = NaN: ocurre si falta una clase en un fold → se usan Balanced Acc / MCC / F1.
	4.	El agente repite el reporte completo: actualizar bloque del agente (usa intents + TF-IDF).

⸻

🧭 Futuras mejoras
	•	Exportar reporte en PDF con gráficos.
	•	Modo Agente con acceso a datos crudos.
	•	Modelo de propensión por especialidad médica y producto.

⸻

👤 Créditos

AG Medical Solutions – Equipo de Análisis y Datos
Autor: Andrés Puerta González

⸻
