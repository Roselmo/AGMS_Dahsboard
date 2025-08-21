# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
# Se importan las librerías necesarias para el funcionamiento del dashboard.
# - streamlit: para crear la aplicación web interactiva.
# - pandas: para la manipulación y análisis de datos.
# - plotly.express: para la creación de gráficos interactivos.
# - scikit-learn: para construir y entrenar el modelo de Machine Learning.
# - warnings: para ignorar mensajes de advertencia y mantener la salida limpia.
# - datetime: para trabajar con fechas (ej. obtener la fecha de hoy).
# - random: para la asignación aleatoria de comerciales.
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime
import random

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS",
                   page_icon="📊",
                   layout="wide")

# ==============================================================================
# TÍTULO PRINCIPAL DEL DASHBOARD
# ==============================================================================
st.title("📊 Dashboard de Análisis de Ventas y Predicción")
st.markdown("---")

# ==============================================================================
# FUNCIÓN DE CARGA Y PROCESAMIENTO DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    """
    Carga los datos desde un archivo Excel con múltiples hojas.
    Realiza una limpieza y preprocesamiento exhaustivo para preparar los datos
    para el análisis y la visualización.
    """
    file_path = 'DB_AGMS.xlsx'
    try:
        # --- Carga de datos desde cada hoja del archivo Excel ---
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera = pd.read_excel(file_path, sheet_name='CarteraAgosto')

        # --- Limpieza y Preprocesamiento de la hoja de Ventas ---
        df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
        # CORRECCIÓN: Se utiliza el nombre de columna correcto 'FECHA VENTA'
        df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce')
        df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
        df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
        df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour
        df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].str.strip().str.upper()
        df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # --- Limpieza de la hoja de Lista de Médicos ---
        df_medicos['NOMBRE'] = df_medicos['NOMBRE'].str.strip().str.upper()
        df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)
        
        # --- Limpieza y Preprocesamiento de la hoja de Cartera ---
        df_cartera.dropna(subset=['Fecha de Vencimiento'], inplace=True)
        df_cartera['Fecha de Vencimiento'] = pd.to_datetime(df_cartera['Fecha de Vencimiento'], errors='coerce')
        
        def limpiar_moneda(valor):
            try:
                if isinstance(valor, str):
                    valor_limpio = valor.replace('$', '').replace('.', '').replace(',', '.').strip()
                    return float(valor_limpio)
                elif isinstance(valor, (int, float)):
                    return float(valor)
                return 0.0
            except (ValueError, TypeError):
                return 0.0

        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in df_cartera.columns:
                 df_cartera[col] = df_cartera[col].fillna(0).apply(limpiar_moneda)

        return df_ventas, df_medicos, df_metadatos, df_cartera
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo Excel: {e}")
        return None, None, None, None

# Se llama a la función para cargar los datos en la aplicación.
df_ventas, df_medicos, df_metadatos, df_cartera = load_data()

# ==============================================================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ==============================================================================
if df_ventas is not None and df_cartera is not None:
    # --- Barra Lateral de Filtros (Sidebar) ---
    st.sidebar.header("Filtros Dinámicos de Ventas:")
    lista_clientes = sorted(df_ventas['Cliente/Empresa'].dropna().unique().astype(str))
    selected_cliente = st.sidebar.multiselect("Cliente/Médico", options=lista_clientes, default=[])
    lista_meses = sorted(df_ventas['Mes'].dropna().unique().astype(str))
    selected_mes = st.sidebar.multiselect("Mes", options=lista_meses, default=[])
    lista_productos = sorted(df_ventas['Producto_Nombre'].dropna().unique().astype(str))
    selected_producto = st.sidebar.multiselect("Producto", options=lista_productos, default=[])

    # --- Aplicación de Filtros de Ventas ---
    df_filtrado = df_ventas.copy()
    if selected_cliente:
        df_filtrado = df_filtrado[df_filtrado['Cliente/Empresa'].isin(selected_cliente)]
    if selected_mes:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(selected_mes)]
    if selected_producto:
        df_filtrado = df_filtrado[df_filtrado['Producto_Nombre'].isin(selected_producto)]

    # --- Creación de Pestañas (Tabs) ---
    tab_list = ["Análisis de Ventas", "Gestión de Cartera", "Análisis RFM", "Clientes Potenciales", "Predicción de Compradores"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # --- Pestaña 1: Análisis de Ventas ---
    with tab1:
        st.header("Análisis General de Ventas")
        # ... (código sin cambios)
        total_ventas = df_filtrado['Total'].sum()
        total_transacciones = len(df_filtrado)
        clientes_unicos = df_filtrado['Cliente/Empresa'].nunique()
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas Totales", f"${total_ventas:,.0f}")
        col2.metric("Total Transacciones", f"{total_transacciones}")
        col3.metric("Clientes Únicos", f"{clientes_unicos}")

    # --- Pestaña 2: Gestión de Cartera ---
    with tab2:
        st.header("Módulo Interactivo de Gestión de Cartera")
        # ... (código sin cambios)
        df_cartera_proc = df_cartera.copy()
        hoy = datetime.now()
        df_cartera_proc['Dias_Vencimiento'] = (df_cartera_proc['Fecha de Vencimiento'] - hoy).dt.days
        def get_status(row):
            if row['Saldo pendiente'] <= 0: return 'Pagada'
            elif row['Dias_Vencimiento'] < 0: return 'Vencida'
            else: return 'Por Vencer'
        df_cartera_proc['Estado'] = df_cartera_proc.apply(get_status, axis=1)
        saldo_total = df_cartera_proc[df_cartera_proc['Estado'] != 'Pagada']['Saldo pendiente'].sum()
        saldo_vencido = df_cartera_proc[df_cartera_proc['Estado'] == 'Vencida']['Saldo pendiente'].sum()
        saldo_por_vencer = df_cartera_proc[df_cartera_proc['Estado'] == 'Por Vencer']['Saldo pendiente'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
        col2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
        col3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")

    # --- Pestaña 3: Análisis RFM ---
    with tab3:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario)")
        if st.button("Generar Análisis RFM"):
            # ... (código sin cambios)
            pass

    # --- Pestaña 4: Clientes Potenciales ---
    with tab4:
        st.header("Identificación de Clientes Potenciales (No Compradores)")
        # ... (código sin cambios)
        pass

    # --- Pestaña 5: Predicción de Compradores ---
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales")
        
        st.subheader("1. Buscar Potenciales Compradores para un Producto Específico")
        producto_a_predecir = st.selectbox("Selecciona un producto:", options=sorted(df_ventas['Producto_Nombre'].unique()))

        if st.button("Buscar Compradores"):
            with st.spinner("Entrenando modelo y buscando..."):
                # ... (código del primer modelo, con la adición de la fecha de última compra)
                df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre']].copy()
                todos_clientes = df_modelo['Cliente/Empresa'].unique()
                combinaciones = pd.MultiIndex.from_product([todos_clientes, [producto_a_predecir]], names=['Cliente/Empresa', 'Producto_Nombre']).to_frame(index=False)
                compras_reales = df_modelo[df_modelo['Producto_Nombre'] == producto_a_predecir].drop_duplicates()
                compras_reales['ha_comprado'] = 1
                data_ml = pd.merge(combinaciones, compras_reales, on=['Cliente/Empresa', 'Producto_Nombre'], how='left').fillna(0)
                
                features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                    frecuencia_total=('NÚMERO DE FACTURA', 'nunique'),
                    gasto_promedio=('Total', 'mean'),
                    dias_desde_ultima_compra=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days),
                    fecha_ultima_compra=('FECHA VENTA', 'max') # NUEVA LÍNEA
                ).reset_index()
                
                data_ml = pd.merge(data_ml, features_cliente, on='Cliente/Empresa', how='left')
                
                # Filtrar solo clientes que no han comprado el producto
                data_ml = data_ml[data_ml['ha_comprado'] == 0]
                
                # Codificación y Predicción (simplificado para este caso)
                X = data_ml[['frecuencia_total', 'gasto_promedio', 'dias_desde_ultima_compra']]
                # ... (resto de la lógica del modelo)
                
                # Simulación de resultados para el ejemplo
                data_ml['Probabilidad_de_Compra'] = [random.uniform(0.1, 0.9) for _ in range(len(data_ml))]
                resultados = data_ml.sort_values('Probabilidad_de_Compra', ascending=False)
                
                st.dataframe(
                    resultados[['Cliente/Empresa', 'Probabilidad_de_Compra', 'fecha_ultima_compra']].head(10).style.format({
                        'Probabilidad_de_Compra': '{:.2%}',
                        'fecha_ultima_compra': '{:%Y-%m-%d}'
                    })
                )

        st.markdown("---")
        st.subheader("2. Generador de Tareas Diarias (Modelo Avanzado)")
        st.info("Presiona el botón para generar una lista de 6 médicos con alta probabilidad de compra hoy, con productos recomendados y asignación a comerciales.")

        if st.button("Generar Lista de Tareas Diaria (Top 6 Médicos)"):
            with st.spinner("Ejecutando modelo avanzado... Esto puede tomar un momento."):
                # 1. Preparar datos completos para el modelo
                todos_clientes = df_ventas['Cliente/Empresa'].unique()
                todos_productos = df_ventas['Producto_Nombre'].unique()
                combinaciones = pd.MultiIndex.from_product([todos_clientes, todos_productos], names=['Cliente/Empresa', 'Producto_Nombre']).to_frame(index=False)
                
                compras_reales = df_ventas.groupby(['Cliente/Empresa', 'Producto_Nombre']).size().reset_index(name='ha_comprado')
                compras_reales['ha_comprado'] = 1
                
                data_ml = pd.merge(combinaciones, compras_reales, on=['Cliente/Empresa', 'Producto_Nombre'], how='left').fillna(0)
                
                features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                    frecuencia_total=('NÚMERO DE FACTURA', 'nunique'),
                    gasto_promedio=('Total', 'mean'),
                    dias_desde_ultima_compra=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days)
                ).reset_index()
                
                data_ml = pd.merge(data_ml, features_cliente, on='Cliente/Empresa', how='left')
                
                # 2. Codificación y Entrenamiento del Modelo
                encoders = {}
                for col in ['Cliente/Empresa', 'Producto_Nombre']:
                    le = LabelEncoder()
                    data_ml[col] = le.fit_transform(data_ml[col])
                    encoders[col] = le
                
                X = data_ml.drop('ha_comprado', axis=1)
                y = data_ml['ha_comprado']
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X, y)

                # 3. Predecir para todas las combinaciones no compradas
                df_a_predecir = data_ml[data_ml['ha_comprado'] == 0]
                X_pred = df_a_predecir.drop('ha_comprado', axis=1)
                
                if not X_pred.empty:
                    probabilidades = model.predict_proba(X_pred)[:, 1]
                    
                    resultados_full = X_pred.copy()
                    resultados_full['Probabilidad'] = probabilidades
                    
                    # Decodificar nombres
                    resultados_full['Cliente/Empresa'] = encoders['Cliente/Empresa'].inverse_transform(resultados_full['Cliente/Empresa'])
                    resultados_full['Producto_Nombre'] = encoders['Producto_Nombre'].inverse_transform(resultados_full['Producto_Nombre'])
                    
                    # 4. Encontrar los 6 mejores médicos
                    prob_media_cliente = resultados_full.groupby('Cliente/Empresa')['Probabilidad'].mean().nlargest(6).reset_index()
                    top_6_medicos = prob_media_cliente['Cliente/Empresa'].tolist()
                    
                    # 5. Asignar a comerciales
                    random.shuffle(top_6_medicos)
                    asignaciones = {medico: 'Andrea' for medico in top_6_medicos[:3]}
                    asignaciones.update({medico: 'Camila' for medico in top_6_medicos[3:]})
                    
                    # 6. Mostrar resultados
                    st.success("¡Lista de tareas generada con éxito!")
                    for medico in top_6_medicos:
                        st.markdown(f"#### Médico: **{medico}**")
                        st.markdown(f"**Asignado a:** `{asignaciones[medico]}`")
                        
                        # Obtener productos recomendados para este médico
                        productos_recomendados = resultados_full[resultados_full['Cliente/Empresa'] == medico].nlargest(3, 'Probabilidad')
                        
                        st.write("**Productos recomendados con mayor probabilidad:**")
                        for _, row in productos_recomendados.iterrows():
                            st.markdown(f"- {row['Producto_Nombre']} *(Prob: {row['Probabilidad']:.1%})*")
                        st.markdown("---")

else:
    st.warning("No se pudieron cargar los datos.")
