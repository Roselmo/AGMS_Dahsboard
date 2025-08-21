# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
# Se importan las librerías necesarias para el funcionamiento del dashboard.
# - streamlit: para crear la aplicación web interactiva.
# - pandas: para la manipulación y análisis de datos.
# - plotly.express: para la creación de gráficos interactivos.
# - scikit-learn: para construir y entrenar el modelo de Machine Learning.
# - warnings: para ignorar mensajes de advertencia y mantener la salida limpia.
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# ==============================================================================
# st.set_page_config() se usa para configurar atributos de la página como el
# título que aparece en la pestaña del navegador, el ícono y el layout.
# "wide" utiliza todo el ancho de la pantalla para el contenido.
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
# Se utiliza el decorador @st.cache_data para que Streamlit guarde en caché
# el resultado de esta función. Esto significa que los datos se cargarán y
# procesarán solo una vez, haciendo que la aplicación sea mucho más rápida
# en ejecuciones posteriores (por ejemplo, al cambiar un filtro).
# ==============================================================================
@st.cache_data
def load_data():
    """
    Carga los datos desde un archivo Excel con múltiples hojas.
    Realiza una limpieza y preprocesamiento exhaustivo para preparar los datos
    para el análisis y la visualización.
    
    Retorna:
        - df_ventas: DataFrame con los datos de ventas limpios.
        - df_medicos: DataFrame con la lista de médicos.
        - df_metadatos: DataFrame con metadatos.
        - df_cartera: DataFrame con información de cartera.
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
        
        # --- Limpieza y Preprocesamiento de la hoja de Cartera ---
        df_cartera.dropna(subset=['Fecha de Vencimiento'], inplace=True)
        df_cartera['Fecha de Vencimiento'] = pd.to_datetime(df_cartera['Fecha de Vencimiento'], errors='coerce')
        
        # Función para limpiar columnas de moneda de forma más robusta
        def limpiar_moneda(valor):
            try:
                if isinstance(valor, str):
                    # Limpiar el string de caracteres no numéricos excepto la coma decimal
                    valor_limpio = valor.replace('$', '').replace('.', '').replace(',', '.').strip()
                    return float(valor_limpio)
                # Si ya es un número (int o float), lo retorna
                elif isinstance(valor, (int, float)):
                    return float(valor)
                # Si no es ninguno de los anteriores, retorna 0
                return 0.0
            except (ValueError, TypeError):
                return 0.0

        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in df_cartera.columns:
                 df_cartera[col] = df_cartera[col].fillna(0).apply(limpiar_moneda)

        return df_ventas, df_medicos, df_metadatos, df_cartera
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que el archivo Excel esté en la misma carpeta que `app.py`.")
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
    
    # CORRECCIÓN: Se asegura que las listas para los filtros no contengan valores nulos (NaN)
    # y que todos los elementos sean strings, para evitar errores en los widgets de Streamlit.
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
    tab_list = [
        "Análisis de Ventas", 
        "Gestión de Cartera",
        "Análisis RFM", 
        "Clientes Potenciales",
        "Predicción de Compradores"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # --- Contenido de la Pestaña 1: Análisis de Ventas ---
    with tab1:
        st.header("Análisis General de Ventas")
        total_ventas = df_filtrado['Total'].sum()
        total_transacciones = len(df_filtrado)
        clientes_unicos = df_filtrado['Cliente/Empresa'].nunique()
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas Totales", f"${total_ventas:,.0f}")
        col2.metric("Total Transacciones", f"{total_transacciones}")
        col3.metric("Clientes Únicos", f"{clientes_unicos}")
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Evolución de Ventas por Mes")
            ventas_por_mes = df_filtrado.groupby('Mes')['Total'].sum().reset_index()
            fig_ventas_mes = px.line(ventas_por_mes, x='Mes', y='Total', title="Ventas Mensuales", markers=True)
            st.plotly_chart(fig_ventas_mes, use_container_width=True)
        with col_b:
            st.subheader("Top 10 Productos por Ventas")
            top_productos = df_filtrado.groupby('Producto_Nombre')['Total'].sum().nlargest(10).reset_index()
            fig_top_productos = px.bar(top_productos, x='Total', y='Producto_Nombre', orientation='h', title="Top 10 Productos")
            fig_top_productos.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_productos, use_container_width=True)

    # --- Contenido de la Pestaña 2: Gestión de Cartera ---
    with tab2:
        st.header("Módulo Interactivo de Gestión de Cartera")
        
        # Preparación de datos de cartera
        df_cartera_proc = df_cartera.copy()
        hoy = datetime.now()
        df_cartera_proc['Dias_Vencimiento'] = (df_cartera_proc['Fecha de Vencimiento'] - hoy).dt.days

        def get_status(row):
            if row['Saldo pendiente'] <= 0:
                return 'Pagada'
            elif row['Dias_Vencimiento'] < 0:
                return 'Vencida'
            else:
                return 'Por Vencer'
        df_cartera_proc['Estado'] = df_cartera_proc.apply(get_status, axis=1)

        # KPIs de Cartera
        saldo_total = df_cartera_proc[df_cartera_proc['Estado'] != 'Pagada']['Saldo pendiente'].sum()
        saldo_vencido = df_cartera_proc[df_cartera_proc['Estado'] == 'Vencida']['Saldo pendiente'].sum()
        saldo_por_vencer = df_cartera_proc[df_cartera_proc['Estado'] == 'Por Vencer']['Saldo pendiente'].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
        col2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
        col3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
        st.markdown("---")

        # Filtros para la tabla de cartera
        filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada'])
        lista_clientes_cartera = sorted(df_cartera_proc['Nombre cliente'].dropna().unique())
        filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera)

        df_cartera_filtrada = df_cartera_proc.copy()
        if filtro_estado != 'Todas':
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Estado'] == filtro_estado]
        if filtro_cliente:
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Nombre cliente'].isin(filtro_cliente)]
        
        # Estilo para la tabla
        def style_vencimiento(row):
            if row['Estado'] == 'Vencida':
                return ['background-color: #ffcccc'] * len(row)
            elif 0 <= row['Dias_Vencimiento'] <= 7:
                 return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)
        
        st.dataframe(df_cartera_filtrada[['Nombre cliente', 'NÚMERO DE FACTURA', 'Fecha de Vencimiento', 'Saldo pendiente', 'Estado', 'Dias_Vencimiento']].style.apply(style_vencimiento, axis=1).format({'Saldo pendiente': '${:,.0f}'}))

        # Gráficos de Cartera
        st.markdown("---")
        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Top 10 Clientes por Saldo Pendiente")
            top_deudores = df_cartera_proc[df_cartera_proc['Estado'] != 'Pagada'].groupby('Nombre cliente')['Saldo pendiente'].sum().nlargest(10).reset_index()
            fig_deudores = px.bar(top_deudores, x='Saldo pendiente', y='Nombre cliente', orientation='h', title="Top 10 Deudores")
            fig_deudores.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_deudores, use_container_width=True)
        with col_d:
            st.subheader("Distribución de la Cartera por Estado")
            estado_dist = df_cartera_proc[df_cartera_proc['Estado'] != 'Pagada']['Estado'].value_counts().reset_index()
            fig_dist = px.pie(estado_dist, values='count', names='Estado', title='Cartera Vencida vs. Por Vencer', color_discrete_map={'Vencida':'red', 'Por Vencer':'green'})
            st.plotly_chart(fig_dist, use_container_width=True)

    # --- Contenido de la Pestaña 3: Análisis RFM ---
    with tab3:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario)")
        if st.button("Generar Análisis RFM"):
            # ... (código RFM sin cambios)
            with st.spinner('Calculando segmentos RFM...'):
                df_rfm = df_ventas.groupby('Cliente/Empresa').agg(
                    Recencia=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days),
                    Frecuencia=('NÚMERO DE FACTURA', 'nunique'),
                    Monetario=('Total', 'sum')
                ).reset_index()
                df_rfm['R_Score'] = pd.qcut(df_rfm['Recencia'], 5, labels=[5, 4, 3, 2, 1])
                df_rfm['F_Score'] = pd.qcut(df_rfm['Frecuencia'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
                df_rfm['M_Score'] = pd.qcut(df_rfm['Monetario'], 5, labels=[1, 2, 3, 4, 5])
                segment_map = {
                    r'[1-2][1-2]': 'Hibernando', r'[1-2][3-4]': 'En Riesgo', r'[1-2]5': 'No se pueden perder',
                    r'3[1-2]': 'A punto de dormir', r'33': 'Necesitan Atención', r'[3-4][4-5]': 'Clientes Leales',
                    r'41': 'Prometedores', r'51': 'Nuevos Clientes', r'[4-5][2-3]': 'Potenciales Leales', r'5[4-5]': 'Campeones'
                }
                df_rfm['Segmento'] = (df_rfm['R_Score'].astype(str) + df_rfm['F_Score'].astype(str)).replace(segment_map, regex=True)
                st.dataframe(df_rfm[['Cliente/Empresa', 'Recencia', 'Frecuencia', 'Monetario', 'Segmento']])

    # --- Contenido de la Pestaña 4: Clientes Potenciales ---
    with tab4:
        st.header("Identificación de Clientes Potenciales (No Compradores)")
        # ... (código de clientes potenciales sin cambios)
        medicos_compradores = df_ventas['Cliente/Empresa'].unique()
        df_medicos_potenciales = df_medicos[~df_medicos['NOMBRE'].isin(medicos_compradores)]
        st.info(f"Se encontraron **{len(df_medicos_potenciales)}** médicos en la lista que aún no han realizado compras.")
        especialidades = sorted(df_medicos_potenciales['ESPECIALIDAD MEDICA'].dropna().unique())
        selected_especialidad = st.selectbox("Filtrar por Especialidad Médica:", options=['Todas'] + especialidades)
        if selected_especialidad != 'Todas':
            df_display = df_medicos_potenciales[df_medicos_potenciales['ESPECIALIDAD MEDICA'] == selected_especialidad]
        else:
            df_display = df_medicos_potenciales
        st.dataframe(df_display[['NOMBRE', 'ESPECIALIDAD MEDICA', 'TELEFONO', 'EMAIL', 'CIUDAD']])

    # --- Contenido de la Pestaña 5: Predicción de Compradores ---
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales")
        # ... (código de predicción sin cambios)
        producto_a_predecir = st.selectbox("Selecciona un producto para encontrar compradores potenciales:", options=sorted(df_ventas['Producto_Nombre'].unique()))
        if st.button("Generar Predicción de Compradores"):
            with st.spinner("Entrenando modelo y generando predicciones..."):
                df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre', 'Dia_Semana', 'Hora']].copy()
                todos_clientes = df_modelo['Cliente/Empresa'].unique()
                todos_productos = df_modelo['Producto_Nombre'].unique()
                combinaciones = pd.MultiIndex.from_product([todos_clientes, todos_productos], names=['Cliente/Empresa', 'Producto_Nombre']).to_frame(index=False)
                compras_reales = df_modelo.groupby(['Cliente/Empresa', 'Producto_Nombre']).size().reset_index(name='ha_comprado')
                compras_reales['ha_comprado'] = 1
                data_ml = pd.merge(combinaciones, compras_reales, on=['Cliente/Empresa', 'Producto_Nombre'], how='left').fillna(0)
                features_cliente = df_ventas.groupby('Cliente/Empresa').agg(frecuencia_total=('NÚMERO DE FACTURA', 'nunique'), gasto_promedio=('Total', 'mean'), dias_desde_ultima_compra=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days)).reset_index()
                data_ml = pd.merge(data_ml, features_cliente, on='Cliente/Empresa', how='left')
                encoders = {}
                for col in ['Cliente/Empresa', 'Producto_Nombre']:
                    le = LabelEncoder()
                    data_ml[col] = le.fit_transform(data_ml[col])
                    encoders[col] = le
                X = data_ml.drop('ha_comprado', axis=1)
                y = data_ml['ha_comprado']
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X, y)
                clientes_actuales = df_ventas[df_ventas['Producto_Nombre'] == producto_a_predecir]['Cliente/Empresa'].unique()
                clientes_potenciales_nombres = [c for c in todos_clientes if c not in clientes_actuales]
                df_prediccion = data_ml[data_ml['Cliente/Empresa'].isin(encoders['Cliente/Empresa'].transform(clientes_potenciales_nombres))].copy()
                df_prediccion['Producto_Nombre'] = encoders['Producto_Nombre'].transform([producto_a_predecir])[0]
                X_pred = df_prediccion.drop('ha_comprado', axis=1).drop_duplicates()
                if not X_pred.empty:
                    probabilidades = model.predict_proba(X_pred)[:, 1]
                    resultados = pd.DataFrame({'Cliente/Empresa_encoded': X_pred['Cliente/Empresa'], 'Probabilidad_de_Compra': probabilidades})
                    resultados['Cliente/Empresa'] = encoders['Cliente/Empresa'].inverse_transform(resultados['Cliente/Empresa_encoded'])
                    resultados = resultados.sort_values(by='Probabilidad_de_Compra', ascending=False).drop_duplicates('Cliente/Empresa')
                    st.subheader(f"Top Potenciales Compradores para: {producto_a_predecir}")
                    st.dataframe(resultados[['Cliente/Empresa', 'Probabilidad_de_Compra']].head(10).style.format({'Probabilidad_de_Compra': '{:.2%}'}))
else:
    st.warning("No se pudieron cargar los datos. Por favor, verifica que el archivo 'DB_AGMS.xlsx' esté en la misma carpeta.")
