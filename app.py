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
        # Se especifica el nombre de la hoja (sheet_name) y, en el caso de Ventas,
        # se indica que el encabezado está en la segunda fila (header=1).
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera = pd.read_excel(file_path, sheet_name='CarteraAgosto')

        # --- Limpieza y Preprocesamiento de la hoja de Ventas ---

        # Eliminar filas donde la fecha de venta es nula, ya que son cruciales.
        df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
        
        # Conversión de tipos de datos para análisis correctos.
        # 'coerce' convierte los valores no válidos en NaT (Not a Time) o NaN (Not a Number).
        df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce')
        
        # --- Feature Engineering (Creación de nuevas columnas) ---
        # Se extraen componentes de la fecha para facilitar filtros y análisis temporales.
        df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
        df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
        df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour
        
        # Limpieza de texto para estandarizar los nombres.
        # .str.strip() elimina espacios al inicio y final.
        # .str.upper() convierte todo a mayúsculas para evitar duplicados por capitalización.
        df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].str.strip().str.upper()
        
        # Se extrae solo el nombre del producto para una mejor visualización y análisis.
        df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # --- Limpieza de la hoja de Lista de Médicos ---
        df_medicos['NOMBRE'] = df_medicos['NOMBRE'].str.strip().str.upper()

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
# Se verifica que los datos se hayan cargado correctamente antes de construir la interfaz.
# Si df_ventas es None, significa que hubo un error en la carga.
# ==============================================================================
if df_ventas is not None:
    # --- Barra Lateral de Filtros (Sidebar) ---
    st.sidebar.header("Filtros Dinámicos:")

    # Filtro por Médico/Cliente
    lista_clientes = sorted(df_ventas['Cliente/Empresa'].unique())
    selected_cliente = st.sidebar.multiselect("Cliente/Médico", options=lista_clientes, default=[])

    # Filtro por Mes
    lista_meses = sorted(df_ventas['Mes'].unique())
    selected_mes = st.sidebar.multiselect("Mes", options=lista_meses, default=[])

    # Filtro por Producto
    lista_productos = sorted(df_ventas['Producto_Nombre'].unique())
    selected_producto = st.sidebar.multiselect("Producto", options=lista_productos, default=[])

    # --- Aplicación de Filtros ---
    # Se crea una copia del DataFrame original para no alterar los datos en caché.
    # Luego, se aplican los filtros seleccionados en la barra lateral.
    df_filtrado = df_ventas.copy()
    if selected_cliente:
        df_filtrado = df_filtrado[df_filtrado['Cliente/Empresa'].isin(selected_cliente)]
    if selected_mes:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(selected_mes)]
    if selected_producto:
        df_filtrado = df_filtrado[df_filtrado['Producto_Nombre'].isin(selected_producto)]

    # --- Creación de Pestañas (Tabs) ---
    # st.tabs() permite organizar el contenido en diferentes secciones navegables.
    tab1, tab2, tab3, tab4 = st.tabs([
        "Análisis de Ventas", 
        "Análisis RFM de Clientes", 
        "Clientes Potenciales",
        "Predicción de Compradores"
    ])

    # --- Contenido de la Pestaña 1: Análisis de Ventas ---
    with tab1:
        st.header("Análisis General de Ventas")

        # Métricas Principales (KPIs)
        total_ventas = df_filtrado['Total'].sum()
        total_transacciones = len(df_filtrado)
        clientes_unicos = df_filtrado['Cliente/Empresa'].nunique()

        # st.columns() crea un layout de columnas para mostrar las métricas una al lado de la otra.
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ventas Totales", f"${total_ventas:,.0f}")
        with col2:
            st.metric("Total Transacciones", f"{total_transacciones}")
        with col3:
            st.metric("Clientes Únicos", f"{clientes_unicos}")

        st.markdown("---")

        # Visualizaciones de Ventas
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Evolución de Ventas por Mes")
            ventas_por_mes = df_filtrado.groupby('Mes')['Total'].sum().reset_index()
            fig_ventas_mes = px.line(ventas_por_mes, x='Mes', y='Total', title="Ventas Mensuales", markers=True)
            st.plotly_chart(fig_ventas_mes, use_container_width=True)

            st.subheader("Evolución de Ventas por Día")
            ventas_por_dia = df_filtrado.groupby(df_filtrado['FECHA VENTA'].dt.date)['Total'].sum().reset_index().rename(columns={'FECHA VENTA': 'Fecha'})
            fig_ventas_dia = px.line(ventas_por_dia, x='Fecha', y='Total', title="Ventas Diarias", markers=True)
            st.plotly_chart(fig_ventas_dia, use_container_width=True)

        with col_b:
            st.subheader("Top 10 Productos por Ventas")
            top_productos = df_filtrado.groupby('Producto_Nombre')['Total'].sum().nlargest(10).reset_index()
            fig_top_productos = px.bar(top_productos, x='Total', y='Producto_Nombre', orientation='h', title="Top 10 Productos")
            fig_top_productos.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_productos, use_container_width=True)

            st.subheader("Top 10 Clientes por Ventas")
            top_clientes = df_filtrado.groupby('Cliente/Empresa')['Total'].sum().nlargest(10).reset_index()
            fig_top_clientes = px.bar(top_clientes, x='Total', y='Cliente/Empresa', orientation='h', title="Top 10 Clientes")
            fig_top_clientes.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_clientes, use_container_width=True)

    # --- Contenido de la Pestaña 2: Análisis RFM ---
    with tab2:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario)")
        if st.button("Generar Análisis RFM"):
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

                st.subheader("Segmentación de Clientes RFM")
                st.dataframe(df_rfm[['Cliente/Empresa', 'Recencia', 'Frecuencia', 'Monetario', 'Segmento']])

                st.subheader("Distribución de Segmentos de Clientes")
                segment_counts = df_rfm['Segmento'].value_counts().reset_index()
                fig_segmentos = px.bar(segment_counts, x='Segmento', y='count', title="Número de Clientes por Segmento RFM", color='Segmento')
                st.plotly_chart(fig_segmentos, use_container_width=True)

    # --- Contenido de la Pestaña 3: Clientes Potenciales ---
    with tab3:
        st.header("Identificación de Clientes Potenciales (No Compradores)")
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
        
    # --- Contenido de la Pestaña 4: Predicción de Compradores ---
    with tab4:
        st.header("Modelo Predictivo de Compradores Potenciales")
        st.write("""
        Esta herramienta utiliza un modelo de Machine Learning para predecir qué clientes 
        tienen una mayor probabilidad de comprar un producto específico, basándose en el 
        comportamiento histórico de todos los clientes.
        """)

        # Selector para que el usuario elija un producto
        producto_a_predecir = st.selectbox(
            "Selecciona un producto para encontrar compradores potenciales:",
            options=sorted(df_ventas['Producto_Nombre'].unique())
        )

        if st.button("Generar Predicción de Compradores"):
            with st.spinner("Entrenando modelo y generando predicciones... Esto puede tardar un momento."):
                # 1. Preparación de datos para el modelo
                df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre', 'Dia_Semana', 'Hora']].copy()
                
                # Crear variable objetivo: 1 si el cliente compró el producto, 0 si no.
                # Se crea una tabla con todas las combinaciones posibles de cliente-producto.
                todos_clientes = df_modelo['Cliente/Empresa'].unique()
                todos_productos = df_modelo['Producto_Nombre'].unique()
                combinaciones = pd.MultiIndex.from_product([todos_clientes, todos_productos], names=['Cliente/Empresa', 'Producto_Nombre']).to_frame(index=False)
                
                # Se marcan las compras reales
                compras_reales = df_modelo.groupby(['Cliente/Empresa', 'Producto_Nombre']).size().reset_name('ha_comprado')
                compras_reales['ha_comprado'] = 1
                
                # Se unen las combinaciones con las compras reales
                data_ml = pd.merge(combinaciones, compras_reales, on=['Cliente/Empresa', 'Producto_Nombre'], how='left').fillna(0)

                # 2. Feature Engineering para el modelo
                # Se calculan características adicionales para cada cliente que puedan predecir su comportamiento.
                features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                    frecuencia_total=('NÚMERO DE FACTURA', 'nunique'),
                    gasto_promedio=('Total', 'mean'),
                    dias_desde_ultima_compra=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days)
                ).reset_index()
                
                data_ml = pd.merge(data_ml, features_cliente, on='Cliente/Empresa', how='left')

                # 3. Codificación de variables categóricas
                # Los modelos de ML necesitan números, no texto. LabelEncoder convierte texto a números.
                encoders = {}
                for col in ['Cliente/Empresa', 'Producto_Nombre']:
                    le = LabelEncoder()
                    data_ml[col] = le.fit_transform(data_ml[col])
                    encoders[col] = le

                # 4. Entrenamiento del Modelo (Random Forest)
                X = data_ml.drop('ha_comprado', axis=1)
                y = data_ml['ha_comprado']
                
                # Se usa RandomForest por su buen rendimiento en problemas de clasificación.
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X, y)

                # 5. Generación de Predicciones
                # Se identifican los clientes que NO han comprado el producto seleccionado.
                clientes_actuales = df_ventas[df_ventas['Producto_Nombre'] == producto_a_predecir]['Cliente/Empresa'].unique()
                clientes_potenciales_nombres = [c for c in todos_clientes if c not in clientes_actuales]

                # Se prepara el set de datos de estos clientes para la predicción.
                df_prediccion = data_ml[data_ml['Cliente/Empresa'].isin(encoders['Cliente/Empresa'].transform(clientes_potenciales_nombres))].copy()
                df_prediccion['Producto_Nombre'] = encoders['Producto_Nombre'].transform([producto_a_predecir])[0]
                
                X_pred = df_prediccion.drop('ha_comprado', axis=1).drop_duplicates()

                # Se predice la probabilidad de compra.
                if not X_pred.empty:
                    probabilidades = model.predict_proba(X_pred)[:, 1]
                    
                    # 6. Mostrar Resultados
                    resultados = pd.DataFrame({
                        'Cliente/Empresa_encoded': X_pred['Cliente/Empresa'],
                        'Probabilidad_de_Compra': probabilidades
                    })
                    # Se decodifican los nombres para que sean legibles.
                    resultados['Cliente/Empresa'] = encoders['Cliente/Empresa'].inverse_transform(resultados['Cliente/Empresa_encoded'])
                    resultados = resultados.sort_values(by='Probabilidad_de_Compra', ascending=False).drop_duplicates('Cliente/Empresa')

                    st.subheader(f"Top Potenciales Compradores para: {producto_a_predecir}")
                    # Se muestra el top 10 con formato de porcentaje.
                    st.dataframe(
                        resultados[['Cliente/Empresa', 'Probabilidad_de_Compra']].head(10).style.format({'Probabilidad_de_Compra': '{:.2%}'})
                    )
                else:
                    st.warning("No se encontraron clientes potenciales para este producto o ya todos lo han comprado.")

else:
    st.warning("No se pudieron cargar los datos. Por favor, verifica que el archivo 'DB_AGMS.xlsx' esté en la misma carpeta.")
