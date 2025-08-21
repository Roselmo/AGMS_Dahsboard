import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard de Ventas AGMS",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Dashboard de Análisis de Ventas")
st.markdown("---")

# --- Carga de datos desde archivos locales ---
# Usamos @st.cache_data para que los datos se carguen una sola vez y la app sea más rápida.
@st.cache_data
def load_data():
    """
    Carga los datos desde los archivos CSV, realiza la limpieza y el preprocesamiento necesarios.
    """
    try:
        # Carga de los archivos CSV
        df_ventas = pd.read_csv('DB_AGMS.xlsx - Ventas.csv')
        df_medicos = pd.read_csv('DB_AGMS.xlsx - Lista Medicos.csv')
        df_metadatos = pd.read_csv('DB_AGMS.xlsx - Metadatos.csv')
        df_cartera = pd.read_csv('DB_AGMS.xlsx - CarteraAgosto.csv')

        # --- Limpieza y Preprocesamiento de Datos ---

        # Hoja de Ventas
        # Asignar la primera fila como encabezado y luego eliminarla
        df_ventas.columns = df_ventas.iloc[0]
        df_ventas = df_ventas[1:].reset_index(drop=True)
        
        # Eliminar filas donde la fecha de venta es nula
        df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
        
        # Conversión de tipos de datos
        df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce')
        
        # Creación de nuevas columnas para facilitar el análisis
        df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
        
        # Limpieza de nombres para consistencia
        df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].str.strip().str.upper()
        
        # Extraer solo el nombre del producto para una mejor visualización
        df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # Hoja de Lista de Médicos
        df_medicos['NOMBRE'] = df_medicos['NOMBRE'].str.strip().str.upper()

        return df_ventas, df_medicos, df_metadatos, df_cartera
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que los archivos CSV estén en la misma carpeta que `app.py`.")
        return None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera = load_data()

# --- Interfaz de la Aplicación ---
if df_ventas is not None:
    # --- Barra Lateral de Filtros ---
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

    # Aplicar filtros al DataFrame
    df_filtrado = df_ventas.copy()
    if selected_cliente:
        df_filtrado = df_filtrado[df_filtrado['Cliente/Empresa'].isin(selected_cliente)]
    if selected_mes:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(selected_mes)]
    if selected_producto:
        df_filtrado = df_filtrado[df_filtrado['Producto_Nombre'].isin(selected_producto)]

    # --- Pestañas de Análisis ---
    tab1, tab2, tab3 = st.tabs(["Análisis de Ventas", "Análisis RFM de Clientes", "Clientes Potenciales"])

    with tab1:
        st.header("Análisis General de Ventas")

        # --- Métricas Principales ---
        total_ventas = df_filtrado['Total'].sum()
        total_transacciones = len(df_filtrado)
        clientes_unicos = df_filtrado['Cliente/Empresa'].nunique()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ventas Totales", f"${total_ventas:,.0f}")
        with col2:
            st.metric("Total Transacciones", f"{total_transacciones}")
        with col3:
            st.metric("Clientes Únicos", f"{clientes_unicos}")

        st.markdown("---")

        # --- Visualizaciones ---
        col_a, col_b = st.columns(2)

        with col_a:
            # Gráfico de Ventas por Mes
            st.subheader("Evolución de Ventas por Mes")
            ventas_por_mes = df_filtrado.groupby('Mes')['Total'].sum().reset_index()
            fig_ventas_mes = px.line(ventas_por_mes, x='Mes', y='Total', title="Ventas Mensuales", markers=True, labels={'Total': 'Ventas ($)', 'Mes': 'Mes'})
            st.plotly_chart(fig_ventas_mes, use_container_width=True)

            # Gráfico de Ventas por Día
            st.subheader("Evolución de Ventas por Día")
            ventas_por_dia = df_filtrado.groupby(df_filtrado['FECHA VENTA'].dt.date)['Total'].sum().reset_index().rename(columns={'FECHA VENTA': 'Fecha'})
            fig_ventas_dia = px.line(ventas_por_dia, x='Fecha', y='Total', title="Ventas Diarias", markers=True, labels={'Total': 'Ventas ($)', 'Fecha': 'Fecha'})
            st.plotly_chart(fig_ventas_dia, use_container_width=True)

        with col_b:
            # Gráfico de Top 10 Productos
            st.subheader("Top 10 Productos por Ventas")
            top_productos = df_filtrado.groupby('Producto_Nombre')['Total'].sum().nlargest(10).reset_index()
            fig_top_productos = px.bar(top_productos, x='Total', y='Producto_Nombre', orientation='h', title="Top 10 Productos", labels={'Total': 'Ventas ($)', 'Producto_Nombre': 'Producto'})
            fig_top_productos.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_productos, use_container_width=True)

            # Gráfico de Top 10 Clientes
            st.subheader("Top 10 Clientes por Ventas")
            top_clientes = df_filtrado.groupby('Cliente/Empresa')['Total'].sum().nlargest(10).reset_index()
            fig_top_clientes = px.bar(top_clientes, x='Total', y='Cliente/Empresa', orientation='h', title="Top 10 Clientes", labels={'Total': 'Ventas ($)', 'Cliente/Empresa': 'Cliente'})
            fig_top_clientes.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_clientes, use_container_width=True)

    with tab2:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario)")

        if st.button("Generar Análisis RFM"):
            with st.spinner('Calculando segmentos RFM...'):
                # Calcular Recencia, Frecuencia, Monetario
                df_rfm = df_ventas.groupby('Cliente/Empresa').agg(
                    Recencia=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days),
                    Frecuencia=('NÚMERO DE FACTURA', 'nunique'),
                    Monetario=('Total', 'sum')
                ).reset_index()

                # Crear quintiles para las puntuaciones
                df_rfm['R_Score'] = pd.qcut(df_rfm['Recencia'], 5, labels=[5, 4, 3, 2, 1])
                df_rfm['F_Score'] = pd.qcut(df_rfm['Frecuencia'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
                df_rfm['M_Score'] = pd.qcut(df_rfm['Monetario'], 5, labels=[1, 2, 3, 4, 5])

                # Segmentación de clientes
                segment_map = {
                    r'[1-2][1-2]': 'Hibernando', r'[1-2][3-4]': 'En Riesgo', r'[1-2]5': 'No se pueden perder',
                    r'3[1-2]': 'A punto de dormir', r'33': 'Necesitan Atención', r'[3-4][4-5]': 'Clientes Leales',
                    r'41': 'Prometedores', r'51': 'Nuevos Clientes', r'[4-5][2-3]': 'Potenciales Leales', r'5[4-5]': 'Campeones'
                }
                df_rfm['Segmento'] = (df_rfm['R_Score'].astype(str) + df_rfm['F_Score'].astype(str)).replace(segment_map, regex=True)

                st.subheader("Segmentación de Clientes RFM")
                st.dataframe(df_rfm[['Cliente/Empresa', 'Recencia', 'Frecuencia', 'Monetario', 'Segmento']])

                # Visualización de la segmentación
                st.subheader("Distribución de Segmentos de Clientes")
                segment_counts = df_rfm['Segmento'].value_counts().reset_index()
                fig_segmentos = px.bar(segment_counts, x='Segmento', y='count', title="Número de Clientes por Segmento RFM", color='Segmento', labels={'count': 'Número de Clientes'})
                st.plotly_chart(fig_segmentos, use_container_width=True)

                # Scatter plot para RFM
                st.subheader("Visualización RFM")
                fig_scatter = px.scatter(df_rfm, x='Recencia', y='Frecuencia', size='Monetario', color='Segmento', hover_name='Cliente/Empresa', title="Recencia vs. Frecuencia (Tamaño por Valor Monetario)")
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.header("Identificación de Clientes Potenciales")

        # Comparar listas de médicos
        medicos_compradores = df_ventas['Cliente/Empresa'].unique()
        df_medicos_potenciales = df_medicos[~df_medicos['NOMBRE'].isin(medicos_compradores)]

        st.info(f"Se encontraron **{len(df_medicos_potenciales)}** médicos en la lista que aún no han realizado compras.")

        # Filtro por especialidad
        especialidades = sorted(df_medicos_potenciales['ESPECIALIDAD MEDICA'].dropna().unique())
        selected_especialidad = st.selectbox("Filtrar por Especialidad Médica:", options=['Todas'] + especialidades)

        if selected_especialidad != 'Todas':
            df_display = df_medicos_potenciales[df_medicos_potenciales['ESPECIALIDAD MEDICA'] == selected_especialidad]
        else:
            df_display = df_medicos_potenciales

        st.dataframe(df_display[['NOMBRE', 'ESPECIALIDAD MEDICA', 'TELEFONO', 'EMAIL', 'CIUDAD']])
else:
    st.warning("No se pudieron cargar los datos. Por favor, verifica que los archivos CSV estén correctos y en la misma carpeta.")

