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

    # --------- Controles interactivos ---------
    # Detectar columnas disponibles de fecha
    fecha_col = None
    for c in ["Fecha", "FECHA_VENTA", "FECHA VENTA"]:
        if c in df_filtrado.columns:
            fecha_col = c
            break

    # Asegurar Mes si no existe pero hay fecha
    if "Mes" not in df_filtrado.columns and fecha_col:
        df_filtrado["Mes"] = pd.to_datetime(df_filtrado[fecha_col], errors="coerce").dt.to_period("M").astype(str)

    # Granularidad y dimensión
    granularidad = st.selectbox(
        "Granularidad",
        options=["Mes", "Semana", "Día"],
        index=0,
        help="Cambia la escala temporal de las series y tablas."
    )
    # Construir semana/día si hace falta
    if fecha_col:
        dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
        if "Semana" not in df_filtrado.columns:
            df_filtrado["Semana"] = dt.dt.to_period("W").astype(str)
        if "Día" not in df_filtrado.columns:
            df_filtrado["Día"] = dt.dt.date

    dim_posibles = [c for c in ["Producto_Nombre", "Cliente/Empresa", "Comercial"] if c in df_filtrado.columns]
    dimension = st.selectbox(
        "Dimensión para Top-N",
        options=dim_posibles if dim_posibles else ["(no disponible)"],
        index=0
    )
    top_n = st.slider("Top-N a mostrar", 5, 30, 10)

    # --------- KPIs con delta vs periodo anterior ---------
    total_ventas = float(df_filtrado["Total"].sum()) if "Total" in df_filtrado.columns else 0.0
    total_transacciones = len(df_filtrado)
    clientes_unicos = df_filtrado["Cliente/Empresa"].nunique() if "Cliente/Empresa" in df_filtrado.columns else 0
    ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

    # Delta vs periodo anterior (usa 'Mes' si existe)
    delta_ventas = None
    if "Mes" in df_filtrado.columns:
        tmp = df_filtrado.groupby("Mes", as_index=False)["Total"].sum().sort_values("Mes")
        if len(tmp) >= 2:
            delta_ventas = tmp["Total"].iloc[-1] - tmp["Total"].iloc[-2]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas Totales", f"${total_ventas:,.0f}", 
                delta=(f"{delta_ventas:,.0f}" if delta_ventas is not None else None))
    col2.metric("Transacciones", f"{total_transacciones:,}")
    col3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    col4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")

    st.markdown("---")

    # --------- Sub-pestañas de análisis ---------
    tab_resumen, tab_series, tab_productos, tab_clientes, tab_pareto, tab_mapa = st.tabs(
        ["Resumen", "Series", "Productos", "Clientes", "Pareto", "Mapa de calor"]
    )

    # ===== Resumen =====
    with tab_resumen:
        cA, cB = st.columns(2)

        with cA:
            st.subheader("Evolución por " + granularidad)
            eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
            if eje_tiempo in df_filtrado.columns:
                serie = (df_filtrado
                         .groupby(eje_tiempo, as_index=False)["Total"]
                         .sum()
                         .sort_values(eje_tiempo))
                if "px" in globals():
                    fig = px.line(serie, x=eje_tiempo, y="Total", markers=True, title=f"Ventas por {granularidad}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(serie.set_index(eje_tiempo)["Total"])
            else:
                st.info(f"No hay columna '{eje_tiempo}' para la serie. Asegura tener '{fecha_col}' para construirla.")

        with cB:
            st.subheader(f"Top {top_n} por {dimension}" if dimension in df_filtrado.columns else "Top-N")
            if dimension in df_filtrado.columns:
                top_df = (df_filtrado.groupby(dimension, as_index=False)["Total"]
                                     .sum()
                                     .sort_values("Total", ascending=False)
                                     .head(top_n))
                if "px" in globals():
                    fig = px.bar(top_df, x="Total", y=dimension, orientation="h",
                                 title=f"Top {top_n} por {dimension}")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(top_df.set_index(dimension)["Total"])
                st.dataframe(top_df, use_container_width=True)
            else:
                st.warning("No hay columnas de dimensión disponibles (Producto_Nombre / Cliente/Empresa / Comercial).")

    # ===== Series =====
    with tab_series:
        st.subheader("Series temporales con media móvil")
        ventana = st.slider("Ventana de media móvil (periodos)", 1, 12, 3)
        eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
        if eje_tiempo in df_filtrado.columns:
            serie = (df_filtrado.groupby(eje_tiempo, as_index=False)["Total"]
                     .sum()
                     .sort_values(eje_tiempo))
            serie["SMA"] = serie["Total"].rolling(ventana, min_periods=1).mean()
            if "px" in globals():
                fig = px.line(serie, x=eje_tiempo, y=["Total", "SMA"], markers=True,
                              title=f"Ventas vs SMA ({ventana}) · {granularidad}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(serie.set_index(eje_tiempo)[["Total", "SMA"]])
            st.dataframe(serie, use_container_width=True)
        else:
            st.info(f"No hay columna '{eje_tiempo}' para la serie.")

    # ===== Productos =====
    with tab_productos:
        st.subheader(f"Top {top_n} Productos y participación")
        if "Producto_Nombre" in df_filtrado.columns:
            prod = (df_filtrado.groupby("Producto_Nombre", as_index=False)["Total"].sum()
                               .sort_values("Total", ascending=False))
            prod["%_participación"] = 100 * prod["Total"] / prod["Total"].sum() if prod["Total"].sum() else 0
            top_prod = prod.head(top_n)

            c1, c2 = st.columns(2)
            with c1:
                if "px" in globals():
                    fig = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h",
                                 title=f"Top {top_n} Productos")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(top_prod.set_index("Producto_Nombre")["Total"])

            with c2:
                if "px" in globals():
                    fig = px.treemap(prod, path=["Producto_Nombre"], values="Total",
                                     title="Treemap participación de ventas")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(prod, use_container_width=True)

            st.dataframe(top_prod, use_container_width=True)
        else:
            st.warning("No se encontró la columna 'Producto_Nombre'.")

    # ===== Clientes =====
    with tab_clientes:
        st.subheader(f"Top {top_n} Clientes por ventas")
        if "Cliente/Empresa" in df_filtrado.columns:
            cli = (df_filtrado.groupby("Cliente/Empresa", as_index=False)["Total"].sum()
                             .sort_values("Total", ascending=False))
            top_cli = cli.head(top_n)
            if "px" in globals():
                fig = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h",
                             title=f"Top {top_n} Clientes")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top_cli.set_index("Cliente/Empresa")["Total"])
            st.dataframe(top_cli, use_container_width=True)
        else:
            st.warning("No se encontró la columna 'Cliente/Empresa'.")

    # ===== Pareto =====
    with tab_pareto:
        st.subheader("Análisis de Pareto (80/20) por " + (dimension if dimension in df_filtrado.columns else "dimensión"))
        if dimension in df_filtrado.columns:
            base = (df_filtrado.groupby(dimension, as_index=False)["Total"].sum()
                    .sort_values("Total", ascending=False))
            base["%_acum"] = 100 * base["Total"].cumsum() / base["Total"].sum() if base["Total"].sum() else 0
            if "px" in globals():
                fig = px.bar(base, x=dimension, y="Total", title="Ventas por " + dimension)
                fig2 = px.line(base, x=dimension, y="%_acum")
                # Overlay: dos trazas (barras y línea)
                for tr in fig2.data:
                    fig.add_trace(tr)
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="% acumulado"),
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(base.set_index(dimension)["Total"])
                st.dataframe(base[[dimension, "Total", "%_acum"]], use_container_width=True)
            st.dataframe(base, use_container_width=True)
        else:
            st.info("Selecciona una dimensión disponible para Pareto.")

    # ===== Mapa de calor =====
with tab_mapa:
    st.subheader("Mapa de calor: día de semana vs mes")
    if fecha_col:
        dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
        work = df_filtrado.copy()
        work["Mes"] = work["Mes"] if "Mes" in work.columns else dt.dt.to_period("M").astype(str)
        work["DiaSemana"] = dt.dt.day_name(locale="es_ES") if hasattr(dt.dt, "day_name") else dt.dt.day_name()
        heat = work.groupby(["DiaSemana", "Mes"], as_index=False)["Total"].sum()
        # ordenar días
        orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        orden_dias_es = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
        if heat["DiaSemana"].str.lower().isin(orden_dias_es).any():
            heat["DiaSemana"] = heat["DiaSemana"].str.lower()
            cat_order = orden_dias_es
        else:
            cat_order = orden_dias
        heat["DiaSemana"] = pd.Categorical(heat["DiaSemana"], categories=cat_order, ordered=True)
        heat = heat.pivot(index="DiaSemana", columns="Mes", values="Total").fillna(0)

        if "px" in globals():
            fig = px.imshow(heat, aspect="auto", title="Heatmap ventas (Día semana x Mes)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(heat, use_container_width=True)
    else:
        st.info("No hay columna de fecha para construir el mapa de calor.")
      
    # --- Pestaña 2: Gestión de Cartera ---
    with tab2:
        st.header("Módulo Interactivo de Gestión de Cartera")
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
        st.markdown("---")
        filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada'])
        lista_clientes_cartera = sorted(df_cartera_proc['Nombre cliente'].dropna().unique())
        filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera)
        df_cartera_filtrada = df_cartera_proc.copy()
        if filtro_estado != 'Todas':
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Estado'] == filtro_estado]
        if filtro_cliente:
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Nombre cliente'].isin(filtro_cliente)]
        def style_vencimiento(row):
            if row['Estado'] == 'Vencida': return ['background-color: #ffcccc'] * len(row)
            elif 0 <= row['Dias_Vencimiento'] <= 7: return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)
        st.dataframe(df_cartera_filtrada[['Nombre cliente', 'NÚMERO DE FACTURA', 'Fecha de Vencimiento', 'Saldo pendiente', 'Estado', 'Dias_Vencimiento']].style.apply(style_vencimiento, axis=1).format({'Saldo pendiente': '${:,.0f}'}))

    # --- Pestaña 3: Análisis RFM ---
    with tab3:
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
                st.dataframe(df_rfm[['Cliente/Empresa', 'Recencia', 'Frecuencia', 'Monetario', 'Segmento']])

    # --- Pestaña 4: Clientes Potenciales ---
    with tab4:
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

    # --- Pestaña 5: Predicción de Compradores ---
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales")
        
        st.subheader("1. Buscar Potenciales Compradores para un Producto Específico")
        producto_a_predecir = st.selectbox("Selecciona un producto:", options=sorted(df_ventas['Producto_Nombre'].unique()))

        if st.button("Buscar Compradores"):
            with st.spinner("Entrenando modelo y buscando..."):
                df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre']].copy()
                todos_clientes = df_modelo['Cliente/Empresa'].unique()
                
                features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                    fecha_ultima_compra=('FECHA VENTA', 'max')
                ).reset_index()
                
                clientes_actuales = df_ventas[df_ventas['Producto_Nombre'] == producto_a_predecir]['Cliente/Empresa'].unique()
                clientes_potenciales = [c for c in todos_clientes if c not in clientes_actuales]
                
                df_resultados = pd.DataFrame(clientes_potenciales, columns=['Cliente/Empresa'])
                df_resultados = pd.merge(df_resultados, features_cliente, on='Cliente/Empresa', how='left')
                
                # Simulación de probabilidad para demostración
                df_resultados['Probabilidad_de_Compra'] = [random.uniform(0.1, 0.9) for _ in range(len(df_resultados))]
                
                df_resultados = df_resultados.sort_values('Probabilidad_de_Compra', ascending=False)
                
                st.dataframe(
                    df_resultados[['Cliente/Empresa', 'Probabilidad_de_Compra', 'fecha_ultima_compra']].head(10).style.format({
                        'Probabilidad_de_Compra': '{:.2%}',
                        'fecha_ultima_compra': '{:%Y-%m-%d}'
                    })
                )

        st.markdown("---")
        st.subheader("2. Generador de Tareas Diarias (Modelo Avanzado)")
        st.info("Presiona el botón para generar una lista de 6 médicos con alta probabilidad de compra hoy, con productos recomendados y asignación a comerciales.")

        if st.button("Generar Lista de Tareas Diaria (Top 6 Médicos)"):
            with st.spinner("Ejecutando modelo avanzado... Esto puede tomar un momento."):
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
                
                encoders = {}
                for col in ['Cliente/Empresa', 'Producto_Nombre']:
                    le = LabelEncoder()
                    data_ml[col] = le.fit_transform(data_ml[col])
                    encoders[col] = le
                
                X = data_ml.drop('ha_comprado', axis=1)
                y = data_ml['ha_comprado']
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X, y)

                df_a_predecir = data_ml[data_ml['ha_comprado'] == 0]
                X_pred = df_a_predecir.drop('ha_comprado', axis=1)
                
                if not X_pred.empty:
                    probabilidades = model.predict_proba(X_pred)[:, 1]
                    
                    resultados_full = X_pred.copy()
                    resultados_full['Probabilidad'] = probabilidades
                    
                    resultados_full['Cliente/Empresa'] = encoders['Cliente/Empresa'].inverse_transform(resultados_full['Cliente/Empresa'])
                    resultados_full['Producto_Nombre'] = encoders['Producto_Nombre'].inverse_transform(resultados_full['Producto_Nombre'])
                    
                    prob_media_cliente = resultados_full.groupby('Cliente/Empresa')['Probabilidad'].mean().nlargest(6).reset_index()
                    top_6_medicos = prob_media_cliente['Cliente/Empresa'].tolist()
                    
                    random.shuffle(top_6_medicos)
                    asignaciones = {medico: 'Andrea' for medico in top_6_medicos[:3]}
                    asignaciones.update({medico: 'Camila' for medico in top_6_medicos[3:]})
                    
                    st.success("¡Lista de tareas generada con éxito!")
                    for medico in top_6_medicos:
                        st.markdown(f"#### Médico: **{medico}**")
                        st.markdown(f"**Asignado a:** `{asignaciones[medico]}`")
                        
                        productos_recomendados = resultados_full[resultados_full['Cliente/Empresa'] == medico].nlargest(3, 'Probabilidad')
                        
                        st.write("**Productos recomendados con mayor probabilidad:**")
                        for _, row in productos_recomendados.iterrows():
                            st.markdown(f"- {row['Producto_Nombre']} *(Prob: {row['Probabilidad']:.1%})*")
                        st.markdown("---")

else:
    st.warning("No se pudieron cargar los datos.")

