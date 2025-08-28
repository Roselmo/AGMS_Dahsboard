# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
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
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
            df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
            df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
            df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
            df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour

        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            if col in df_ventas.columns:
                df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce')

        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        if 'Producto' in df_ventas.columns:
            df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # --- Limpieza de la hoja de Lista de Médicos ---
        if 'NOMBRE' in df_medicos.columns:
            df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
        if 'TELEFONO' in df_medicos.columns:
            df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # --- Limpieza y Preprocesamiento de la hoja de Cartera ---
        if 'Fecha de Vencimiento' in df_cartera.columns:
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
    lista_clientes = sorted(df_ventas['Cliente/Empresa'].dropna().unique().astype(str)) if 'Cliente/Empresa' in df_ventas.columns else []
    selected_cliente = st.sidebar.multiselect("Cliente/Médico", options=lista_clientes, default=[])
    lista_meses = sorted(df_ventas['Mes'].dropna().unique().astype(str)) if 'Mes' in df_ventas.columns else []
    selected_mes = st.sidebar.multiselect("Mes", options=lista_meses, default=[])
    lista_productos = sorted(df_ventas['Producto_Nombre'].dropna().unique().astype(str)) if 'Producto_Nombre' in df_ventas.columns else []
    selected_producto = st.sidebar.multiselect("Producto", options=lista_productos, default=[])

    # --- Aplicación de Filtros de Ventas ---
    df_filtrado = df_ventas.copy()
    if selected_cliente and 'Cliente/Empresa' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Cliente/Empresa'].isin(selected_cliente)]
    if selected_mes and 'Mes' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(selected_mes)]
    if selected_producto and 'Producto_Nombre' in df_filtrado.columns:
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
                    fig = px.line(serie, x=eje_tiempo, y="Total", markers=True, title=f"Ventas por {granularidad}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No hay columna '{eje_tiempo}' para la serie. Asegura tener '{fecha_col}' para construirla.")

            with cB:
                st.subheader(f"Top {top_n} por {dimension}" if dimension in df_filtrado.columns else "Top-N")
                if dimension in df_filtrado.columns:
                    top_df = (df_filtrado.groupby(dimension, as_index=False)["Total"]
                                         .sum()
                                         .sort_values("Total", ascending=False)
                                         .head(top_n))
                    fig = px.bar(top_df, x="Total", y=dimension, orientation="h",
                                 title=f"Top {top_n} por {dimension}")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
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
                fig = px.line(serie, x=eje_tiempo, y=["Total", "SMA"], markers=True,
                              title=f"Ventas vs SMA ({ventana}) · {granularidad}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(serie, use_container_width=True)
            else:
                st.info(f"No hay columna '{eje_tiempo}' para la serie.")

        # ===== Productos =====
        with tab_productos:
            st.subheader(f"Top {top_n} Productos y participación")
            if "Producto_Nombre" in df_filtrado.columns:
                prod = (df_filtrado.groupby("Producto_Nombre", as_index=False)["Total"].sum()
                                   .sort_values("Total", ascending=False))
                total_prod = prod["Total"].sum()
                prod["%_participación"] = 100 * prod["Total"] / total_prod if total_prod else 0
                top_prod = prod.head(top_n)

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h",
                                 title=f"Top {top_n} Productos")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig = px.treemap(prod, path=["Producto_Nombre"], values="Total",
                                     title="Treemap participación de ventas")
                    st.plotly_chart(fig, use_container_width=True)

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
                fig = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h",
                             title=f"Top {top_n} Clientes")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_cli, use_container_width=True)
            else:
                st.warning("No se encontró la columna 'Cliente/Empresa'.")

        # ===== Pareto =====
        with tab_pareto:
            st.subheader("Análisis de Pareto (80/20) por " + (dimension if dimension in df_filtrado.columns else "dimensión"))
            if dimension in df_filtrado.columns:
                base = (df_filtrado.groupby(dimension, as_index=False)["Total"].sum()
                        .sort_values("Total", ascending=False))
                total_base = base["Total"].sum()
                base["%_acum"] = 100 * base["Total"].cumsum() / total_base if total_base else 0

                fig = px.bar(base, x=dimension, y="Total", title="Ventas por " + dimension)
                fig2 = px.line(base, x=dimension, y="%_acum")
                for tr in fig2.data:
                    fig.add_trace(tr)
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="% acumulado"),
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
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
                work["DiaSemana"] = dt.dt.day_name()
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

                fig = px.imshow(heat, aspect="auto", title="Heatmap ventas (Día semana x Mes)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columna de fecha para construir el mapa de calor.")

    # --- Pestaña 2: Gestión de Cartera ---
    with tab2:
        st.header("Módulo Interactivo de Gestión de Cartera")
        df_cartera_proc = df_cartera.copy()
        hoy = datetime.now()
        if 'Fecha de Vencimiento' in df_cartera_proc.columns:
            df_cartera_proc['Dias_Vencimiento'] = (df_cartera_proc['Fecha de Vencimiento'] - hoy).dt.days
        else:
            df_cartera_proc['Dias_Vencimiento'] = None

        def get_status(row):
            if 'Saldo pendiente' in row and row['Saldo pendiente'] <= 0: 
                return 'Pagada'
            elif row['Dias_Vencimiento'] is not None and row['Dias_Vencimiento'] < 0: 
                return 'Vencida'
            else: 
                return 'Por Vencer'

        df_cartera_proc['Estado'] = df_cartera_proc.apply(get_status, axis=1)

        saldo_total = df_cartera_proc[df_cartera_proc['Estado'] != 'Pagada']['Saldo pendiente'].sum() if 'Saldo pendiente' in df_cartera_proc.columns else 0
        saldo_vencido = df_cartera_proc[df_cartera_proc['Estado'] == 'Vencida']['Saldo pendiente'].sum() if 'Saldo pendiente' in df_cartera_proc.columns else 0
        saldo_por_vencer = df_cartera_proc[df_cartera_proc['Estado'] == 'Por Vencer']['Saldo pendiente'].sum() if 'Saldo pendiente' in df_cartera_proc.columns else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
        col2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
        col3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
        st.markdown("---")

        filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada'])
        lista_clientes_cartera = sorted(df_cartera_proc['Nombre cliente'].dropna().unique()) if 'Nombre cliente' in df_cartera_proc.columns else []
        filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera)

        df_cartera_filtrada = df_cartera_proc.copy()
        if filtro_estado != 'Todas':
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Estado'] == filtro_estado]
        if filtro_cliente and 'Nombre cliente' in df_cartera_filtrada.columns:
            df_cartera_filtrada = df_cartera_filtrada[df_cartera_filtrada['Nombre cliente'].isin(filtro_cliente)]

        def style_vencimiento(row):
            if row['Estado'] == 'Vencida': 
                return ['background-color: #ffcccc'] * len(row)
            elif isinstance(row['Dias_Vencimiento'], (int, float)) and 0 <= row['Dias_Vencimiento'] <= 7: 
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)

        cols_show = [c for c in ['Nombre cliente', 'NÚMERO DE FACTURA', 'Fecha de Vencimiento', 'Saldo pendiente', 'Estado', 'Dias_Vencimiento'] if c in df_cartera_filtrada.columns]
        st.dataframe(
            df_cartera_filtrada[cols_show]
            .style.apply(style_vencimiento, axis=1)
            .format({'Saldo pendiente': '${:,.0f}'}) if cols_show else pd.DataFrame()
        )

        # --- Pestaña 3: Análisis RFM ---
    with tab3:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario)")
        if st.button("Generar Análisis RFM"):
            with st.spinner('Calculando segmentos RFM...'):
                if {'Cliente/Empresa','FECHA VENTA','Total'}.issubset(df_ventas.columns):
                    df_rfm = df_ventas.groupby('Cliente/Empresa').agg(
                        Recencia=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days),
                        Frecuencia=('NÚMERO DE FACTURA', 'nunique') if 'NÚMERO DE FACTURA' in df_ventas.columns else ('FECHA VENTA','count'),
                        Monetario=('Total', 'sum')
                    ).reset_index()

                    # -------- Construcción de features adicionales --------
                    df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.dayofweek
                    df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour

                    feats_dia = df_ventas.groupby(['Cliente/Empresa','Dia_Semana']).size().unstack(fill_value=0)
                    feats_hora = df_ventas.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)

                    df_feat = df_rfm.merge(feats_dia, on='Cliente/Empresa', how='left')
                    df_feat = df_feat.merge(feats_hora, on='Cliente/Empresa', how='left')
                    df_feat = df_feat.fillna(0)

                    # -------- Target: compró en últimos 30 días --------
                    ref_date = df_ventas['FECHA VENTA'].max()
                    recientes = df_ventas[df_ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=30)]['Cliente/Empresa'].unique()
                    df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                    X = df_feat.drop(['Cliente/Empresa','comprador_reciente'], axis=1)
                    y = df_feat['comprador_reciente']

                    # Split train/test
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

                    # Modelos
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.ensemble import RandomForestClassifier
                    from xgboost import XGBClassifier

                    modelos = {
                        "LogisticRegression": LogisticRegression(max_iter=500),
                        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }

                    resultados = []
                    mejores_preds = None
                    mejor_modelo = None
                    mejor_auc = -1

                    for nombre, modelo in modelos.items():
                        modelo.fit(X_train, y_train)
                        y_pred = modelo.predict(X_test)
                        y_prob = modelo.predict_proba(X_test)[:,1]

                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        auc = roc_auc_score(y_test, y_prob)
                        resultados.append({"Modelo": nombre, "Accuracy": acc, "F1": f1, "AUC": auc})

                        if auc > mejor_auc:
                            mejor_auc = auc
                            mejor_modelo = modelo
                            mejores_preds = modelo.predict_proba(X)[:,1]

                    st.subheader("Comparación de Modelos ML")
                    st.dataframe(pd.DataFrame(resultados).round(3))

                    # -------- Recomendación Top 10 --------
                    df_feat['Prob_Compra'] = mejores_preds
                    df_no_compradores = df_feat[df_feat['comprador_reciente']==0].copy()
                    top10 = df_no_compradores.nlargest(10, 'Prob_Compra')[['Cliente/Empresa','Prob_Compra']]

                    # Asignación de Camila/Andrea
                    asignaciones = ['Camila','Andrea'] * 5
                    top10['Asignado_a'] = asignaciones

                    st.subheader("🎯 Top 10 Clientes Potenciales a Contactar")
                    st.dataframe(top10.style.format({'Prob_Compra':"{:.1%}"}))
                else:
                    st.warning("No están las columnas mínimas para RFM y modelado.")

    # --- Pestaña 4: Clientes Potenciales ---
    with tab4:
        st.header("Identificación de Clientes Potenciales (No Compradores)")
        if 'Cliente/Empresa' in df_ventas.columns and 'NOMBRE' in df_medicos.columns:
            medicos_compradores = df_ventas['Cliente/Empresa'].unique()
            df_medicos_potenciales = df_medicos[~df_medicos['NOMBRE'].isin(medicos_compradores)]
            st.info(f"Se encontraron **{len(df_medicos_potenciales)}** médicos en la lista que aún no han realizado compras.")
            if 'ESPECIALIDAD MEDICA' in df_medicos_potenciales.columns:
                especialidades = sorted(df_medicos_potenciales['ESPECIALIDAD MEDICA'].dropna().unique())
                selected_especialidad = st.selectbox("Filtrar por Especialidad Médica:", options=['Todas'] + especialidades)
                if selected_especialidad != 'Todas':
                    df_display = df_medicos_potenciales[df_medicos_potenciales['ESPECIALIDAD MEDICA'] == selected_especialidad]
                else:
                    df_display = df_medicos_potenciales
                st.dataframe(df_display[['NOMBRE', 'ESPECIALIDAD MEDICA', 'TELEFONO', 'EMAIL', 'CIUDAD']] if set(['NOMBRE','ESPECIALIDAD MEDICA','TELEFONO','EMAIL','CIUDAD']).issubset(df_display.columns) else df_display)
            else:
                st.dataframe(df_medicos_potenciales)
        else:
            st.warning("Faltan columnas para cruzar compradores con la lista de médicos.")

    # --- Pestaña 5: Predicción de Compradores ---
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales")

        if 'Producto_Nombre' in df_ventas.columns:
            st.subheader("1. Buscar Potenciales Compradores para un Producto Específico")
            producto_a_predecir = st.selectbox("Selecciona un producto:", options=sorted(df_ventas['Producto_Nombre'].unique()))

            if st.button("Buscar Compradores"):
                with st.spinner("Entrenando modelo y buscando..."):
                    df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre']].copy()
                    todos_clientes = df_modelo['Cliente/Empresa'].unique()

                    features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                        fecha_ultima_compra=('FECHA VENTA', 'max') if 'FECHA VENTA' in df_ventas.columns else ('Producto_Nombre','count')
                    ).reset_index()

                    clientes_actuales = df_ventas[df_ventas['Producto_Nombre'] == producto_a_predecir]['Cliente/Empresa'].unique()
                    clientes_potenciales = [c for c in todos_clientes if c not in clientes_actuales]

                    df_resultados = pd.DataFrame(clientes_potenciales, columns=['Cliente/Empresa'])
                    df_resultados = pd.merge(df_resultados, features_cliente, on='Cliente/Empresa', how='left')

                    # Simulación de probabilidad (demo)
                    df_resultados['Probabilidad_de_Compra'] = [random.uniform(0.1, 0.9) for _ in range(len(df_resultados))]
                    df_resultados = df_resultados.sort_values('Probabilidad_de_Compra', ascending=False)

                    st.dataframe(
                        df_resultados[['Cliente/Empresa', 'Probabilidad_de_Compra', 'fecha_ultima_compra']].head(10).style.format({
                            'Probabilidad_de_Compra': '{:.2%}',
                            'fecha_ultima_compra': '{:%Y-%m-%d}' if 'fecha_ultima_compra' in df_resultados.columns else '{}'
                        })
                    )
        else:
            st.info("No hay columna 'Producto_Nombre' para realizar esta predicción.")

        st.markdown("---")
        st.subheader("2. Generador de Tareas Diarias (Modelo Avanzado)")
        st.info("Presiona el botón para generar una lista de 6 médicos con alta probabilidad de compra hoy, con productos recomendados y asignación a comerciales.")

        if st.button("Generar Lista de Tareas Diaria (Top 6 Médicos)"):
            with st.spinner("Ejecutando modelo avanzado..."):
                if {'Cliente/Empresa','Producto_Nombre'}.issubset(df_ventas.columns):
                    todos_clientes = df_ventas['Cliente/Empresa'].unique()
                    todos_productos = df_ventas['Producto_Nombre'].unique()
                    combinaciones = pd.MultiIndex.from_product([todos_clientes, todos_productos], names=['Cliente/Empresa', 'Producto_Nombre']).to_frame(index=False)

                    compras_reales = df_ventas.groupby(['Cliente/Empresa', 'Producto_Nombre']).size().reset_index(name='ha_comprado')
                    compras_reales['ha_comprado'] = 1

                    data_ml = pd.merge(combinaciones, compras_reales, on=['Cliente/Empresa', 'Producto_Nombre'], how='left').fillna(0)

                    features_cliente = df_ventas.groupby('Cliente/Empresa').agg(
                        frecuencia_total=('NÚMERO DE FACTURA', 'nunique') if 'NÚMERO DE FACTURA' in df_ventas.columns else ('Producto_Nombre','count'),
                        gasto_promedio=('Total', 'mean') if 'Total' in df_ventas.columns else ('Producto_Nombre','count'),
                        dias_desde_ultima_compra=('FECHA VENTA', lambda date: (df_ventas['FECHA VENTA'].max() - date.max()).days) if 'FECHA VENTA' in df_ventas.columns else ('Producto_Nombre','count')
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
                    st.warning("No hay columnas suficientes (Cliente/Empresa y Producto_Nombre) para generar la lista.")

else:
    st.warning("No se pudieron cargar los datos.")
