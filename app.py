# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime
import random
import numpy as np

warnings.filterwarnings('ignore')

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

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
    Realiza una limpieza y preprocesamiento para preparar los datos.
    """
    file_path = 'DB_AGMS.xlsx'
    try:
        # --- Carga ---
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera = pd.read_excel(file_path, sheet_name='CarteraAgosto')

        # --- Ventas ---
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
            df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
            df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
            df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
            df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour

        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            if col in df_ventas.columns:
                df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce').fillna(0)

        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        if 'Producto' in df_ventas.columns:
            df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # --- Médicos ---
        if 'NOMBRE' in df_medicos.columns:
            df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
        if 'TELEFONO' in df_medicos.columns:
            df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # --- Cartera ---
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

# Carga
df_ventas, df_medicos, df_metadatos, df_cartera = load_data()

# ==============================================================================
# CUERPO PRINCIPAL
# ==============================================================================
if df_ventas is not None and df_cartera is not None:
    # --- Sidebar ---
    st.sidebar.header("Filtros Dinámicos de Ventas:")
    lista_clientes = sorted(df_ventas['Cliente/Empresa'].dropna().unique().astype(str)) if 'Cliente/Empresa' in df_ventas.columns else []
    selected_cliente = st.sidebar.multiselect("Cliente/Médico", options=lista_clientes, default=[])
    lista_meses = sorted(df_ventas['Mes'].dropna().unique().astype(str)) if 'Mes' in df_ventas.columns else []
    selected_mes = st.sidebar.multiselect("Mes", options=lista_meses, default=[])
    lista_productos = sorted(df_ventas['Producto_Nombre'].dropna().unique().astype(str)) if 'Producto_Nombre' in df_ventas.columns else []
    selected_producto = st.sidebar.multiselect("Producto", options=lista_productos, default=[])

    # --- Filtrado ---
    df_filtrado = df_ventas.copy()
    if selected_cliente and 'Cliente/Empresa' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Cliente/Empresa'].isin(selected_cliente)]
    if selected_mes and 'Mes' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(selected_mes)]
    if selected_producto and 'Producto_Nombre' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Producto_Nombre'].isin(selected_producto)]

    # --- Tabs ---
    tab_list = ["Análisis de Ventas", "Gestión de Cartera", "Análisis RFM", "Clientes Potenciales", "Predicción de Compradores"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # ---------------------------------------------------------------------------------
    # Pestaña 1: Análisis de Ventas
    # ---------------------------------------------------------------------------------
    with tab1:
        st.header("Análisis General de Ventas")

        # Detectar columna de fecha
        fecha_col = None
        for c in ["Fecha", "FECHA_VENTA", "FECHA VENTA"]:
            if c in df_filtrado.columns:
                fecha_col = c
                break

        # Asegurar Mes si falta
        if "Mes" not in df_filtrado.columns and fecha_col:
            df_filtrado["Mes"] = pd.to_datetime(df_filtrado[fecha_col], errors="coerce").dt.to_period("M").astype(str)

        # Controles
        granularidad = st.selectbox("Granularidad", options=["Mes", "Semana", "Día"], index=0)
        if fecha_col:
            dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
            if "Semana" not in df_filtrado.columns:
                df_filtrado["Semana"] = dt.dt.to_period("W").astype(str)
            if "Día" not in df_filtrado.columns:
                df_filtrado["Día"] = dt.dt.date

        dim_posibles = [c for c in ["Producto_Nombre", "Cliente/Empresa", "Comercial"] if c in df_filtrado.columns]
        dimension = st.selectbox("Dimensión para Top-N", options=dim_posibles if dim_posibles else ["(no disponible)"], index=0)
        top_n = st.slider("Top-N a mostrar", 5, 30, 10)

        # KPIs
        total_ventas = float(df_filtrado["Total"].sum()) if "Total" in df_filtrado.columns else 0.0
        total_transacciones = len(df_filtrado)
        clientes_unicos = df_filtrado["Cliente/Empresa"].nunique() if "Cliente/Empresa" in df_filtrado.columns else 0
        ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

        delta_ventas = None
        if "Mes" in df_filtrado.columns:
            tmp = df_filtrado.groupby("Mes", as_index=False)["Total"].sum().sort_values("Mes")
            if len(tmp) >= 2:
                delta_ventas = tmp["Total"].iloc[-1] - tmp["Total"].iloc[-2]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ventas Totales", f"${total_ventas:,.0f}", delta=(f"{delta_ventas:,.0f}" if delta_ventas is not None else None))
        c2.metric("Transacciones", f"{total_transacciones:,}")
        c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
        c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")

        st.markdown("---")

        # Subpestañas
        tab_resumen, tab_series, tab_productos, tab_clientes, tab_pareto, tab_mapa = st.tabs(
            ["Resumen", "Series", "Productos", "Clientes", "Pareto", "Mapa de calor"]
        )

        # Resumen
        with tab_resumen:
            a, b = st.columns(2)
            with a:
                st.subheader("Evolución por " + granularidad)
                eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
                if eje_tiempo in df_filtrado.columns:
                    serie = (df_filtrado.groupby(eje_tiempo, as_index=False)["Total"].sum().sort_values(eje_tiempo))
                    fig = px.line(serie, x=eje_tiempo, y="Total", markers=True, title=f"Ventas por {granularidad}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No hay columna '{eje_tiempo}'.")

            with b:
                st.subheader(f"Top {top_n} por {dimension}" if dimension in df_filtrado.columns else "Top-N")
                if dimension in df_filtrado.columns:
                    top_df = (df_filtrado.groupby(dimension, as_index=False)["Total"]
                              .sum().sort_values("Total", ascending=False).head(top_n))
                    fig = px.bar(top_df, x="Total", y=dimension, orientation="h", title=f"Top {top_n} por {dimension}")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(top_df, use_container_width=True)
                else:
                    st.warning("No hay columnas para Top-N.")

        # Series
        with tab_series:
            st.subheader("Series temporales con media móvil")
            ventana = st.slider("Ventana de media móvil (periodos)", 1, 12, 3)
            eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
            if eje_tiempo in df_filtrado.columns:
                serie = (df_filtrado.groupby(eje_tiempo, as_index=False)["Total"].sum().sort_values(eje_tiempo))
                serie["SMA"] = serie["Total"].rolling(ventana, min_periods=1).mean()
                fig = px.line(serie, x=eje_tiempo, y=["Total", "SMA"], markers=True,
                              title=f"Ventas vs SMA ({ventana}) · {granularidad}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(serie, use_container_width=True)
            else:
                st.info(f"No hay columna '{eje_tiempo}'.")

        # Productos
        with tab_productos:
            st.subheader(f"Top {top_n} Productos y participación")
            if "Producto_Nombre" in df_filtrado.columns:
                prod = (df_filtrado.groupby("Producto_Nombre", as_index=False)["Total"].sum()
                        .sort_values("Total", ascending=False))
                total_prod = prod["Total"].sum()
                prod["%_participación"] = 100 * prod["Total"] / total_prod if total_prod else 0
                top_prod = prod.head(top_n)

                cA, cB = st.columns(2)
                with cA:
                    fig = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h", title=f"Top {top_n} Productos")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                with cB:
                    fig = px.treemap(prod, path=["Producto_Nombre"], values="Total", title="Treemap participación")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_prod, use_container_width=True)
            else:
                st.warning("No se encontró 'Producto_Nombre'.")

        # Clientes
        with tab_clientes:
            st.subheader(f"Top {top_n} Clientes por ventas")
            if "Cliente/Empresa" in df_filtrado.columns:
                cli = (df_filtrado.groupby("Cliente/Empresa", as_index=False)["Total"].sum()
                       .sort_values("Total", ascending=False))
                top_cli = cli.head(top_n)
                fig = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h", title=f"Top {top_n} Clientes")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_cli, use_container_width=True)
            else:
                st.warning("No se encontró 'Cliente/Empresa'.")

        # Pareto
        with tab_pareto:
            st.subheader("Análisis de Pareto (80/20) por " + (dimension if dimension in df_filtrado.columns else "dimensión"))
            if dimension in df_filtrado.columns:
                base = (df_filtrado.groupby(dimension, as_index=False)["Total"].sum().sort_values("Total", ascending=False))
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
                st.info("Selecciona una dimensión para Pareto.")

        # Mapa de calor
        with tab_mapa:
            st.subheader("Mapa de calor: día de semana vs mes")
            if fecha_col:
                dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
                work = df_filtrado.copy()
                work["Mes"] = work["Mes"] if "Mes" in work.columns else dt.dt.to_period("M").astype(str)
                work["DiaSemana"] = dt.dt.day_name()
                heat = work.groupby(["DiaSemana", "Mes"], as_index=False)["Total"].sum()
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
                st.info("No hay columna de fecha para el mapa de calor.")

    # ---------------------------------------------------------------------------------
    # Pestaña 2: Gestión de Cartera  (con fix de Rango/antigüedad)
    # ---------------------------------------------------------------------------------
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
        c2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
        c3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
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

        st.markdown("---")
        # Antigüedad de saldos (fix con 'Rango')
        if {'Fecha de Vencimiento','Saldo pendiente'}.issubset(df_cartera.columns):
            car = df_cartera[['Fecha de Vencimiento','Saldo pendiente']].copy()
            car['DIAS_VENCIDOS'] = (pd.Timestamp.today().normalize() - car['Fecha de Vencimiento']).dt.days
            labels = ["Al día", "1-30", "31-60", "61-90", "91-180", "181-365", "+365"]
            bins = [-float("inf"), 0, 30, 60, 90, 180, 365, float("inf")]
            car["Rango"] = pd.cut(car["DIAS_VENCIDOS"], bins=bins, labels=labels, ordered=True)
            venc = car.groupby("Rango", as_index=False).agg(Saldo=("Saldo pendiente","sum"))
            fig = px.bar(venc, x="Rango", y="Saldo", title="Antigüedad de saldos")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------------
    # Pestaña 3: Análisis RFM + Recomendador ML
    # ---------------------------------------------------------------------------------
    with tab3:
        st.header("Análisis RFM (Recencia, Frecuencia, Monetario) + Recomendador ML")

        # Parámetros
        colp1, colp2, colp3 = st.columns(3)
        dias_recencia = colp1.slider("Ventana para 'comprador reciente' (días)", 7, 120, 30)
        top_k_sugerencias = colp2.slider("Nº de sugerencias a mostrar", 5, 30, 10)
        usar_top_productos = colp3.checkbox("Usar señales de productos (Top 10)", value=True)

        st.caption("Entrena Logistic, RandomForest y XGBoost* (si no está, usa GradientBoosting). Compara Accuracy/F1/AUC y sugiere Top-N clientes (asignados a Camila/Andrea).")

        cols_necesarias = {'Cliente/Empresa', 'FECHA VENTA', 'Total'}
        if not cols_necesarias.issubset(df_ventas.columns):
            st.warning(f"Faltan columnas para RFM/ML. Se requieren: {cols_necesarias}.")
        else:
            ejecutar = st.button("🚀 Ejecutar RFM + Modelos y Generar Sugerencias")
            if ejecutar:
                with st.spinner("Procesando RFM, creando features y entrenando modelos..."):
                    ventas = df_ventas.copy()
                    ventas['FECHA VENTA'] = pd.to_datetime(ventas['FECHA VENTA'], errors="coerce")
                    ventas = ventas.dropna(subset=['FECHA VENTA'])
                    ref_date = ventas['FECHA VENTA'].max()
                    tiene_factura = 'NÚMERO DE FACTURA' in ventas.columns

                    # RFM
                    rfm = ventas.groupby('Cliente/Empresa').agg(
                        Recencia=('FECHA VENTA', lambda s: (ref_date - s.max()).days),
                        Frecuencia=('NÚMERO DE FACTURA', 'nunique') if tiene_factura else ('FECHA VENTA','count'),
                        Monetario=('Total', 'sum')
                    ).reset_index()

                    # Features día/hora
                    ventas['DiaSemana'] = ventas['FECHA VENTA'].dt.dayofweek
                    ventas['Hora'] = ventas['FECHA VENTA'].dt.hour
                    feats_dia = ventas.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                    feats_dia.columns = [f"dw_{int(c)}" for c in feats_dia.columns]
                    feats_hora = ventas.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                    feats_hora.columns = [f"h_{int(c)}" for c in feats_hora.columns]

                    def row_norm(df_in):
                        sums = df_in.sum(axis=1).replace(0, 1)
                        return df_in.div(sums, axis=0)

                    feats_dia = row_norm(feats_dia)
                    feats_hora = row_norm(feats_hora)

                    # Productos Top10 (opcional)
                    feats_prod = None
                    if usar_top_productos and 'Producto_Nombre' in ventas.columns:
                        top10_prod = (ventas.groupby('Producto_Nombre')['Total'].sum()
                                      .sort_values(ascending=False).head(10).index.tolist())
                        v_prod = ventas[ventas['Producto_Nombre'].isin(top10_prod)].copy()
                        feats_prod = (v_prod.groupby(['Cliente/Empresa','Producto_Nombre'])
                                      .size().unstack(fill_value=0))
                        feats_prod = row_norm(feats_prod)

                    # Merge de features
                    df_feat = rfm.merge(feats_dia, on='Cliente/Empresa', how='left') \
                                 .merge(feats_hora, on='Cliente/Empresa', how='left')
                    if feats_prod is not None:
                        df_feat = df_feat.merge(feats_prod, on='Cliente/Empresa', how='left')

                    df_feat = df_feat.fillna(0)

                    # Target reciente
                    recientes = ventas[ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=dias_recencia)]['Cliente/Empresa'].unique()
                    df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                    X = df_feat.drop(columns=['Cliente/Empresa','comprador_reciente'])
                    y = df_feat['comprador_reciente']

                    if y.nunique() < 2:
                        st.warning("La variable objetivo tiene una sola clase en esta ventana. Ajusta la ventana de recencia o revisa datos.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

                        modelos = {
                            "LogisticRegression": LogisticRegression(max_iter=800, class_weight='balanced'),
                            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
                        }
                        if HAS_XGB:
                            modelos["XGBoost"] = XGBClassifier(
                                n_estimators=300, learning_rate=0.06, max_depth=4,
                                subsample=0.9, colsample_bytree=0.9,
                                reg_lambda=1.0, random_state=42, eval_metric='logloss', tree_method="hist"
                            )
                        else:
                            modelos["GradientBoosting"] = GradientBoostingClassifier(random_state=42)

                        resultados = []
                        mejor_modelo = None
                        mejor_score_sel = -1
                        mejores_probs_full = None

                        for nombre, modelo in modelos.items():
                            modelo.fit(X_train, y_train)

                            if hasattr(modelo, "predict_proba"):
                                y_prob = modelo.predict_proba(X_test)[:, 1]
                                y_prob_full = modelo.predict_proba(X)[:, 1]
                            elif hasattr(modelo, "decision_function"):
                                s = modelo.decision_function(X_test)
                                s_full = modelo.decision_function(X)
                                y_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
                                y_prob_full = (s_full - s_full.min()) / (s_full.max() - s_full.min() + 1e-9)
                            else:
                                y_prob = modelo.predict(X_test)
                                y_prob_full = modelo.predict(X)

                            y_pred = (y_prob >= 0.5).astype(int)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            try:
                                auc = roc_auc_score(y_test, y_prob)
                            except ValueError:
                                auc = float("nan")

                            resultados.append({"Modelo": nombre, "Accuracy": acc, "F1": f1, "AUC": auc})

                            score_sel = (auc if not pd.isna(auc) else f1)
                            if score_sel > mejor_score_sel:
                                mejor_score_sel = score_sel
                                mejor_modelo = nombre
                                mejores_probs_full = y_prob_full

                        st.subheader("Comparación de Modelos")
                        st.dataframe(pd.DataFrame(resultados).round(3), use_container_width=True)
                        st.success(f"🏆 Mejor modelo: **{mejor_modelo}**")

                        # Sugerencias
                        df_feat['Prob_Compra'] = mejores_probs_full
                        candidatos = df_feat[df_feat['comprador_reciente'] == 0].copy()
                        if candidatos.empty:
                            st.info("No hay candidatos (todos compraron recientemente). Ajusta la ventana.")
                        else:
                            # Producto sugerido por cliente
                            if 'Producto_Nombre' in ventas.columns and not ventas['Producto_Nombre'].isna().all():
                                top_prod_cliente = (ventas.groupby(['Cliente/Empresa', 'Producto_Nombre'])['Total']
                                                    .sum().reset_index())
                                idx = top_prod_cliente.groupby('Cliente/Empresa')['Total'].idxmax()
                                top_prod_cliente = top_prod_cliente.loc[idx][['Cliente/Empresa', 'Producto_Nombre']]
                                top_prod_cliente = top_prod_cliente.rename(columns={'Producto_Nombre':'Producto_Sugerido'})
                                candidatos = candidatos.merge(top_prod_cliente, on='Cliente/Empresa', how='left')
                            else:
                                candidatos['Producto_Sugerido'] = None

                            # Mejor día/hora por proporción histórica
                            dia_cols = [c for c in candidatos.columns if c.startswith("dw_")]
                            hora_cols = [c for c in candidatos.columns if c.startswith("h_")]

                            def mejor_idx(row, pref):
                                cols = [c for c in row.index if c.startswith(pref)]
                                if not cols:
                                    return None
                                sub = row[cols]
                                if (sub.max() == 0) or sub.isna().all():
                                    return None
                                return cols[sub.argmax()]

                            candidatos['mejor_dw'] = candidatos.apply(lambda r: mejor_idx(r, "dw_"), axis=1)
                            candidatos['mejor_h']  = candidatos.apply(lambda r: mejor_idx(r, "h_"), axis=1)
                            mapa_dw = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
                            candidatos['Dia_Contacto'] = candidatos['mejor_dw'].str.replace("dw_","",regex=False).astype(float).map(mapa_dw)
                            candidatos['Hora_Contacto'] = candidatos['mejor_h'].str.replace("h_","",regex=False).astype(float).fillna(10).astype(int)

                            topN = candidatos.nlargest(top_k_sugerencias, 'Prob_Compra')[
                                ['Cliente/Empresa','Prob_Compra','Producto_Sugerido','Dia_Contacto','Hora_Contacto']
                            ].copy()

                            asignaciones = (["Camila", "Andrea"] * ((len(topN)//2)+1))[:len(topN)]
                            topN['Asignado_a'] = asignaciones

                            st.subheader("🎯 Top clientes potenciales a contactar")
                            st.dataframe(
                                topN.rename(columns={'Cliente/Empresa':'Cliente','Prob_Compra':'Probabilidad_Compra'})
                                    .style.format({'Probabilidad_Compra':'{:.1%}'}),
                                use_container_width=True
                            )

                            st.download_button(
                                "⬇️ Descargar sugerencias (CSV)",
                                data=topN.to_csv(index=False).encode('utf-8'),
                                file_name=f"sugerencias_rfm_ml_{pd.Timestamp.today().date()}.csv",
                                mime="text/csv"
                            )

    # ---------------------------------------------------------------------------------
    # Pestaña 4: Clientes Potenciales (no compradores)
    # ---------------------------------------------------------------------------------
    with tab4:
        st.header("Identificación de Clientes Potenciales (No Compradores)")
        if 'Cliente/Empresa' in df_ventas.columns and 'NOMBRE' in df_medicos.columns:
            medicos_compradores = df_ventas['Cliente/Empresa'].unique()
            df_medicos_potenciales = df_medicos[~df_medicos['NOMBRE'].isin(medicos_compradores)]
            st.info(f"Se encontraron **{len(df_medicos_potenciales)}** médicos que aún no han comprado.")
            if 'ESPECIALIDAD MEDICA' in df_medicos_potenciales.columns:
                especialidades = sorted(df_medicos_potenciales['ESPECIALIDAD MEDICA'].dropna().unique())
                selected_especialidad = st.selectbox("Filtrar por Especialidad:", options=['Todas'] + especialidades)
                if selected_especialidad != 'Todas':
                    df_display = df_medicos_potenciales[df_medicos_potenciales['ESPECIALIDAD MEDICA'] == selected_especialidad]
                else:
                    df_display = df_medicos_potenciales
                cols_disp = [c for c in ['NOMBRE','ESPECIALIDAD MEDICA','TELEFONO','EMAIL','CIUDAD'] if c in df_display.columns]
                st.dataframe(df_display[cols_disp] if cols_disp else df_display, use_container_width=True)
            else:
                st.dataframe(df_medicos_potenciales, use_container_width=True)
        else:
            st.warning("Faltan columnas para cruzar compradores con la lista de médicos.")

    # ---------------------------------------------------------------------------------
    # Pestaña 5: Predicción de Compradores (demo)
    # ---------------------------------------------------------------------------------
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales (Demo)")

        if 'Producto_Nombre' in df_ventas.columns:
            st.subheader("1) Buscar Potenciales por Producto")
            producto_a_predecir = st.selectbox("Producto:", options=sorted(df_ventas['Producto_Nombre'].unique()))
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
                        }),
                        use_container_width=True
                    )
        else:
            st.info("No hay columna 'Producto_Nombre' para esta predicción.")

        st.markdown("---")
        st.subheader("2) Generador de Tareas Diarias (Top 6 Médicos)")
        st.info("Genera 6 médicos con alta probabilidad (demo), productos recomendados y asignación a comerciales.")
        if st.button("Generar Lista de Tareas (Demo)"):
            with st.spinner("Ejecutando modelo demo..."):
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
                        dias_desde_ultima_compra=('FECHA VENTA', lambda d: (df_ventas['FECHA VENTA'].max() - d.max()).days) if 'FECHA VENTA' in df_ventas.columns else ('Producto_Nombre','count')
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

                        st.success("¡Lista generada!")
                        for medico in top_6_medicos:
                            st.markdown(f"#### Médico: **{medico}**")
                            st.markdown(f"**Asignado a:** `{asignaciones[medico]}`")
                            productos_recomendados = resultados_full[resultados_full['Cliente/Empresa'] == medico].nlargest(3, 'Probabilidad')
                            st.write("**Productos recomendados:**")
                            for _, row in productos_recomendados.iterrows():
                                st.markdown(f"- {row['Producto_Nombre']} *(Prob: {row['Probabilidad']:.1%})*")
                            st.markdown("---")
                else:
                    st.warning("Faltan columnas (Cliente/Empresa, Producto_Nombre).")

else:
    st.warning("No se pudieron cargar los datos.")
