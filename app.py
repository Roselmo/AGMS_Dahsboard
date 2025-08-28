# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
    tab_list = ["Análisis de Ventas", "Gestión de Cartera", "Análisis RFM", "Clientes Potenciales", "Predicción (Demo)"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    # ---------------------------------------------------------------------------------
    # Pestaña 1: Análisis de Ventas
    # ---------------------------------------------------------------------------------
    with tab1:
        st.header("Análisis General de Ventas")
        fecha_col = None
        for c in ["Fecha", "FECHA_VENTA", "FECHA VENTA"]:
            if c in df_filtrado.columns:
                fecha_col = c
                break
        if "Mes" not in df_filtrado.columns and fecha_col:
            df_filtrado["Mes"] = pd.to_datetime(df_filtrado[fecha_col], errors="coerce").dt.to_period("M").astype(str)

        granularidad = st.selectbox("Granularidad", options=["Mes", "Semana", "Día"], index=0, key="sel_gran_tab1")
        if fecha_col:
            dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
            if "Semana" not in df_filtrado.columns:
                df_filtrado["Semana"] = dt.dt.to_period("W").astype(str)
            if "Día" not in df_filtrado.columns:
                df_filtrado["Día"] = dt.dt.date

        dim_posibles = [c for c in ["Producto_Nombre", "Cliente/Empresa", "Comercial"] if c in df_filtrado.columns]
        dimension = st.selectbox("Dimensión para Top-N", options=dim_posibles if dim_posibles else ["(no disponible)"], index=0, key="sel_dim_tab1")
        top_n = st.slider("Top-N a mostrar", 5, 30, 10, key="sld_topn_tab1")

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

        tab_resumen, tab_series, tab_productos, tab_clientes, tab_pareto, tab_mapa = st.tabs(
            ["Resumen", "Series", "Productos", "Clientes", "Pareto", "Mapa de calor"]
        )

        with tab_resumen:
            a, b = st.columns(2)
            with a:
                st.subheader("Evolución temporal")
                eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
                if eje_tiempo in df_filtrado.columns:
                    serie = (df_filtrado.groupby(eje_tiempo, as_index=False)["Total"].sum().sort_values(eje_tiempo))
                    st.plotly_chart(px.line(serie, x=eje_tiempo, y="Total", markers=True, title=f"Ventas por {granularidad}"),
                                    use_container_width=True, key="ch_tab1_resumen_line")
            with b:
                if dimension in df_filtrado.columns:
                    top_df = (df_filtrado.groupby(dimension, as_index=False)["Total"].sum()
                              .sort_values("Total", ascending=False).head(top_n))
                    fig = px.bar(top_df, x="Total", y=dimension, orientation="h", title=f"Top {top_n} por {dimension}")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True, key="ch_tab1_resumen_topn")
                    st.dataframe(top_df, use_container_width=True)

        with tab_series:
            ventana = st.slider("Ventana SMA", 1, 12, 3, key="sld_tab1_sma")
            eje_tiempo = {"Mes": "Mes", "Semana": "Semana", "Día": "Día"}[granularidad]
            if eje_tiempo in df_filtrado.columns:
                serie = (df_filtrado.groupby(eje_tiempo, as_index=False)["Total"].sum().sort_values(eje_tiempo))
                serie["SMA"] = serie["Total"].rolling(ventana, min_periods=1).mean()
                st.plotly_chart(px.line(serie, x=eje_tiempo, y=["Total","SMA"], markers=True, title=f"Ventas vs SMA ({ventana})"),
                                use_container_width=True, key="ch_tab1_series_sma")
                st.dataframe(serie, use_container_width=True)

        with tab_productos:
            if "Producto_Nombre" in df_filtrado.columns:
                prod = (df_filtrado.groupby("Producto_Nombre", as_index=False)["Total"].sum().sort_values("Total", ascending=False))
                total_prod = prod["Total"].sum()
                prod["%_participación"] = 100 * prod["Total"] / total_prod if total_prod else 0
                top_prod = prod.head(top_n)
                cA, cB = st.columns(2)
                with cA:
                    fig = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h", title=f"Top {top_n} Productos")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True, key="ch_tab1_prod_bar")
                with cB:
                    st.plotly_chart(px.treemap(prod, path=["Producto_Nombre"], values="Total", title="Participación"),
                                    use_container_width=True, key="ch_tab1_prod_treemap")
                st.dataframe(top_prod, use_container_width=True)

        with tab_clientes:
            if "Cliente/Empresa" in df_filtrado.columns:
                cli = (df_filtrado.groupby("Cliente/Empresa", as_index=False)["Total"].sum().sort_values("Total", ascending=False))
                top_cli = cli.head(top_n)
                fig = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h", title=f"Top {top_n} Clientes")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="ch_tab1_clientes_bar")
                st.dataframe(top_cli, use_container_width=True)

        with tab_pareto:
            if dimension in df_filtrado.columns:
                base = (df_filtrado.groupby(dimension, as_index=False)["Total"].sum().sort_values("Total", ascending=False))
                total_base = base["Total"].sum()
                base["%_acum"] = 100 * base["Total"].cumsum() / total_base if total_base else 0
                fig = px.bar(base, x=dimension, y="Total", title="Pareto")
                fig2 = px.line(base, x=dimension, y="%_acum")
                for tr in fig2.data:
                    fig.add_trace(tr)
                st.plotly_chart(fig, use_container_width=True, key="ch_tab1_pareto")
                st.dataframe(base, use_container_width=True)

        with tab_mapa:
            if fecha_col:
                dt = pd.to_datetime(df_filtrado[fecha_col], errors="coerce")
                work = df_filtrado.copy()
                work["Mes"] = work["Mes"] if "Mes" in work.columns else dt.dt.to_period("M").astype(str)
                work["DiaSemana"] = dt.dt.day_name()
                heat = work.groupby(["DiaSemana","Mes"], as_index=False)["Total"].sum()
                orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                orden_dias_es = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
                if heat["DiaSemana"].str.lower().isin(orden_dias_es).any():
                    heat["DiaSemana"] = heat["DiaSemana"].str.lower()
                    cat_order = orden_dias_es
                else:
                    cat_order = orden_dias
                heat["DiaSemana"] = pd.Categorical(heat["DiaSemana"], categories=cat_order, ordered=True)
                heat = heat.pivot(index="DiaSemana", columns="Mes", values="Total").fillna(0)
                st.plotly_chart(px.imshow(heat, aspect="auto", title="Heatmap (Día x Mes)"),
                                use_container_width=True, key="ch_tab1_heatmap")

    # ---------------------------------------------------------------------------------
    # Pestaña 2: Cartera
    # ---------------------------------------------------------------------------------
    with tab2:
        st.header("Gestión de Cartera")
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

        filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada'], key="sel_estado_car")
        lista_clientes_cartera = sorted(df_cartera_proc['Nombre cliente'].dropna().unique()) if 'Nombre cliente' in df_cartera_proc.columns else []
        filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera, key="ms_car_cliente")

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
            .format({'Saldo pendiente': '${:,.0f}'}) if cols_show else pd.DataFrame(),
            use_container_width=True
        )

        st.markdown("---")
        if {'Fecha de Vencimiento','Saldo pendiente'}.issubset(df_cartera.columns):
            car = df_cartera[['Fecha de Vencimiento','Saldo pendiente']].copy()
            car['DIAS_VENCIDOS'] = (pd.Timestamp.today().normalize() - car['Fecha de Vencimiento']).dt.days
            labels = ["Al día", "1-30", "31-60", "61-90", "91-180", "181-365", "+365"]
            bins = [-float("inf"), 0, 30, 60, 90, 180, 365, float("inf")]
            car["Rango"] = pd.cut(car["DIAS_VENCIDOS"], bins=bins, labels=labels, ordered=True)
            venc = car.groupby("Rango", as_index=False).agg(Saldo=("Saldo pendiente","sum"))
            st.plotly_chart(px.bar(venc, x="Rango", y="Saldo", title="Antigüedad de saldos"),
                            use_container_width=True, key="ch_tab2_aged_balance")

    # ---------------------------------------------------------------------------------
    # Pestaña 3: RFM + Recomendador ML (refinado)
    # ---------------------------------------------------------------------------------
    with tab3:
        st.header("Análisis RFM + Recomendador ML (refinado)")

        colp1, colp2, colp3, colp4 = st.columns(4)
        dias_recencia = colp1.slider("Ventana para 'comprador reciente' (días)", 7, 120, 30, key="sld_rfm_recencia")
        top_k_sugerencias = colp2.slider("Nº de sugerencias a mostrar", 5, 30, 10, key="sld_rfm_topN")
        usar_top_productos = colp3.checkbox("Usar señales de productos (Top 10)", value=True, key="chk_rfm_topprod")
        excluir_recencia = colp4.checkbox("Excluir 'Recencia' como feature", value=True, key="chk_rfm_excluir_rec")

        dias_op = ["(Todos)","Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
        dia_reporte = st.selectbox("Día deseado para el reporte de candidatos", dias_op, index=0, key="sel_rfm_dia")

        st.caption("Validación con Stratified 5-Fold. Selección del mejor modelo por AUC promedio (fallback F1).")

        cols_necesarias = {'Cliente/Empresa', 'FECHA VENTA', 'Total'}
        if not cols_necesarias.issubset(df_ventas.columns):
            st.warning(f"Faltan columnas para RFM/ML. Se requieren: {cols_necesarias}.")
        else:
            ejecutar = st.button("🚀 Ejecutar RFM + Entrenar y Comparar Modelos", key="btn_rfm_run")
            if ejecutar:
                with st.spinner("Procesando..."):
                    ventas = df_ventas.copy()
                    ventas['FECHA VENTA'] = pd.to_datetime(ventas['FECHA VENTA'], errors="coerce")
                    ventas = ventas.dropna(subset=['FECHA VENTA'])
                    ref_date = ventas['FECHA VENTA'].max()
                    tiene_factura = 'NÚMERO DE FACTURA' in ventas.columns

                    # -------- 1) RFM
                    rfm = ventas.groupby('Cliente/Empresa').agg(
                        Recencia=('FECHA VENTA', lambda s: (ref_date - s.max()).days),
                        Frecuencia=('NÚMERO DE FACTURA', 'nunique') if tiene_factura else ('FECHA VENTA','count'),
                        Monetario=('Total', 'sum')
                    ).reset_index()

                    # Puntuación y Segmento RFM
                    rfm['R_Score'] = pd.qcut(rfm['Recencia'].rank(method='first', ascending=True), 5, labels=[5,4,3,2,1]).astype(int)
                    rfm['F_Score'] = pd.qcut(rfm['Frecuencia'].rank(method='first', ascending=False), 5, labels=[1,2,3,4,5]).astype(int)
                    rfm['M_Score'] = pd.qcut(rfm['Monetario'].rank(method='first', ascending=False), 5, labels=[1,2,3,4,5]).astype(int)

                    def rfm_segment(row):
                        r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
                        if r>=4 and f>=4 and m>=4: return "Champions"
                        if r>=4 and f>=3: return "Loyal"
                        if r>=3 and f>=3 and m>=3: return "Potential Loyalist"
                        if r<=2 and f>=4: return "At Risk"
                        if r<=2 and f<=2 and m<=2: return "Hibernating"
                        if r>=3 and f<=2: return "New"
                        return "Need Attention"
                    rfm['Segmento'] = rfm.apply(rfm_segment, axis=1)

                    # -------- 2) Features de comportamiento
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

                    feats_prod = None
                    if usar_top_productos and 'Producto_Nombre' in ventas.columns:
                        top10_prod = (ventas.groupby('Producto_Nombre')['Total'].sum()
                                      .sort_values(ascending=False).head(10).index.tolist())
                        v_prod = ventas[ventas['Producto_Nombre'].isin(top10_prod)].copy()
                        feats_prod = (v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0))
                        feats_prod = row_norm(feats_prod)

                    # Merge de features + RFM
                    df_feat = rfm.merge(feats_dia, on='Cliente/Empresa', how='left') \
                                 .merge(feats_hora, on='Cliente/Empresa', how='left')
                    if feats_prod is not None:
                        df_feat = df_feat.merge(feats_prod, on='Cliente/Empresa', how='left')
                    df_feat = df_feat.fillna(0)

                    # -------- 3) Target: compró en últimos N días
                    recientes = ventas[ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=dias_recencia)]['Cliente/Empresa'].unique()
                    df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                    # Armar X/y (opción: excluir Recencia)
                    feature_cols_base = ['Frecuencia','Monetario'] + [c for c in df_feat.columns if c.startswith('dw_') or c.startswith('h_')]
                    if feats_prod is not None:
                        feature_cols_base += [c for c in df_feat.columns if c in feats_prod.columns]
                    if not excluir_recencia:
                        feature_cols_base = ['Recencia'] + feature_cols_base

                    X = df_feat[feature_cols_base]
                    y = df_feat['comprador_reciente']

                    if y.nunique() < 2:
                        st.warning("La variable objetivo tiene una sola clase en esta ventana. Ajusta la ventana o revisa datos.")
                    else:
                        # -------- 4) Modelos + CV (5-fold)
                        modelos = {
                            "LogisticRegression": LogisticRegression(max_iter=800, C=0.3, penalty="l2", class_weight='balanced'),
                            "RandomForest": RandomForestClassifier(
                                n_estimators=250, max_depth=6, min_samples_leaf=10,
                                random_state=42, class_weight='balanced', n_jobs=-1
                            ),
                        }
                        if HAS_XGB:
                            modelos["XGBoost"] = XGBClassifier(
                                n_estimators=350, learning_rate=0.06, max_depth=4,
                                min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                                reg_lambda=1.2, random_state=42, eval_metric='logloss', tree_method="hist"
                            )
                        else:
                            modelos["GradientBoosting"] = GradientBoostingClassifier(
                                n_estimators=300, learning_rate=0.06, max_depth=3, min_samples_leaf=20, random_state=42
                            )

                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        resultados = []
                        for nombre, modelo in modelos.items():
                            cv_res = cross_validate(modelo, X, y, cv=cv,
                                                    scoring={'accuracy':'accuracy','f1':'f1','roc_auc':'roc_auc'},
                                                    n_jobs=-1, return_estimator=False)
                            resultados.append({
                                "Modelo": nombre,
                                "Accuracy": f"{cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}",
                                "F1":       f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
                                "AUC":      f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
                                "_auc_mean": cv_res['test_roc_auc'].mean(),
                                "_f1_mean":  cv_res['test_f1'].mean()
                            })

                        df_res = pd.DataFrame(resultados).sort_values(by=["_auc_mean","_f1_mean"], ascending=False)
                        mejor_modelo_nombre = df_res.iloc[0]["Modelo"]
                        st.subheader("Comparación de Modelos (5-Fold CV)")
                        st.dataframe(df_res.drop(columns=["_auc_mean","_f1_mean"]), use_container_width=True)
                        st.success(f"🏆 Mejor modelo: **{mejor_modelo_nombre}**")

                        # Entrenar mejor modelo en todo X/y para probabilidades actuales
                        best_model = modelos[mejor_modelo_nombre]
                        best_model.fit(X, y)
                        if hasattr(best_model, "predict_proba"):
                            probs_full = best_model.predict_proba(X)[:,1]
                        elif hasattr(best_model, "decision_function"):
                            s_full = best_model.decision_function(X)
                            probs_full = (s_full - s_full.min()) / (s_full.max() - s_full.min() + 1e-9)
                        else:
                            probs_full = best_model.predict(X)

                        df_feat['Prob_Compra'] = probs_full

                        # -------- 5) Sugerencias
                        candidatos = df_feat[df_feat['comprador_reciente'] == 0].copy()

                        # Mejor día histórico (dw_*)
                        dia_cols = [c for c in candidatos.columns if c.startswith("dw_")]
                        def mejor_dia(row):
                            if not dia_cols: return None
                            sub = row[dia_cols]
                            if (sub.max() == 0) or sub.isna().all(): return None
                            idx = int(sub.idxmax().split("_")[1])
                            mapa_dw = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
                            return mapa_dw.get(idx)
                        candidatos['Dia_Contacto'] = candidatos.apply(mejor_dia, axis=1)

                        # Producto sugerido
                        if 'Producto_Nombre' in ventas.columns and not ventas['Producto_Nombre'].isna().all():
                            top_prod_cliente = (ventas.groupby(['Cliente/Empresa', 'Producto_Nombre'])['Total']
                                                .sum().reset_index())
                            idx = top_prod_cliente.groupby('Cliente/Empresa')['Total'].idxmax()
                            top_prod_cliente = top_prod_cliente.loc[idx][['Cliente/Empresa', 'Producto_Nombre']] \
                                                               .rename(columns={'Producto_Nombre':'Producto_Sugerido'})
                            candidatos = candidatos.merge(top_prod_cliente, on='Cliente/Empresa', how='left')
                        else:
                            candidatos['Producto_Sugerido'] = None

                        # Añadir Segmento RFM
                        candidatos = candidatos.merge(rfm[['Cliente/Empresa','Segmento']], on='Cliente/Empresa', how='left')

                        # Filtros por día y segmento
                        if dia_reporte != "(Todos)":
                            candidatos = candidatos[candidatos['Dia_Contacto'] == dia_reporte]
                        segs = sorted(candidatos['Segmento'].dropna().unique().tolist())
                        seg_sel = st.multiselect("Filtrar por Segmento RFM", options=segs, default=segs, key="ms_rfm_segmento")
                        if seg_sel:
                            candidatos = candidatos[candidatos['Segmento'].isin(seg_sel)]

                        if candidatos.empty:
                            st.info("No hay candidatos que cumplan los filtros seleccionados.")
                        else:
                            topN = candidatos.nlargest(top_k_sugerencias, 'Prob_Compra')[
                                ['Cliente/Empresa','Prob_Compra','Producto_Sugerido','Dia_Contacto','Segmento']
                            ].copy()

                            # Asignación balanceada
                            asignaciones = (["Camila", "Andrea"] * ((len(topN)//2)+1))[:len(topN)]
                            topN['Asignado_a'] = asignaciones

                            st.subheader("🎯 Top clientes potenciales a contactar")
                            st.dataframe(
                                topN.rename(columns={
                                    'Cliente/Empresa':'Cliente',
                                    'Prob_Compra':'Probabilidad_Compra'
                                }).style.format({'Probabilidad_Compra':'{:.1%}'}),
                                use_container_width=True
                            )

                            st.download_button(
                                "⬇️ Descargar sugerencias (CSV)",
                                data=topN.to_csv(index=False).encode('utf-8'),
                                file_name=f"sugerencias_rfm_ml_{pd.Timestamp.today().date()}.csv",
                                mime="text/csv",
                                key="dl_rfm_csv"
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
                selected_especialidad = st.selectbox("Filtrar por Especialidad:", options=['Todas'] + especialidades, key="sel_tab4_esp")
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
    # Pestaña 5: Predicción (Demo)
    # ---------------------------------------------------------------------------------
    with tab5:
        st.header("Modelo Predictivo de Compradores Potenciales (Demo)")
        if 'Producto_Nombre' in df_ventas.columns:
            producto_a_predecir = st.selectbox("Producto:", options=sorted(df_ventas['Producto_Nombre'].unique()), key="sel_tab5_prod")
            if st.button("Buscar Compradores (Demo)", key="btn_tab5_demo"):
                with st.spinner("Entrenando modelo básico..."):
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
                    random.seed(42)
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
    st.warning("No se pudieron cargar los datos.")
