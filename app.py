# ==============================================================================
# APP: Dashboard AGMS – Ventas, Cartera, RFM, Predicción y Agente de Análisis
# ==============================================================================
# Requisitos sugeridos (requirements.txt):
# streamlit, pandas, numpy, plotly, scikit-learn, openai (opcional), xgboost (opcional)
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="📊", layout="wide")
st.title("📊 Dashboard AGMS: Ventas, Cartera, RFM, Predicción y Agente")
st.markdown("---")

# ==============================================================================
# UTILIDADES
# ==============================================================================
def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

def build_time_derivatives(df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[fecha_col], errors="coerce")
    if "Mes" not in df.columns:
        df["Mes"] = dt.dt.to_period("M").astype(str)
    if "Semana" not in df.columns:
        df["Semana"] = dt.dt.to_period("W").astype(str)
    if "Día" not in df.columns:
        df["Día"] = dt.dt.date
    return df

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

# ===== RFM helpers =====
def _safe_qcut_score(series, ascending, labels=[1,2,3,4,5]):
    s = series.copy()
    rk = s.rank(method='first', ascending=ascending)
    try:
        q = pd.qcut(rk, 5, labels=labels)
        return q.astype(int)
    except Exception:
        q = pd.cut(rk, bins=5, labels=labels, include_lowest=True, duplicates='drop')
        q = q.astype('float')
        fill_val = np.ceil(q.mean()) if not np.isnan(q.mean()) else 3
        return q.fillna(fill_val).astype(int)

def rfm_segment(row):
    r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
    if r>=4 and f>=4 and m>=4: return "Champions"
    if r>=4 and f>=3: return "Loyal"
    if r>=3 and f>=3 and m>=3: return "Potential Loyalist"
    if r<=2 and f>=4: return "At Risk"
    if r<=2 and f<=2 and m<=2: return "Hibernating"
    if r>=3 and f<=2: return "New"
    return "Need Attention"

def compute_rfm_table(dfv: pd.DataFrame) -> pd.DataFrame:
    if 'FECHA VENTA' not in dfv.columns:
        return pd.DataFrame()
    tmp = dfv.copy()
    tmp['FECHA VENTA'] = pd.to_datetime(tmp['FECHA VENTA'], errors="coerce")
    tmp = tmp.dropna(subset=['FECHA VENTA'])
    ref = tmp['FECHA VENTA'].max()
    tiene_factura = ('NÚMERO DE FACTURA' in tmp.columns)

    rfm = tmp.groupby('Cliente/Empresa').agg(
        Recencia=('FECHA VENTA', lambda s: (ref - s.max()).days),
        Frecuencia=('NÚMERO DE FACTURA','nunique') if tiene_factura else ('FECHA VENTA','count'),
        Monetario=('Total','sum')
    ).reset_index()

    rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'],  ascending=True,  labels=[5,4,3,2,1])
    rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
    rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])
    rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")
    return rfm

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera = pd.read_excel(file_path, sheet_name='CarteraAgosto')

        # Ventas
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

        # Médicos
        if 'NOMBRE' in df_medicos.columns:
            df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
        if 'TELEFONO' in df_medicos.columns:
            df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # Cartera
        if 'Fecha de Vencimiento' in df_cartera.columns:
            df_cartera.dropna(subset=['Fecha de Vencimiento'], inplace=True)
            df_cartera['Fecha de Vencimiento'] = pd.to_datetime(df_cartera['Fecha de Vencimiento'], errors='coerce')
        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in df_cartera.columns:
                df_cartera[col] = df_cartera[col].fillna(0).apply(limpiar_moneda)

        return df_ventas, df_medicos, df_metadatos, df_cartera
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera = load_data()
if df_ventas is None or df_cartera is None:
    st.stop()

# ==============================================================================
# TABS PRINCIPALES
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Análisis de Ventas", "Gestión de Cartera", "Análisis RFM",
    "Modelo Predictivo de Compradores Potenciales", "Agente de Análisis"
])

# ---------------------------------------------------------------------------------
# TAB 1: ANÁLISIS DE VENTAS + Generador de Reporte Consolidado
# ---------------------------------------------------------------------------------
with tab1:
    st.header("Análisis General de Ventas")

    dfv = df_ventas.copy()
    fecha_col = None
    for c in ["Fecha", "FECHA_VENTA", "FECHA VENTA"]:
        if c in dfv.columns:
            fecha_col = c
            break
    if fecha_col:
        dfv = build_time_derivatives(dfv, fecha_col)

    granularidad = st.selectbox("Granularidad", options=["Mes", "Semana", "Día"], index=0, key="gran_t1")
    dim_posibles = [c for c in ["Producto_Nombre", "Cliente/Empresa", "Comercial"] if c in dfv.columns]
    dimension = st.selectbox("Dimensión para Top-N", options=dim_posibles if dim_posibles else ["(no disponible)"], index=0, key="dim_t1")
    top_n = st.slider("Top-N a mostrar", 5, 30, 10, key="topn_t1")

    total_ventas = float(dfv["Total"].sum()) if "Total" in dfv.columns else 0.0
    total_transacciones = len(dfv)
    clientes_unicos = dfv["Cliente/Empresa"].nunique() if "Cliente/Empresa" in dfv.columns else 0
    ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

    delta_ventas = None
    if "Mes" in dfv.columns:
        tmp = dfv.groupby("Mes", as_index=False)["Total"].sum().sort_values("Mes")
        if len(tmp) >= 2:
            delta_ventas = tmp["Total"].iloc[-1] - tmp["Total"].iloc[-2]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventas Totales", f"${total_ventas:,.0f}", delta=(f"{delta_ventas:,.0f}" if delta_ventas is not None else None))
    c2.metric("Transacciones", f"{total_transacciones:,}")
    c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
    st.markdown("---")

    tab_r1, tab_r2, tab_r3, tab_r4, tab_r5, tab_r6 = st.tabs(
        ["Resumen", "Series", "Productos", "Clientes", "Pareto", "Mapa de calor"]
    )

    with tab_r1:
        a, b = st.columns(2)
        with a:
            st.subheader("Evolución temporal")
            eje = {"Mes":"Mes","Semana":"Semana","Día":"Día"}[granularidad]
            if eje in dfv.columns:
                serie = dfv.groupby(eje, as_index=False)["Total"].sum().sort_values(eje)
                st.plotly_chart(px.line(serie, x=eje, y="Total", markers=True, title=f"Ventas por {granularidad}"),
                                use_container_width=True, key="t1_res_line")
        with b:
            if dimension in dfv.columns:
                top_df = (dfv.groupby(dimension, as_index=False)["Total"].sum()
                          .sort_values("Total", ascending=False).head(top_n))
                fig = px.bar(top_df, x="Total", y=dimension, orientation="h", title=f"Top {top_n} por {dimension}")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="t1_res_top")
                st.dataframe(top_df, use_container_width=True)

    with tab_r2:
        ventana = st.slider("Ventana SMA", 1, 12, 3, key="t1_sma")
        eje = {"Mes":"Mes","Semana":"Semana","Día":"Día"}[granularidad]
        if eje in dfv.columns:
            serie = dfv.groupby(eje, as_index=False)["Total"].sum().sort_values(eje)
            serie["SMA"] = serie["Total"].rolling(ventana, min_periods=1).mean()
            st.plotly_chart(px.line(serie, x=eje, y=["Total","SMA"], markers=True, title=f"Ventas vs SMA ({ventana})"),
                            use_container_width=True, key="t1_series")

    with tab_r3:
        if "Producto_Nombre" in dfv.columns:
            prod = dfv.groupby("Producto_Nombre", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
            total_prod = prod["Total"].sum()
            prod["%_participación"] = 100 * prod["Total"] / total_prod if total_prod else 0
            top_prod = prod.head(top_n)
            cA, cB = st.columns(2)
            with cA:
                fig = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h", title=f"Top {top_n} Productos")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="t1_prod_bar")
            with cB:
                st.plotly_chart(px.treemap(prod, path=["Producto_Nombre"], values="Total", title="Participación"),
                                use_container_width=True, key="t1_prod_treemap")
            st.dataframe(top_prod, use_container_width=True)

    with tab_r4:
        if "Cliente/Empresa" in dfv.columns:
            cli = dfv.groupby("Cliente/Empresa", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
            top_cli = cli.head(top_n)
            fig = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h", title=f"Top {top_n} Clientes")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True, key="t1_cli_bar")
            st.dataframe(top_cli, use_container_width=True)

    with tab_r5:
        if dimension in dfv.columns:
            base = dfv.groupby(dimension, as_index=False)["Total"].sum().sort_values("Total", ascending=False)
            total_base = base["Total"].sum()
            base["%_acum"] = 100 * base["Total"].cumsum() / total_base if total_base else 0
            fig = px.bar(base, x=dimension, y="Total", title="Pareto")
            fig2 = px.line(base, x=dimension, y="%_acum")
            for tr in fig2.data:
                fig.add_trace(tr)
            st.plotly_chart(fig, use_container_width=True, key="t1_pareto")
            st.dataframe(base, use_container_width=True)

    with tab_r6:
        if fecha_col:
            dt = pd.to_datetime(dfv[fecha_col], errors="coerce")
            work = dfv.copy()
            work["Mes"] = work["Mes"] if "Mes" in work.columns else dt.dt.to_period("M").astype(str)
            work["DiaSemana"] = dt.dt.day_name()
            heat = work.groupby(["DiaSemana","Mes"], as_index=False)["Total"].sum()
            orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat["DiaSemana"] = pd.Categorical(heat["DiaSemana"], categories=orden_dias, ordered=True)
            heat = heat.pivot(index="DiaSemana", columns="Mes", values="Total").fillna(0)
            st.plotly_chart(px.imshow(heat, aspect="auto", title="Heatmap (Día x Mes)"),
                            use_container_width=True, key="t1_heatmap")

    # ====== Generador de Reporte Único (EDA + Cartera + RFM) ======
    def build_agms_report(dfv: pd.DataFrame, dfc: pd.DataFrame) -> str:
        parts = []
        # EDA
        if not dfv.empty:
            total = float(dfv['Total'].sum()) if 'Total' in dfv.columns else 0.0
            clientes = dfv['Cliente/Empresa'].nunique() if 'Cliente/Empresa' in dfv.columns else 0
            ventas_mes = (dfv.groupby('Mes')['Total'].sum().sort_index()
                          if 'Mes' in dfv.columns else pd.Series(dtype=float))
            top_prod = (dfv.groupby('Producto_Nombre')['Total'].sum()
                        .sort_values(ascending=False).head(10)
                        if 'Producto_Nombre' in dfv.columns else pd.Series(dtype=float))
            eda_txt = [ "# EDA (Ventas)",
                        f"- Ventas totales: ${total:,.0f}",
                        f"- Clientes únicos: {clientes}" ]
            if len(ventas_mes) > 0:
                eda_txt += [
                    f"- Mejor mes: {ventas_mes.idxmax()} (${ventas_mes.max():,.0f})",
                    f"- Peor mes: {ventas_mes.idxmin()} (${ventas_mes.min():,.0f})",
                ]
            if len(top_prod) > 0:
                top_list = "\n".join([f"    - {k}: ${v:,.0f}" for k, v in top_prod.items()])
                eda_txt += [f"- Top 10 productos:\n{top_list}"]
            parts.append("\n".join(eda_txt))
        # Cartera
        if not dfc.empty and {'Fecha de Vencimiento','Saldo pendiente'}.issubset(dfc.columns):
            hoy = pd.Timestamp.today()
            car = dfc.copy()
            car['Fecha de Vencimiento'] = pd.to_datetime(car['Fecha de Vencimiento'], errors='coerce')
            car['DIAS_VENCIDOS'] = (hoy.normalize() - car['Fecha de Vencimiento']).dt.days
            total_pend = float(car['Saldo pendiente'].sum())
            vencido    = float(car[car['DIAS_VENCIDOS']>0]['Saldo pendiente'].sum())
            por_vencer = float(car[car['DIAS_VENCIDOS']<=0]['Saldo pendiente'].sum())
            labels = ["Al día","1-30","31-60","61-90","91-180","181-365","+365"]
            bins   = [-np.inf,0,30,60,90,180,365,np.inf]
            car['Rango'] = pd.cut(car['DIAS_VENCIDOS'], bins=bins, labels=labels, ordered=True)
            bucket = car.groupby('Rango')['Saldo pendiente'].sum().to_dict()
            bucket_txt = "\n".join([f"    - {k}: ${v:,.0f}" for k,v in bucket.items()])
            parts.append("\n".join([
                "# Cartera",
                f"- Cartera total pendiente: ${total_pend:,.0f}",
                f"- Vencido: ${vencido:,.0f}",
                f"- Por vencer: ${por_vencer:,.0f}",
                f"- Antigüedad de saldos:\n{bucket_txt}"
            ]))
        # RFM
        rfm_tab = compute_rfm_table(dfv)
        if not rfm_tab.empty:
            dist = rfm_tab['Segmento'].value_counts().to_dict()
            dist_txt = "\n".join([f"    - {k}: {v}" for k,v in dist.items()])
            parts.append("\n".join([
                "# RFM",
                f"- Recencia media: {rfm_tab['Recencia'].mean():.1f} días",
                f"- Frecuencia media: {rfm_tab['Frecuencia'].mean():.2f}",
                f"- Monetario medio: ${rfm_tab['Monetario'].mean():,.0f}",
                f"- Distribución por segmento:\n{dist_txt}"
            ]))
        if not parts:
            return "# Reporte AGMS\n(No hay datos suficientes para generar el reporte)."
        return "# Reporte AGMS Consolidado (EDA + Cartera + RFM)\n\n" + "\n".join(parts)

    st.markdown("---")
    st.subheader("📄 Reporte consolidado para el Agente")
    colr1, colr2 = st.columns([1,1])
    with colr1:
        if st.button("Generar/Actualizar reporte (EDA + Cartera + RFM)", key="btn_make_report"):
            report_text = build_agms_report(dfv, df_cartera.copy())
            st.session_state["AGMS_REPORT"] = report_text
            st.success("Reporte generado y guardado en memoria (AGMS_REPORT).")
    with colr2:
        if "AGMS_REPORT" in st.session_state:
            st.download_button(
                "Descargar reporte (TXT)",
                data=st.session_state["AGMS_REPORT"].encode("utf-8"),
                file_name=f"Reporte_AGMS_{pd.Timestamp.today().date()}.txt",
                mime="text/plain",
                key="dl_report_txt"
            )
    if "AGMS_REPORT" in st.session_state:
        with st.expander("👁️ Vista previa del reporte (lo que leerá el agente)"):
            st.code(st.session_state["AGMS_REPORT"], language="markdown")
    else:
        st.info("Aún no hay reporte. Presiona el botón para generarlo.")

# ---------------------------------------------------------------------------------
# TAB 2: GESTIÓN DE CARTERA
# ---------------------------------------------------------------------------------
with tab2:
    st.header("Gestión de Cartera")
    dfc = df_cartera.copy()
    hoy = datetime.now()
    if 'Fecha de Vencimiento' in dfc.columns:
        dfc['Dias_Vencimiento'] = (dfc['Fecha de Vencimiento'] - hoy).dt.days
    else:
        dfc['Dias_Vencimiento'] = None

    def get_status(row):
        if 'Saldo pendiente' in row and row['Saldo pendiente'] <= 0:
            return 'Pagada'
        elif row['Dias_Vencimiento'] is not None and row['Dias_Vencimiento'] < 0:
            return 'Vencida'
        else:
            return 'Por Vencer'

    dfc['Estado'] = dfc.apply(get_status, axis=1)
    saldo_total = dfc[dfc['Estado'] != 'Pagada']['Saldo pendiente'].sum() if 'Saldo pendiente' in dfc.columns else 0
    saldo_vencido = dfc[dfc['Estado'] == 'Vencida']['Saldo pendiente'].sum() if 'Saldo pendiente' in dfc.columns else 0
    saldo_por_vencer = dfc[dfc['Estado'] == 'Por Vencer']['Saldo pendiente'].sum() if 'Saldo pendiente' in dfc.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
    c2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
    c3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
    st.markdown("---")

    filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada'], key="t2_estado")
    lista_clientes_cartera = sorted(dfc['Nombre cliente'].dropna().unique()) if 'Nombre cliente' in dfc.columns else []
    filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera, key="t2_cliente")

    dfc_filtrada = dfc.copy()
    if filtro_estado != 'Todas':
        dfc_filtrada = dfc_filtrada[dfc_filtrada['Estado'] == filtro_estado]
    if filtro_cliente and 'Nombre cliente' in dfc_filtrada.columns:
        dfc_filtrada = dfc_filtrada[dfc_filtrada['Nombre cliente'].isin(filtro_cliente)]

    def style_venc(row):
        if row['Estado'] == 'Vencida':
            return ['background-color: #ffcccc'] * len(row)
        elif isinstance(row['Dias_Vencimiento'], (int, float)) and 0 <= row['Dias_Vencimiento'] <= 7:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    cols_show = [c for c in ['Nombre cliente', 'NÚMERO DE FACTURA', 'Fecha de Vencimiento', 'Saldo pendiente', 'Estado', 'Dias_Vencimiento'] if c in dfc_filtrada.columns]
    st.dataframe(
        dfc_filtrada[cols_show].style.apply(style_venc, axis=1).format({'Saldo pendiente': '${:,.0f}'}) if cols_show else pd.DataFrame(),
        use_container_width=True
    )

    st.markdown("---")
    if {'Fecha de Vencimiento','Saldo pendiente'}.issubset(dfc.columns):
        car = dfc[['Fecha de Vencimiento','Saldo pendiente']].copy()
        car['DIAS_VENCIDOS'] = (pd.Timestamp.today().normalize() - car['Fecha de Vencimiento']).dt.days
        labels = ["Al día", "1-30", "31-60", "61-90", "91-180", "181-365", "+365"]
        bins = [-float("inf"), 0, 30, 60, 90, 180, 365, float("inf")]
        car["Rango"] = pd.cut(car["DIAS_VENCIDOS"], bins=bins, labels=labels, ordered=True)
        venc = car.groupby("Rango", as_index=False).agg(Saldo=("Saldo pendiente","sum"))
        st.plotly_chart(px.bar(venc, x="Rango", y="Saldo", title="Antigüedad de saldos"),
                        use_container_width=True, key="t2_aged")

# ---------------------------------------------------------------------------------
# TAB 3: ANÁLISIS RFM + Recomendador ML (segmentos & día)
# ---------------------------------------------------------------------------------
with tab3:
    st.header("Análisis RFM + Recomendador ML")

    colp1, colp2, colp3, colp4 = st.columns(4)
    dias_recencia = colp1.slider("Ventana 'comprador reciente' (días)", 7, 120, 30, key="t3_rec")
    top_k_sugerencias = colp2.slider("Nº sugerencias a mostrar", 5, 30, 10, key="t3_top")
    usar_top_productos = colp3.checkbox("Usar señales de productos (Top 10)", value=True, key="t3_topprod")
    excluir_recencia = colp4.checkbox("Excluir 'Recencia' como feature", value=True, key="t3_exrec")

    dias_op = ["(Todos)","Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    dia_reporte = st.selectbox("Día deseado del reporte", dias_op, index=0, key="t3_dia")

    cols_nec = {'Cliente/Empresa', 'FECHA VENTA', 'Total'}
    if not cols_nec.issubset(df_ventas.columns):
        st.warning(f"Faltan columnas para RFM/ML: {cols_nec}.")
    else:
        ejecutar = st.button("🚀 Ejecutar RFM + Entrenar y Comparar Modelos", key="t3_run")
        if ejecutar:
            with st.spinner("Procesando..."):
                ventas = df_ventas.copy()
                ventas['Cliente/Empresa'] = ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()
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
                rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'], ascending=True, labels=[5,4,3,2,1])
                rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
                rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])
                rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")

                st.caption("Distribución de segmentos RFM")
                st.dataframe(rfm['Segmento'].value_counts(dropna=False).rename_axis('Segmento').to_frame('Clientes'),
                             use_container_width=True)

                # Features comportamiento
                ventas['DiaSemana'] = ventas['FECHA VENTA'].dt.dayofweek
                ventas['Hora'] = ventas['FECHA VENTA'].dt.hour
                feats_dia  = ventas.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                feats_dia.columns  = [f"dw_{int(c)}" for c in feats_dia.columns]
                feats_hora = ventas.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                feats_hora.columns = [f"h_{int(c)}" for c in feats_hora.columns]
                feats_dia  = row_normalize(feats_dia)
                feats_hora = row_normalize(feats_hora)

                feats_prod = None
                if usar_top_productos and 'Producto_Nombre' in ventas.columns:
                    top10_prod = (ventas.groupby('Producto_Nombre')['Total'].sum()
                                  .sort_values(ascending=False).head(10).index.tolist())
                    v_prod = ventas[ventas['Producto_Nombre'].isin(top10_prod)].copy()
                    feats_prod = (v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0))
                    feats_prod = row_normalize(feats_prod)

                df_feat = rfm.merge(feats_dia, on='Cliente/Empresa', how='left') \
                             .merge(feats_hora, on='Cliente/Empresa', how='left')
                if feats_prod is not None:
                    df_feat = df_feat.merge(feats_prod, on='Cliente/Empresa', how='left')
                for c in df_feat.select_dtypes(include=[np.number]).columns:
                    df_feat[c] = df_feat[c].fillna(0)

                # Target comprador reciente
                recientes = ventas[ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=dias_recencia)]['Cliente/Empresa'].unique()
                df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                # Filtro de segmentos multi
                segmentos_all = sorted(df_feat['Segmento'].dropna().unique().tolist())
                seg_sel = st.multiselect("Filtrar por Segmento RFM (multi)", options=segmentos_all, default=segmentos_all, key="t3_seg")
                if seg_sel:
                    df_feat = df_feat[df_feat['Segmento'].isin(seg_sel)]

                feature_cols = ['Frecuencia','Monetario'] + [c for c in df_feat.columns if c.startswith('dw_') or c.startswith('h_')]
                if feats_prod is not None:
                    feature_cols += [c for c in df_feat.columns if c in feats_prod.columns]
                if not excluir_recencia:
                    feature_cols = ['Recencia'] + feature_cols
                X = df_feat[feature_cols]
                y = df_feat['comprador_reciente']

                if y.nunique() < 2:
                    st.warning("La variable objetivo tiene una sola clase. Ajusta la ventana/segmentos.")
                    st.stop()

                modelos = {
                    "LogisticRegression": LogisticRegression(max_iter=800, C=0.3, penalty="l2", class_weight='balanced'),
                    "RandomForest": RandomForestClassifier(
                        n_estimators=250, max_depth=6, min_samples_leaf=10,
                        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
                    ),
                }
                if HAS_XGB:
                    modelos["XGBoost"] = XGBClassifier(
                        n_estimators=350, learning_rate=0.06, max_depth=4,
                        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                        reg_lambda=1.2, random_state=RANDOM_STATE, eval_metric='logloss', tree_method="hist"
                    )
                else:
                    modelos["GradientBoosting"] = GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE
                    )

                n_splits = int(np.clip(y.value_counts().min(), 2, 5))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
                resultados = []
                for nombre, modelo in modelos.items():
                    cv_res = cross_validate(modelo, X, y, cv=cv,
                                            scoring={'accuracy':'accuracy','f1':'f1','roc_auc':'roc_auc'},
                                            n_jobs=-1)
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
                st.subheader("Comparación de Modelos (CV)")
                st.dataframe(df_res.drop(columns=["_auc_mean","_f1_mean"]), use_container_width=True)
                st.success(f"🏆 Mejor modelo: **{mejor_modelo_nombre}**")

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

                # Candidatos (no recientes)
                candidatos = df_feat[df_feat['comprador_reciente'] == 0].copy()

                # Mejor día histórico (dw_)
                dia_cols = [c for c in candidatos.columns if c.startswith("dw_")]
                def mejor_dia(row):
                    if not dia_cols: return None
                    sub = row[dia_cols]
                    if (sub.max() == 0) or sub.isna().all(): return None
                    idx = int(sub.idxmax().split("_")[1])
                    mapa_dw = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
                    return mapa_dw.get(idx)
                candidatos['Dia_Contacto'] = candidatos.apply(mejor_dia, axis=1)

                # Producto sugerido (más comprado históricamente)
                if 'Producto_Nombre' in ventas.columns and not ventas['Producto_Nombre'].isna().all():
                    top_prod_cliente = (ventas.groupby(['Cliente/Empresa', 'Producto_Nombre'])['Total']
                                        .sum().reset_index())
                    idx = top_prod_cliente.groupby('Cliente/Empresa')['Total'].idxmax()
                    top_prod_cliente = top_prod_cliente.loc[idx][['Cliente/Empresa', 'Producto_Nombre']] \
                                                       .rename(columns={'Producto_Nombre':'Producto_Sugerido'})
                    candidatos = candidatos.merge(top_prod_cliente, on='Cliente/Empresa', how='left')
                else:
                    candidatos['Producto_Sugerido'] = None

                # Filtro día
                if dia_reporte != "(Todos)":
                    candidatos = candidatos[candidatos['Dia_Contacto'] == dia_reporte]

                if candidatos.empty:
                    st.info("No hay candidatos con los filtros seleccionados.")
                else:
                    topN = candidatos.nlargest(top_k_sugerencias, 'Prob_Compra')[
                        ['Cliente/Empresa','Prob_Compra','Producto_Sugerido','Dia_Contacto','Segmento']
                    ].copy()
                    asignaciones = (["Camila", "Andrea"] * ((len(topN)//2)+1))[:len(topN)]
                    topN['Asignado_a'] = asignaciones

                    st.subheader("🎯 Top clientes potenciales a contactar")
                    st.dataframe(
                        topN.rename(columns={'Cliente/Empresa':'Cliente','Prob_Compra':'Probabilidad_Compra'}) \
                            .style.format({'Probabilidad_Compra':'{:.1%}'}),
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Descargar sugerencias (CSV)",
                        data=topN.to_csv(index=False).encode('utf-8'),
                        file_name=f"sugerencias_rfm_ml_{pd.Timestamp.today().date()}.csv",
                        mime="text/csv",
                        key="t3_dl"
                    )

# ---------------------------------------------------------------------------------
# TAB 4: MODELO PREDICTIVO DE COMPRADORES POTENCIALES (Balanced Acc / MCC / F1-macro)
# ---------------------------------------------------------------------------------
with tab4:
    st.header("Modelo Predictivo de Compradores Potenciales")

    if 'Producto_Nombre' not in df_ventas.columns:
        st.warning("No se encuentra la columna 'Producto_Nombre' en ventas.")
    else:
        producto_sel = st.selectbox(
            "Producto objetivo:",
            options=sorted(df_ventas['Producto_Nombre'].dropna().unique()),
            key="t4_prod"
        )

        colh1, colh2, colh3 = st.columns(3)
        n_iter_rf  = colh1.slider("Iteraciones búsqueda RF",   5, 40, 15, key="t4_rf_iter")
        n_iter_mlp = colh2.slider("Iteraciones búsqueda MLP",  5, 40, 15, key="t4_mlp_iter")
        n_iter_xgb = colh3.slider(f"Iteraciones búsqueda {'XGB' if HAS_XGB else 'GB'}", 5, 50, 20, key="t4_xgb_iter")

        if st.button("Entrenar y Optimizar Modelos", key="t4_train"):
            with st.spinner("Construyendo dataset, optimizando hiperparámetros y seleccionando el mejor modelo..."):
                data = df_ventas[['Cliente/Empresa','Producto_Nombre','Total','FECHA VENTA']].copy()
                data['Cliente/Empresa'] = data['Cliente/Empresa'].astype(str).str.strip().str.upper()
                data['FECHA VENTA'] = pd.to_datetime(data['FECHA VENTA'], errors="coerce")
                data = data.dropna(subset=['FECHA VENTA'])

                data['Mes'] = data['FECHA VENTA'].dt.month
                data['DiaSemana'] = data['FECHA VENTA'].dt.dayofweek
                data['Hora'] = data['FECHA VENTA'].dt.hour
                data['target'] = (data['Producto_Nombre'] == producto_sel).astype(int)

                feats = data.groupby('Cliente/Empresa').agg(
                    Total_Gastado=('Total','sum'),
                    Num_Transacciones=('Producto_Nombre','count'),
                    Ultimo_Mes=('Mes','max'),
                    Promedio_DiaSemana=('DiaSemana','mean'),
                    Promedio_Hora=('Hora','mean'),
                    Compró=('target','max')
                ).reset_index()

                f_dw = data.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                f_dw.columns = [f"dw_{int(c)}" for c in f_dw.columns]
                f_dw = row_normalize(f_dw)

                f_h  = data.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                f_h.columns = [f"h_{int(c)}" for c in f_h.columns]
                f_h = row_normalize(f_h)

                top_prod = (data.groupby('Producto_Nombre')['Total'].sum()
                            .sort_values(ascending=False).head(10).index.tolist())
                v_prod = data[data['Producto_Nombre'].isin(top_prod)].copy()
                f_prod = v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0)
                f_prod = row_normalize(f_prod)

                DS = feats.merge(f_dw, on='Cliente/Empresa', how='left') \
                          .merge(f_h,  on='Cliente/Empresa', how='left') \
                          .merge(f_prod, on='Cliente/Empresa', how='left')
                for c in DS.select_dtypes(include=[np.number]).columns:
                    DS[c] = DS[c].fillna(0)

                y = DS['Compró'].astype(int)
                X = DS.drop(columns=['Cliente/Empresa','Compró'])

                cls_counts = y.value_counts()
                st.caption(f"Distribución de clases (Compró / No Compró): {cls_counts.to_dict()}")
                if y.nunique() < 2:
                    st.error("El objetivo tiene una sola clase. Cambia el producto o revisa datos.")
                    st.stop()

                n_splits = int(np.clip(cls_counts.min(), 2, 5))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

                scorer_balacc = make_scorer(balanced_accuracy_score)
                scorer_mcc    = make_scorer(matthews_corrcoef)
                scorer_f1m    = make_scorer(f1_score, average="macro")

                # RF
                rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
                rf_space = {
                    "n_estimators": np.linspace(200, 800, 7, dtype=int).tolist(),
                    "max_depth": [None, 6, 10, 14],
                    "min_samples_leaf": [1, 2, 4, 8, 12],
                    "max_features": ["sqrt", 0.5, None]
                }
                rf_search = RandomizedSearchCV(
                    rf, rf_space, n_iter=n_iter_rf, scoring=scorer_balacc, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                ).fit(X, y)

                # XGB / GB
                if HAS_XGB:
                    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", tree_method="hist")
                    xgb_space = {
                        "n_estimators": np.linspace(200, 700, 6, dtype=int).tolist(),
                        "learning_rate": [0.03, 0.05, 0.07, 0.1],
                        "max_depth": [3, 4, 5, 6],
                        "subsample": [0.8, 0.9, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
                        "min_child_weight": [1, 3, 5]
                    }
                    xgb_search = RandomizedSearchCV(
                        xgb, xgb_space, n_iter=n_iter_xgb, scoring=scorer_balacc, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    ).fit(X, y)
                    gb_label = "XGBoost"
                else:
                    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
                    gb_space = {
                        "n_estimators": np.linspace(150, 500, 8, dtype=int).tolist(),
                        "learning_rate": [0.03, 0.05, 0.07, 0.1],
                        "max_depth": [2, 3, 4],
                        "min_samples_leaf": [1, 5, 10, 20]
                    }
                    xgb_search = RandomizedSearchCV(
                        gb, gb_space, n_iter=n_iter_xgb, scoring=scorer_balacc, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    ).fit(X, y)
                    gb_label = "GradientBoosting"

                # MLP
                mlp = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=800))
                ])
                mlp_space = {
                    "clf__hidden_layer_sizes": [(64,32), (128,64), (64,64,32)],
                    "clf__alpha": [1e-4, 1e-3, 1e-2],
                    "clf__learning_rate_init": [1e-3, 5e-4],
                    "clf__batch_size": [32, 64]
                }
                mlp_search = RandomizedSearchCV(
                    mlp, mlp_space, n_iter=n_iter_mlp, scoring=scorer_balacc, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                ).fit(X, y)

                def eval_model(est):
                    scores = cross_validate(
                        est, X, y, cv=cv,
                        scoring={'BalAcc': scorer_balacc, 'MCC': scorer_mcc, 'F1_macro': scorer_f1m},
                        n_jobs=-1
                    )
                    return (
                        float(np.mean(scores['test_BalAcc'])),
                        float(np.mean(scores['test_MCC'])),
                        float(np.mean(scores['test_F1_macro']))
                    )

                models_best = [
                    ("RandomForest", rf_search),
                    (gb_label,       xgb_search),
                    ("MLPClassifier", mlp_search)
                ]

                rows = []
                for name, search in models_best:
                    balacc, mcc, f1m = eval_model(search.best_estimator_)
                    rows.append({
                        "Modelo": name,
                        "Balanced Acc (CV)": f"{balacc:.3f}",
                        "MCC (CV)": f"{mcc:.3f}",
                        "F1-macro (CV)": f"{f1m:.3f}",
                        "Mejores Hiperparámetros": str(search.best_params_),
                        "_key": (balacc, mcc, f1m)
                    })

                df_cmp = pd.DataFrame(rows).sort_values(
                    by=["Balanced Acc (CV)","MCC (CV)","F1-macro (CV)"], ascending=False
                ).drop(columns=["_key"], errors="ignore")
                st.subheader("📈 Resultados de Optimización (mejor configuración por modelo)")
                st.dataframe(df_cmp, use_container_width=True)

                best_row = max(rows, key=lambda r: r["_key"])
                best_name = best_row["Modelo"]
                best_search = dict(models_best)[best_name]
                st.success(
                    f"🏆 Mejor modelo: **{best_name}** · "
                    f"Balanced Acc={best_row['Balanced Acc (CV)']} · "
                    f"MCC={best_row['MCC (CV)']} · F1-macro={best_row['F1-macro (CV)']}"
                )

                best_estimator = best_search.best_estimator_
                best_estimator.fit(X, y)
                if hasattr(best_estimator, "predict_proba"):
                    probas = best_estimator.predict_proba(X)[:, 1]
                elif hasattr(best_estimator, "decision_function"):
                    s = best_estimator.decision_function(X)
                    probas = (s - s.min()) / (s.max() - s.min() + 1e-9)
                else:
                    probas = best_estimator.predict(X).astype(float)

                DS = X.copy()
                DS['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].drop_duplicates().values[:len(DS)]
                DS['Probabilidad_Compra'] = probas
                candidatos = DS[['Cliente/Empresa','Probabilidad_Compra']].copy()
                top10 = candidatos.nlargest(10, 'Probabilidad_Compra')

                st.subheader("🎯 Top 10 clientes potenciales (mejor modelo optimizado)")
                st.dataframe(
                    top10.rename(columns={'Cliente/Empresa':'Cliente'}) \
                         .style.format({'Probabilidad_Compra':'{:.1%}'}),
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Descargar candidatos (CSV)",
                    data=top10.to_csv(index=False).encode('utf-8'),
                    file_name=f"candidatos_{producto_sel}_opt_balanced.csv",
                    mime="text/csv",
                    key="t4_dl"
                )

# ---------------------------------------------------------------------------------
# TAB 5: AGENTE DE ANÁLISIS (intents + ROG opcional + fallback semántico)
# ---------------------------------------------------------------------------------
with tab5:
    st.header("🤖 Agente de Análisis (EDA · Cartera · RFM)")
    st.caption("Haz preguntas específicas; el agente responde sólo lo preguntado.")

    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _to_dt(s):
        return pd.to_datetime(s, errors="coerce")

    # Preparación de datos para el agente
    ventas_agent = df_ventas.copy()
    if 'FECHA VENTA' in ventas_agent.columns:
        ventas_agent['FECHA VENTA'] = _to_dt(ventas_agent['FECHA VENTA'])
        ventas_agent = ventas_agent.dropna(subset=['FECHA VENTA'])
        ventas_agent['Mes'] = ventas_agent['FECHA VENTA'].dt.to_period('M').astype(str)

    cartera_agent = df_cartera.copy()
    if 'Fecha de Vencimiento' in cartera_agent.columns:
        cartera_agent['Fecha de Vencimiento'] = _to_dt(cartera_agent['Fecha de Vencimiento'])
        cartera_agent = cartera_agent.dropna(subset=['Fecha de Vencimiento'])
        cartera_agent['DIAS_VENCIDOS'] = (pd.Timestamp.today().normalize() - cartera_agent['Fecha de Vencimiento']).dt.days

    rfm_tab_agent = compute_rfm_table(ventas_agent)

    # ROG opcional
    def rog_llm_answer(question: str, context: str):
        try:
            from openai import OpenAI
        except Exception:
            return None
        api_key = st.secrets.get("ROG_API_KEY")
        if not api_key:
            return None
        base  = st.secrets.get("ROG_API_BASE", None)
        model = st.secrets.get("ROG_MODEL", "gpt-4o-mini")
        try:
            client = OpenAI(api_key=api_key, base_url=base) if base else OpenAI(api_key=api_key)
            system = ("Eres un analista que SOLO responde sobre EDA de ventas, Cartera y RFM. "
                      "Responde únicamente con base en el CONTEXTO provisto; si no hay datos, dilo.")
            prompt = (f"CONTEXTO RELEVANTE:\n{context}\n\nPREGUNTA: {question}\n\n"
                      "Responde breve, precisa y accionable. No inventes datos.")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":prompt}],
                temperature=0.2, max_tokens=350
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    # Fallback semántico
    def semantic_answer(question: str, texts: list[str]) -> str:
        if not texts:
            return "No tengo datos suficientes para responder eso."
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        dv  = vec.fit_transform(texts)
        qv  = vec.transform([question])
        sims = cosine_similarity(qv, dv).ravel()
        idx = sims.argsort()[::-1][:6]
        snips = [texts[i] for i in idx]
        ctx = "\n".join(snips)
        ans = rog_llm_answer(question, ctx)
        if ans:
            return ans
        return "Resumen según lo más relacionado:\n" + "\n".join([f"- {s}" for s in snips])

    def quick_corpus(dfv: pd.DataFrame, dfc: pd.DataFrame, rfm: pd.DataFrame) -> list[str]:
        out = []
        if not dfv.empty:
            total = float(dfv['Total'].sum()) if 'Total' in dfv.columns else 0.0
            out.append(f"EDA: Ventas totales ${total:,.0f}")
            if 'Cliente/Empresa' in dfv.columns: out.append(f"EDA: Clientes únicos {dfv['Cliente/Empresa'].nunique()}")
            if 'Mes' in dfv.columns and 'Total' in dfv.columns:
                s = dfv.groupby('Mes')['Total'].sum()
                if len(s)>0:
                    out.append(f"EDA: Mejor mes {s.idxmax()} ${s.max():,.0f}; peor mes {s.idxmin()} ${s.min():,.0f}")
            if 'Producto_Nombre' in dfv.columns and 'Total' in dfv.columns:
                tp = dfv.groupby('Producto_Nombre')['Total'].sum().sort_values(ascending=False).head(10)
                out.append("EDA: Top productos " + "; ".join([f"{k} ${v:,.0f}" for k,v in tp.items()]))
        if not dfc.empty and {'Saldo pendiente','DIAS_VENCIDOS'}.issubset(dfc.columns):
            total = float(dfc['Saldo pendiente'].sum())
            out.append(f"CARTERA: Total ${total:,.0f}")
            venc = float(dfc[dfc['DIAS_VENCIDOS']>0]['Saldo pendiente'].sum())
            out.append(f"CARTERA: Vencido ${venc:,.0f}")
        if not rfm.empty:
            dist = rfm['Segmento'].value_counts().to_dict()
            out.append("RFM: Distribución " + "; ".join([f"{k}:{v}" for k,v in dist.items()]))
        return out

    # INTENTS: Cartera
    def handle_cartera(query: str):
        if cartera_agent.empty or 'DIAS_VENCIDOS' not in cartera_agent.columns:
            return None
        q = query.lower()

        m = re.search(r"(menos de|<)\s*(\d+)\s*d[ií]as\s*(de\s*)?(mora|vencid[oa]s?)", q)
        if m:
            x = int(m.group(2))
            df = cartera_agent[(cartera_agent['DIAS_VENCIDOS']>0) & (cartera_agent['DIAS_VENCIDOS'] < x)].copy()
            if df.empty: return f"No hay clientes con mora menor a {x} días."
            cols = [c for c in ['Nombre cliente','NÚMERO DE FACTURA','Fecha de Vencimiento','Saldo pendiente','DIAS_VENCIDOS'] if c in df.columns]
            st.dataframe(df[cols].sort_values('DIAS_VENCIDOS'), use_container_width=True)
            total = df['Saldo pendiente'].sum() if 'Saldo pendiente' in df.columns else 0
            return f"{len(df)} facturas con mora < {x} días. Saldo total ${total:,.0f}."

        m = re.search(r"(m[aá]s de|>)\s*(\d+)\s*d[ií]as\s*(de\s*)?(mora|vencid[oa]s?)", q)
        if m:
            x = int(m.group(2))
            df = cartera_agent[cartera_agent['DIAS_VENCIDOS'] > x].copy()
            if df.empty: return f"No hay clientes con mora mayor a {x} días."
            cols = [c for c in ['Nombre cliente','NÚMERO DE FACTURA','Fecha de Vencimiento','Saldo pendiente','DIAS_VENCIDOS'] if c in df.columns]
            st.dataframe(df[cols].sort_values('DIAS_VENCIDOS', ascending=False), use_container_width=True)
            total = df['Saldo pendiente'].sum() if 'Saldo pendiente' in df.columns else 0
            return f"{len(df)} facturas con mora > {x} días. Saldo total ${total:,.0f}."

        m = re.search(r"entre\s*(\d+)\s*y\s*(\d+)\s*d[ií]as\s*(de\s*)?(mora|vencid[oa]s?)", q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = min(a,b), max(a,b)
            df = cartera_agent[(cartera_agent['DIAS_VENCIDOS'] >= lo) & (cartera_agent['DIAS_VENCIDOS'] <= hi)].copy()
            if df.empty: return f"No hay clientes con mora entre {lo} y {hi} días."
            cols = [c for c in ['Nombre cliente','NÚMERO DE FACTURA','Fecha de Vencimiento','Saldo pendiente','DIAS_VENCIDOS'] if c in df.columns]
            st.dataframe(df[cols].sort_values('DIAS_VENCIDOS'), use_container_width=True)
            total = df['Saldo pendiente'].sum() if 'Saldo pendiente' in df.columns else 0
            return f"{len(df)} facturas con mora entre {lo}-{hi} días. Saldo total ${total:,.0f}."

        if "total" in q and ("cartera" in q or "saldo" in q):
            tot = cartera_agent['Saldo pendiente'].sum() if 'Saldo pendiente' in cartera_agent.columns else 0
            venc = cartera_agent[cartera_agent['DIAS_VENCIDOS']>0]['Saldo pendiente'].sum() if 'Saldo pendiente' in cartera_agent.columns else 0
            return f"Cartera total ${tot:,.0f}; Vencido ${venc:,.0f}; Por vencer ${tot - venc:,.0f}."
        return None

    # INTENTS: EDA
    def handle_eda(query: str):
        if ventas_agent.empty:
            return None
        q = query.lower()

        if "ventas totales" in q or "total de ventas" in q:
            total = ventas_agent['Total'].sum() if 'Total' in ventas_agent.columns else 0
            return f"Ventas totales: ${total:,.0f}."

        if "clientes únicos" in q or "numero de clientes" in q:
            n = ventas_agent['Cliente/Empresa'].nunique() if 'Cliente/Empresa' in ventas_agent.columns else 0
            return f"Clientes únicos: {n}."

        if "mejor mes" in q or "mes con mayores ventas" in q:
            if {'Mes','Total'}.issubset(ventas_agent.columns):
                s = ventas_agent.groupby('Mes')['Total'].sum()
                if len(s)>0: return f"Mejor mes: {s.idxmax()} (${s.max():,.0f})."
            return "No tengo serie por mes para responder."

        if "peor mes" in q or "mes con menores ventas" in q:
            if {'Mes','Total'}.issubset(ventas_agent.columns):
                s = ventas_agent.groupby('Mes')['Total'].sum()
                if len(s)>0: return f"Peor mes: {s.idxmin()} (${s.min():,.0f})."
            return "No tengo serie por mes para responder."

        m = re.search(r"top\s*(\d+)?\s*productos", q)
        if m and 'Producto_Nombre' in ventas_agent.columns and 'Total' in ventas_agent.columns:
            n = int(m.group(1)) if m.group(1) else 10
            tp = ventas_agent.groupby('Producto_Nombre')['Total'].sum().sort_values(ascending=False).head(n).reset_index()
            st.dataframe(tp.rename(columns={'Total':'Ventas'}), use_container_width=True)
            return f"Top {n} productos por ventas mostrados en la tabla."

        m = re.search(r"top\s*(\d+)?\s*clientes", q)
        if m and 'Cliente/Empresa' in ventas_agent.columns and 'Total' in ventas_agent.columns:
            n = int(m.group(1)) if m.group(1) else 10
            tc = ventas_agent.groupby('Cliente/Empresa')['Total'].sum().sort_values(ascending=False).head(n).reset_index()
            st.dataframe(tc.rename(columns={'Total':'Ventas'}), use_container_width=True)
            return f"Top {n} clientes por ventas mostrados en la tabla."

        return None

    # INTENTS: RFM
    def handle_rfm(query: str):
        if rfm_tab_agent.empty:
            return None
        q = query.lower()

        if "cuantos" in q and "segmento" in q:
            dist = rfm_tab_agent['Segmento'].value_counts()
            return "; ".join([f"{k}: {v}" for k,v in dist.items()])

        m = re.search(r"(clientes|listar).*(segmento)\s+([a-z\s]+)", q)
        if m:
            seg = m.group(3).strip().title()
            df = rfm_tab_agent[rfm_tab_agent['Segmento'].str.title() == seg]
            if df.empty:
                return f"No encuentro clientes en el segmento '{seg}'."
            st.dataframe(df[['Cliente/Empresa','Recencia','Frecuencia','Monetario','Segmento']], use_container_width=True)
            return f"{len(df)} clientes en el segmento {seg} (tabla mostrada)."

        m = re.search(r"top\s*(\d+)\s*(clientes)?\s*(por)?\s*(monetario|frecuencia|recencia)", q)
        if m:
            n = int(m.group(1))
            metric = m.group(4).lower()
            col_map = {"monetario":"Monetario","frecuencia":"Frecuencia","recencia":"Recencia"}
            col = col_map[metric]
            asc = False if metric in ("monetario","frecuencia") else True
            show = rfm_tab_agent.sort_values(col, ascending=asc).head(n)[['Cliente/Empresa','Recencia','Frecuencia','Monetario','Segmento']]
            st.dataframe(show, use_container_width=True)
            return f"Top {n} por {col} mostrado en la tabla."

        return None

    # Reporte consolidado (si existe)
    report_ctx: str | None = st.session_state.get("AGMS_REPORT")

    # Estado de chat
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = [
            {"role":"assistant","content":"Hola, soy tu agente. Ejemplos: 'clientes con menos de 6 días de mora', 'top 5 productos', 'listar clientes segmento champions'."}
        ]

    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Escribe tu pregunta…")
    if question:
        st.session_state.agent_history.append({"role":"user","content":question})

        # 1) Intents
        answer = handle_cartera(question)
        if answer is None:
            answer = handle_eda(question)
        if answer is None:
            answer = handle_rfm(question)

        # 2) Si no hubo intent, usa reporte/corpus con selección semántica
        if answer is None:
            if report_ctx:
                sents = re.split(r'(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ#\-•])', report_ctx.strip())
                sents = [s.strip() for s in sents if s.strip()]
                if sents:
                    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
                    dv  = vec.fit_transform(sents)
                    qv  = vec.transform([question])
                    sims = cosine_similarity(qv, dv).ravel()
                    idx = sims.argsort()[::-1][:6]
                    top_snips = [sents[i] for i in idx]
                    ctx = "\n".join(top_snips)
                    llm = rog_llm_answer(question, ctx)
                    answer = llm if llm else "Resumen:\n" + "\n".join([f"- {s}" for s in top_snips])
                else:
                    answer = "El reporte está vacío."
            else:
                texts = quick_corpus(ventas_agent, cartera_agent, rfm_tab_agent)
                answer = semantic_answer(question, texts)

        st.session_state.agent_history.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    if report_ctx:
        st.success("Reglas de intents + Reporte Consolidado para preguntas abiertas.")
    else:
        st.info("No hay reporte consolidado. Reglas + corpus automático desde los datos.")
