# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
from datetime import datetime

# ML
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# XGBoost opcional (si no está instalado, usamos GradientBoosting)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="📊", layout="wide")

st.title("📊 Dashboard de Análisis de Ventas y Predicción")
st.markdown("---")

# ==============================================================================
# CARGA Y LIMPIEZA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        # Hojas
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

        # Cartera
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
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera = load_data()

# ==============================================================================
# UTILIDADES
# ==============================================================================
def build_time_derivatives(df: pd.DataFrame, fecha_col: str):
    df = df.copy()
    dt = pd.to_datetime(df[fecha_col], errors="coerce")
    if "Mes" not in df.columns:
        df["Mes"] = dt.dt.to_period("M").astype(str)
    if "Semana" not in df.columns:
        df["Semana"] = dt.dt.to_period("W").astype(str)
    if "Día" not in df.columns:
        df["Día"] = dt.dt.date
    return df

def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

# ==============================================================================
# LAYOUT (SIN FILTROS DE SIDEBAR)
# ==============================================================================
if df_ventas is None or df_cartera is None:
    st.warning("No se pudieron cargar los datos.")
else:
    df_filtrado = df_ventas.copy()

    # --- Tabs (se eliminaron filtros de sidebar y la pestaña de 'Clientes Potenciales') ---
    tab1, tab2, tab3, tab4 = st.tabs(["Análisis de Ventas", "Gestión de Cartera", "Análisis RFM", "Modelo Predictivo de Compradores Potenciales"])

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
        if fecha_col:
            df_filtrado = build_time_derivatives(df_filtrado, fecha_col)

        granularidad = st.selectbox("Granularidad", options=["Mes", "Semana", "Día"], index=0, key="sel_gran_tab1")
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
    # Pestaña 2: Gestión de Cartera
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
    # Pestaña 3: Análisis RFM + Recomendador
    # ---------------------------------------------------------------------------------
    with tab3:
        st.header("Análisis RFM + Recomendador ML")

        colp1, colp2, colp3, colp4 = st.columns(4)
        dias_recencia = colp1.slider("Ventana para 'comprador reciente' (días)", 7, 120, 30, key="sld_rfm_recencia")
        top_k_sugerencias = colp2.slider("Nº de sugerencias a mostrar", 5, 30, 10, key="sld_rfm_topN")
        usar_top_productos = colp3.checkbox("Usar señales de productos (Top 10)", value=True, key="chk_rfm_topprod")
        excluir_recencia = colp4.checkbox("Excluir 'Recencia' como feature", value=True, key="chk_rfm_excluir_rec")

        dias_op = ["(Todos)","Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
        dia_reporte = st.selectbox("Día deseado para el reporte de candidatos", dias_op, index=0, key="sel_rfm_dia")

        cols_necesarias = {'Cliente/Empresa', 'FECHA VENTA', 'Total'}
        if not cols_necesarias.issubset(df_ventas.columns):
            st.warning(f"Faltan columnas para RFM/ML. Se requieren: {cols_necesarias}.")
        else:
            ejecutar = st.button("🚀 Ejecutar RFM + Entrenar y Comparar Modelos", key="btn_rfm_run")
            if ejecutar:
                with st.spinner("Procesando..."):
                    ventas = df_ventas.copy()
                    ventas['Cliente/Empresa'] = ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()
                    ventas['FECHA VENTA'] = pd.to_datetime(ventas['FECHA VENTA'], errors="coerce")
                    ventas = ventas.dropna(subset=['FECHA VENTA'])
                    ref_date = ventas['FECHA VENTA'].max()
                    tiene_factura = 'NÚMERO DE FACTURA' in ventas.columns

                    # -------- RFM ----------
                    rfm = ventas.groupby('Cliente/Empresa').agg(
                        Recencia=('FECHA VENTA', lambda s: (ref_date - s.max()).days),
                        Frecuencia=('NÚMERO DE FACTURA', 'nunique') if tiene_factura else ('FECHA VENTA','count'),
                        Monetario=('Total', 'sum')
                    ).reset_index()

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

                    rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'], ascending=True, labels=[5,4,3,2,1])   # menor=mejor
                    rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
                    rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])

                    def rfm_segment(row):
                        r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
                        if r>=4 and f>=4 and m>=4: return "Champions"
                        if r>=4 and f>=3: return "Loyal"
                        if r>=3 and f>=3 and m>=3: return "Potential Loyalist"
                        if r<=2 and f>=4: return "At Risk"
                        if r<=2 and f<=2 and m<=2: return "Hibernating"
                        if r>=3 and f<=2: return "New"
                        return "Need Attention"

                    rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")

                    st.caption("Distribución de segmentos RFM")
                    st.dataframe(rfm['Segmento'].value_counts(dropna=False).rename_axis('Segmento').to_frame('Clientes'),
                                 use_container_width=True)

                    # -------- Features comportamiento ----------
                    ventas['DiaSemana'] = ventas['FECHA VENTA'].dt.dayofweek
                    ventas['Hora'] = ventas['FECHA VENTA'].dt.hour
                    feats_dia = ventas.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                    feats_dia.columns = [f"dw_{int(c)}" for c in feats_dia.columns]
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

                    # Target: compró en últimos N días (clasificación binaria)
                    recientes = ventas[ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=dias_recencia)]['Cliente/Empresa'].unique()
                    df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                    # Elegir segmentos antes de entrenar
                    segmentos_all = sorted(df_feat['Segmento'].dropna().unique().tolist())
                    seg_sel = st.multiselect("Filtrar por Segmento RFM (selección múltiple)",
                                             options=segmentos_all, default=segmentos_all,
                                             key="ms_rfm_segmento_include")
                    if seg_sel:
                        df_feat = df_feat[df_feat['Segmento'].isin(seg_sel)]

                    # X / y
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

                    # Modelos
                    modelos = {
                        "RegLog": LogisticRegression(max_iter=800, C=0.3, penalty="l2", class_weight='balanced', random_state=RANDOM_STATE),
                        "RandomForest": RandomForestClassifier(
                            n_estimators=250, max_depth=6, min_samples_leaf=10,
                            random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
                        ),
                    }
                    if HAS_XGB:
                        modelos["XGB"] = XGBClassifier(
                            n_estimators=350, learning_rate=0.06, max_depth=4,
                            min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                            reg_lambda=1.2, random_state=RANDOM_STATE, eval_metric='logloss', tree_method="hist"
                        )
                    else:
                        modelos["GradBoost"] = GradientBoostingClassifier(
                            n_estimators=300, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE
                        )

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
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
                    st.subheader("Comparación de Modelos (5-Fold CV)")
                    st.dataframe(df_res.drop(columns=["_auc_mean","_f1_mean"]), use_container_width=True)
                    st.success(f"🏆 Mejor modelo base: **{mejor_modelo_nombre}**")

                    # Entrenar mejor modelo base
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

                    candidatos = df_feat[df_feat['comprador_reciente'] == 0].copy()
                    dia_cols = [c for c in candidatos.columns if c.startswith("dw_")]
                    def mejor_dia(row):
                        if not dia_cols: return None
                        sub = row[dia_cols]
                        if (sub.max() == 0) or sub.isna().all(): return None
                        idx = int(sub.idxmax().split("_")[1])
                        mapa_dw = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
                        return mapa_dw.get(idx)
                    candidatos['Dia_Contacto'] = candidatos.apply(mejor_dia, axis=1)
                    if dia_reporte != "(Todos)":
                        candidatos = candidatos[candidatos['Dia_Contacto'] == dia_reporte]

                    topN = candidatos.nlargest(top_k_sugerencias, 'Prob_Compra')[
                        ['Cliente/Empresa','Prob_Compra','Dia_Contacto','Segmento']
                    ].copy()
                    asignaciones = (["Camila", "Andrea"] * ((len(topN)//2)+1))[:len(topN)]
                    topN['Asignado_a'] = asignaciones

                    st.subheader("🎯 Top clientes potenciales a contactar (modelo base)")
                    st.dataframe(
                        topN.rename(columns={'Cliente/Empresa':'Cliente','Prob_Compra':'Probabilidad_Compra'}) \
                            .style.format({'Probabilidad_Compra':'{:.1%}'}),
                        use_container_width=True
                    )

    # ---------------------------------------------------------------------------------
# Pestaña 4: Modelo Predictivo de Compradores Potenciales (robusto y con reporting)
# ---------------------------------------------------------------------------------
with tab4:
    st.header("Modelo Predictivo de Compradores Potenciales")

    import json
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
    from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    import numpy as np

    RANDOM_STATE = 42
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    if 'Producto_Nombre' not in df_ventas.columns:
        st.warning("No se encuentra la columna 'Producto_Nombre' en ventas.")
    else:
        producto_sel = st.selectbox(
            "Producto objetivo:",
            options=sorted(df_ventas['Producto_Nombre'].dropna().unique()),
            key="sel_tab4_prod"
        )

        # Controles de la optimización
        colh1, colh2, colh3 = st.columns(3)
        n_iter_rf  = colh1.slider("Iteraciones búsqueda RF",   5, 40, 15, key="rf_iter")
        n_iter_mlp = colh2.slider("Iteraciones búsqueda MLP",  5, 40, 15, key="mlp_iter")
        n_iter_xgb = colh3.slider(f"Iteraciones búsqueda {'XGB' if HAS_XGB else 'GB'}", 5, 50, 20, key="xgb_iter")

        if st.button("Entrenar y Optimizar Modelos", key="btn_tab4_train"):
            with st.spinner("Construyendo dataset, optimizando hiperparámetros y seleccionando el mejor modelo..."):
                # =========================
                # 1) Construcción de dataset tabular por cliente
                # =========================
                data = df_ventas[['Cliente/Empresa','Producto_Nombre','Total','FECHA VENTA']].copy()
                data['Cliente/Empresa'] = data['Cliente/Empresa'].astype(str).str.strip().str.upper()
                data['FECHA VENTA'] = pd.to_datetime(data['FECHA VENTA'], errors="coerce")
                data = data.dropna(subset=['FECHA VENTA'])

                # Señales temporales
                data['Mes'] = data['FECHA VENTA'].dt.month
                data['DiaSemana'] = data['FECHA VENTA'].dt.dayofweek
                data['Hora'] = data['FECHA VENTA'].dt.hour

                # Target por cliente: ha comprado el producto objetivo alguna vez
                data['target'] = (data['Producto_Nombre'] == producto_sel).astype(int)

                # Agregación por cliente (features base)
                feats = data.groupby('Cliente/Empresa').agg(
                    Total_Gastado=('Total','sum'),
                    Num_Transacciones=('Producto_Nombre','count'),
                    Ultimo_Mes=('Mes','max'),
                    Promedio_DiaSemana=('DiaSemana','mean'),
                    Promedio_Hora=('Hora','mean'),
                    Compró=('target','max')
                ).reset_index()

                # Distribuciones por día/hora normalizadas
                def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
                    sums = df_counts.sum(axis=1).replace(0, 1)
                    return df_counts.div(sums, axis=0)

                f_dw = data.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                f_dw.columns = [f"dw_{int(c)}" for c in f_dw.columns]
                f_dw = row_normalize(f_dw)

                f_h  = data.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                f_h.columns = [f"h_{int(c)}" for c in f_h.columns]
                f_h = row_normalize(f_h)

                # Top productos (sin excluir el objetivo, porque aporta señal de afinidad)
                top_prod = (data.groupby('Producto_Nombre')['Total'].sum()
                            .sort_values(ascending=False).head(10).index.tolist())
                v_prod = data[data['Producto_Nombre'].isin(top_prod)].copy()
                f_prod = v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0)
                f_prod = row_normalize(f_prod)

                DS = feats.merge(f_dw, on='Cliente/Empresa', how='left') \
                          .merge(f_h,  on='Cliente/Empresa', how='left') \
                          .merge(f_prod, on='Cliente/Empresa', how='left')

                # Relleno de numéricas
                for c in DS.select_dtypes(include=[np.number]).columns:
                    DS[c] = DS[c].fillna(0)

                y = DS['Compró'].astype(int)
                X = DS.drop(columns=['Cliente/Empresa','Compró'])

                # Chequeos de clase
                cls_counts = y.value_counts()
                st.caption(f"Distribución de clases (Compró / No Compró): {cls_counts.to_dict()}")
                if y.nunique() < 2:
                    st.error("El objetivo tiene una sola clase (todos compraron o ninguno). Cambia el producto o revisa datos.")
                    st.stop()

                # Número de folds robusto con clase minoritaria
                n_splits = int(np.clip(cls_counts.min(), 2, 5))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

                # =========================
                # 2) Scorers y explicación AUC NaN
                # =========================
                def roc_auc_safe(y_true, y_proba):
                    """Devuelve ROC-AUC o NaN si no hay 2 clases en el fold."""
                    try:
                        if len(np.unique(y_true)) < 2:
                            return np.nan
                        return roc_auc_score(y_true, y_proba)
                    except Exception:
                        return np.nan

                # Usamos AP (PR-AUC) para optimizar — es más estable en clases desbalanceadas
                scorer_ap  = 'average_precision'
                scorer_auc = make_scorer(roc_auc_safe, needs_proba=True)

                # =========================
                # 3) Búsquedas de hiperparámetros
                # =========================
                resultados = []

                # --- RandomForest ---
                rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
                rf_space = {
                    "n_estimators": np.linspace(200, 800, 7, dtype=int).tolist(),
                    "max_depth": [None, 6, 10, 14],
                    "min_samples_leaf": [1, 2, 4, 8, 12],
                    "max_features": ["sqrt", 0.5, None]
                }
                rf_search = RandomizedSearchCV(
                    rf, rf_space, n_iter=n_iter_rf, scoring=scorer_ap, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                )
                rf_search.fit(X, y)

                # --- XGBoost o GradientBoosting ---
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
                        xgb, xgb_space, n_iter=n_iter_xgb, scoring=scorer_ap, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    )
                    xgb_search.fit(X, y)
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
                        gb, gb_space, n_iter=n_iter_xgb, scoring=scorer_ap, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    )
                    xgb_search.fit(X, y)
                    gb_label = "GradientBoosting"

                # --- MLP (pipeline con escalado) ---
                mlp = Pipeline([
                    ("scaler", StandardScaler()),  # X es denso; centrado OK
                    ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=700))
                ])
                mlp_space = {
                    "clf__hidden_layer_sizes": [(64,32), (128,64), (64,64,32)],
                    "clf__alpha": [1e-4, 1e-3, 1e-2],
                    "clf__learning_rate_init": [1e-3, 5e-4],
                    "clf__batch_size": [32, 64]
                }
                mlp_search = RandomizedSearchCV(
                    mlp, mlp_space, n_iter=n_iter_mlp, scoring=scorer_ap, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                )
                mlp_search.fit(X, y)

                # =========================
                # 4) Re-evaluación con AUC y AP (k-fold) para cada mejor estimador
                # =========================
                def eval_model(est):
                    scores = cross_validate(
                        est, X, y, cv=cv,
                        scoring={'AUC': scorer_auc, 'AP': 'average_precision'},
                        n_jobs=-1, return_estimator=False
                    )
                    auc_mean = np.nanmean(scores['test_AUC'])  # puede ser NaN si un fold no tiene 2 clases
                    ap_mean = np.nanmean(scores['test_AP'])
                    return auc_mean, ap_mean

                models_best = [
                    ("RandomForest", rf_search),
                    (gb_label,       xgb_search),
                    ("MLPClassifier", mlp_search)
                ]

                rows = []
                for name, search in models_best:
                    auc_m, ap_m = eval_model(search.best_estimator_)
                    rows.append({
                        "Modelo": name,
                        "ROC-AUC (CV)": "Indefinido (clase única en algún fold)" if np.isnan(auc_m) else f"{auc_m:.4f}",
                        "PR-AUC (CV)": f"{ap_m:.4f}",
                        "Mejores Hiperparámetros": json.dumps(search.best_params_, ensure_ascii=False, indent=2)
                    })

                df_cmp = pd.DataFrame(rows).sort_values(
                    by=["ROC-AUC (CV)", "PR-AUC (CV)"],
                    ascending=False,
                    key=lambda s: s.astype(str).str.replace("Indefinido.*","-1", regex=True).astype(float)
                )

                st.subheader("📈 Resultados de Optimización (mejor configuración por modelo)")
                st.dataframe(df_cmp, use_container_width=True)

                # Mostrar hiperparámetros detallados en expanders
                st.markdown("**Hiperparámetros ganadores (detalle):**")
                for name, search in models_best:
                    with st.expander(f"{name} · mejores hiperparámetros"):
                        st.json(search.best_params_)

                # Selección del mejor: priorizamos AUC si está definido; si no, PR-AUC
                def ranking_key(row):
                    auc_val = np.nan if "Indefinido" in str(row["ROC-AUC (CV)"]) else float(row["ROC-AUC (CV)"])
                    ap_val  = float(row["PR-AUC (CV)"])
                    # Devolvemos tupla para ordenar: primero AUC (tratando NaN como -1), luego AP
                    return (auc_val if not np.isnan(auc_val) else -1.0, ap_val)

                best_row = max(rows, key=ranking_key)
                best_name = best_row["Modelo"]
                best_search = dict(models_best)[best_name]
                st.success(f"🏆 Mejor modelo: **{best_name}** · ROC-AUC={best_row['ROC-AUC (CV)']} · PR-AUC={best_row['PR-AUC (CV)']}")

                # =========================
                # 5) Entrenamiento final y ranking de candidatos
                # =========================
                best_estimator = best_search.best_estimator_
                best_estimator.fit(X, y)

                if hasattr(best_estimator, "predict_proba"):
                    probas = best_estimator.predict_proba(X)[:, 1]
                elif hasattr(best_estimator, "decision_function"):
                    s = best_estimator.decision_function(X)
                    probas = (s - s.min()) / (s.max() - s.min() + 1e-9)
                else:
                    probas = best_estimator.predict(X)

                DS['Probabilidad_Compra'] = probas
                candidatos = DS[DS['Compró'] == 0][['Cliente/Empresa','Probabilidad_Compra']].copy()
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
                    file_name=f"candidatos_{producto_sel}_opt.csv",
                    mime="text/csv",
                    key="dl_pred_opt_csv"
                )

                st.info("ℹ️ Si ves **ROC-AUC = Indefinido**, significa que en algún fold de la validación cruzada solo había una clase; "
                        "en esos casos el AUC no puede calcularse. El modelo se optimizó con **PR-AUC (Average Precision)**, "
                        "que es más estable en escenarios desbalanceados.")
