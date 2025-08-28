# ==============================================================================
# LIBRERÍAS E IMPORTACIONES
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import warnings
from datetime import datetime
import random
import numpy as np

warnings.filterwarnings('ignore')

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

st.title("📊 Dashboard de Análisis de Ventas y Predicción")
st.markdown("---")

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = "DB_AGMS.xlsx"
    df_ventas = pd.read_excel(file_path, sheet_name="Ventas", header=1)
    return df_ventas

df_ventas = load_data()

# ==============================================================================
# PESTAÑA: MODELO PREDICTIVO DE COMPRADORES POTENCIALES
# ==============================================================================
st.header("Modelo Predictivo de Compradores Potenciales")

if 'Producto_Nombre' in df_ventas.columns:
    producto_a_predecir = st.selectbox("Producto:", options=sorted(df_ventas['Producto_Nombre'].unique()), key="sel_tab_pred")

    if st.button("Buscar Compradores", key="btn_tab_pred"):
        with st.spinner("Entrenando modelos de Machine Learning..."):

            # --- Preparación de datos ---
            df_modelo = df_ventas[['Cliente/Empresa', 'Producto_Nombre', 'Total', 'FECHA VENTA']].copy()
            df_modelo['FECHA VENTA'] = pd.to_datetime(df_modelo['FECHA VENTA'], errors="coerce")
            df_modelo['Mes'] = df_modelo['FECHA VENTA'].dt.month
            df_modelo['DiaSemana'] = df_modelo['FECHA VENTA'].dt.dayofweek
            df_modelo['Hora'] = df_modelo['FECHA VENTA'].dt.hour

            # Target: compró o no el producto seleccionado
            df_modelo['Compró'] = (df_modelo['Producto_Nombre'] == producto_a_predecir).astype(int)

            # Features por cliente
            feats = df_modelo.groupby('Cliente/Empresa').agg(
                Total_Gastado=('Total','sum'),
                Num_Transacciones=('Producto_Nombre','count'),
                Ultimo_Mes=('Mes','max'),
                Promedio_DiaSemana=('DiaSemana','mean'),
                Promedio_Hora=('Hora','mean'),
                Compró=('Compró','max')
            ).reset_index()

            X = feats.drop(columns=['Cliente/Empresa','Compró'])
            y = feats['Compró']

            # --- Modelos a evaluar ---
            modelos = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=300, max_depth=6, random_state=42, class_weight='balanced'
                ),
                "MLPClassifier": MLPClassifier(
                    hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                    max_iter=500, random_state=42
                )
            }
            if HAS_XGB:
                modelos["XGBoost"] = XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    subsample=0.9, colsample_bytree=0.9,
                    reg_lambda=1.0, random_state=42, eval_metric="logloss"
                )

            # --- Validación cruzada ---
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            resultados = []
            for nombre, modelo in modelos.items():
                cv_res = cross_validate(modelo, X, y, cv=cv,
                                        scoring={'accuracy':'accuracy','f1':'f1','roc_auc':'roc_auc'},
                                        n_jobs=-1)
                resultados.append({
                    "Modelo": nombre,
                    "Accuracy": f"{cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}",
                    "F1": f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
                    "AUC": f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
                    "_auc_mean": cv_res['test_roc_auc'].mean(),
                    "_f1_mean": cv_res['test_f1'].mean()
                })

            df_res = pd.DataFrame(resultados).sort_values(by=["_auc_mean","_f1_mean"], ascending=False)
            mejor_modelo_nombre = df_res.iloc[0]["Modelo"]

            st.subheader("Comparación de Modelos (5-Fold CV)")
            st.dataframe(df_res.drop(columns=["_auc_mean","_f1_mean"]), use_container_width=True)
            st.success(f"🏆 Mejor modelo: **{mejor_modelo_nombre}**")

            # --- Entrenar modelo final ---
            best_model = modelos[mejor_modelo_nombre]
            best_model.fit(X, y)

            # --- Probabilidades ---
            if hasattr(best_model, "predict_proba"):
                feats['Probabilidad_Compra'] = best_model.predict_proba(X)[:,1]
            else:
                s_full = best_model.decision_function(X)
                feats['Probabilidad_Compra'] = (s_full - s_full.min()) / (s_full.max() - s_full.min() + 1e-9)

            # --- Resultados: top 10 candidatos ---
            candidatos = feats[feats['Compró']==0].copy()
            top10 = candidatos.nlargest(10, 'Probabilidad_Compra')[['Cliente/Empresa','Probabilidad_Compra']]

            st.subheader("🎯 Top 10 clientes potenciales")
            st.dataframe(
                top10.rename(columns={'Cliente/Empresa':'Cliente'}) \
                     .style.format({'Probabilidad_Compra':'{:.1%}'}),
                use_container_width=True
            )

            # Descargar CSV
            st.download_button(
                "⬇️ Descargar candidatos (CSV)",
                data=top10.to_csv(index=False).encode('utf-8'),
                file_name=f"candidatos_{producto_a_predecir}.csv",
                mime="text/csv",
                key="dl_pred_csv"
            )
