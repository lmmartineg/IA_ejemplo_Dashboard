"""
app.py — Plataforma interactiva de EDA sobre datos sintéticos
=============================================================
Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

Estructura:
    1. Generación de datos sintéticos (parametrizable desde la barra lateral)
    2. Filtros interactivos
    3. EDA: resumen, cuantitativas, cualitativas, bivariado, multivariado
    4. Descarga de los datos generados/filtrados
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EDA Interactivo · Datos Sintéticos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETA = px.colors.qualitative.Set2


# ---------------------------------------------------------------
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Generando datos sintéticos...")
def generar_datos(n: int, seed: int, ruido: float, pct_faltantes: float) -> pd.DataFrame:
    """Genera un dataset sintético de clientes con variables cuanti y cuali
    relacionadas entre sí (no independientes), para que el EDA tenga señal."""
    rng = np.random.default_rng(seed)

    # ---- Variables cualitativas ----
    ciudades = ["Medellín", "Bogotá", "Cali", "Barranquilla", "Cartagena"]
    ciudad = rng.choice(ciudades, n, p=[0.32, 0.30, 0.18, 0.12, 0.08])

    segmento = rng.choice(["Básico", "Plus", "Premium"], n, p=[0.50, 0.33, 0.17])
    canal = rng.choice(["Web", "App", "Sucursal", "Call center"], n, p=[0.40, 0.30, 0.20, 0.10])
    genero = rng.choice(["F", "M", "Otro"], n, p=[0.48, 0.48, 0.04])

    # ---- Variables cuantitativas ----
    edad = np.clip(rng.normal(36, 11, n), 18, 80).round(0)

    # El ingreso depende del segmento (lognormal => asimetría positiva realista)
    base_ingreso = {"Básico": 2.6e6, "Plus": 4.2e6, "Premium": 7.5e6}
    mu = np.log(np.array([base_ingreso[s] for s in segmento]))
    ingreso = rng.lognormal(mu, 0.32 * ruido)

    antiguedad = rng.integers(1, 121, n)  # meses como cliente

    # El gasto depende del ingreso, la antigüedad y el canal
    efecto_canal = {"Web": 1.00, "App": 1.12, "Sucursal": 0.92, "Call center": 0.85}
    gasto = (
        0.22 * ingreso
        + 6_000 * antiguedad
        + np.array([efecto_canal[c] for c in canal]) * 150_000
        + rng.normal(0, 250_000 * ruido, n)
    )
    gasto = np.clip(gasto, 50_000, None)

    n_transacciones = rng.poisson(np.clip(gasto / 350_000, 1, 40)).astype(float)

    satisfaccion = np.clip(
        rng.normal(
            3.4 + 0.5 * (segmento == "Premium") - 0.4 * (canal == "Call center"),
            0.9 * ruido,
        ),
        1,
        5,
    ).round(0)

    # ---- Variable objetivo (churn) vía modelo logístico ----
    z = (
        -0.8
        - 0.9 * (satisfaccion - 3)
        + 0.012 * (60 - antiguedad)
        - 0.35 * (segmento == "Premium")
        + rng.normal(0, 0.5 * ruido, n)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    fecha_alta = pd.to_datetime("2026-07-01") - pd.to_timedelta(antiguedad * 30, unit="D")

    df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, n + 1),
            "fecha_alta": fecha_alta,
            "ciudad": ciudad,
            "segmento": pd.Categorical(segmento, ["Básico", "Plus", "Premium"], ordered=True),
            "canal": canal,
            "genero": genero,
            "edad": edad,
            "ingreso_mensual": ingreso.round(0),
            "gasto_anual": gasto.round(0),
            "antiguedad_meses": antiguedad,
            "n_transacciones": n_transacciones,
            "satisfaccion": satisfaccion,
            "churn": churn,
        }
    )

    # ---- Inyección controlada de datos faltantes (para practicar EDA) ----
    if pct_faltantes > 0:
        for col in ["ingreso_mensual", "satisfaccion", "edad", "canal"]:
            mask = rng.random(n) < pct_faltantes
            df.loc[mask, col] = np.nan

    return df


def clasificar_columnas(df: pd.DataFrame):
    """Separa columnas numéricas y categóricas (excluye identificadores/fechas)."""
    excluir = {"id_cliente", "fecha_alta"}
    num = [c for c in df.select_dtypes(include=np.number).columns if c not in excluir]
    cat = [
        c
        for c in df.columns
        if c not in excluir and c not in num
    ]
    return num, cat


# ---------------------------------------------------------------
# 2. BARRA LATERAL: PARÁMETROS + FILTROS
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros")

with st.sidebar.expander("Generación de datos", expanded=True):
    n = st.slider("Número de observaciones", 200, 20_000, 2_000, step=200)
    seed = st.number_input("Semilla aleatoria", 0, 10_000, 42, step=1)
    ruido = st.slider("Nivel de ruido", 0.2, 3.0, 1.0, step=0.1)
    pct_faltantes = st.slider("% de datos faltantes", 0.0, 0.20, 0.03, step=0.01)
    if st.button("🔄 Regenerar datos", use_container_width=True):
        st.cache_data.clear()

df_raw = generar_datos(n, int(seed), ruido, pct_faltantes)
num_cols, cat_cols = clasificar_columnas(df_raw)

st.sidebar.divider()
st.sidebar.subheader("🔍 Filtros")

df = df_raw.copy()

# Filtros categóricos
for c in ["ciudad", "segmento", "canal"]:
    opciones = sorted(df_raw[c].dropna().astype(str).unique())
    sel = st.sidebar.multiselect(c.capitalize(), opciones, default=opciones)
    df = df[df[c].astype(str).isin(sel) | df[c].isna()]

# Filtros numéricos
rango_edad = st.sidebar.slider(
    "Edad",
    float(df_raw["edad"].min()),
    float(df_raw["edad"].max()),
    (float(df_raw["edad"].min()), float(df_raw["edad"].max())),
)
df = df[df["edad"].between(*rango_edad) | df["edad"].isna()]

if st.sidebar.checkbox("Eliminar filas con datos faltantes", value=False):
    df = df.dropna()

st.sidebar.caption(f"Filas tras filtrar: **{len(df):,}** de {len(df_raw):,}")


# ---------------------------------------------------------------
# 3. CABECERA Y KPIs
# ---------------------------------------------------------------
st.title("📊 Análisis exploratorio de datos sintéticos")
st.caption(
    "Dataset simulado de clientes. Ajusta los parámetros y filtros en la barra lateral: "
    "todos los gráficos y estadísticos se recalculan en tiempo real."
)

if df.empty:
    st.warning("Los filtros actuales no dejan ninguna observación. Amplía la selección.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Observaciones", f"{len(df):,}")
k2.metric("Variables", df.shape[1])
k3.metric("Ingreso medio", f"${df['ingreso_mensual'].mean():,.0f}")
k4.metric("Gasto medio", f"${df['gasto_anual'].mean():,.0f}")
k5.metric("Tasa de churn", f"{df['churn'].mean():.1%}")

st.divider()

tab_res, tab_num, tab_cat, tab_bi, tab_multi, tab_datos = st.tabs(
    ["📋 Resumen", "🔢 Cuantitativas", "🔤 Cualitativas", "🔗 Bivariado", "📈 Multivariado", "⬇️ Datos"]
)


# ---------------------------------------------------------------
# TAB 1 — RESUMEN / CALIDAD DE DATOS
# ---------------------------------------------------------------
with tab_res:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Estructura del dataset")
        estructura = pd.DataFrame(
            {
                "tipo": df.dtypes.astype(str),
                "no_nulos": df.notna().sum(),
                "faltantes": df.isna().sum(),
                "% faltantes": (df.isna().mean() * 100).round(2),
                "únicos": df.nunique(),
            }
        )
        st.dataframe(estructura, use_container_width=True)

    with c2:
        st.subheader("Mapa de datos faltantes")
        faltantes = df.isna().mean().sort_values(ascending=True) * 100
        fig = px.bar(
            x=faltantes.values,
            y=faltantes.index,
            orientation="h",
            labels={"x": "% faltantes", "y": ""},
            color=faltantes.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Primeras filas")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Estadísticos descriptivos (variables numéricas)")
    desc = df[num_cols].describe().T
    desc["asimetría"] = df[num_cols].skew()
    desc["curtosis"] = df[num_cols].kurtosis()
    desc["CV"] = desc["std"] / desc["mean"]
    st.dataframe(desc.round(2), use_container_width=True)


# ---------------------------------------------------------------
# TAB 2 — VARIABLES CUANTITATIVAS
# ---------------------------------------------------------------
with tab_num:
    c1, c2, c3 = st.columns(3)
    var = c1.selectbox("Variable numérica", num_cols, index=num_cols.index("ingreso_mensual"))
    grupo = c2.selectbox("Segmentar por", ["(ninguna)"] + cat_cols)
    bins = c3.slider("Número de bins", 5, 100, 30)

    serie = df[var].dropna()
    color = None if grupo == "(ninguna)" else grupo

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Media", f"{serie.mean():,.2f}")
    m2.metric("Mediana", f"{serie.median():,.2f}")
    m3.metric("Desv. est.", f"{serie.std():,.2f}")
    m4.metric("Asimetría", f"{serie.skew():.2f}")
    m5.metric("Rango IQR", f"{serie.quantile(.75) - serie.quantile(.25):,.2f}")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(
            df, x=var, color=color, nbins=bins, marginal="rug",
            opacity=0.8, color_discrete_sequence=PALETA,
            title=f"Distribución de {var}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        fig = px.box(
            df, y=var, x=color, color=color, points="outliers",
            color_discrete_sequence=PALETA, title=f"Boxplot de {var}",
        )
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig = px.violin(
            df, y=var, x=color, color=color, box=True,
            color_discrete_sequence=PALETA, title=f"Violin plot de {var}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with g4:
        # Q-Q plot contra la normal
        s = np.sort(serie.values)
        teoricos = stats.norm.ppf((np.arange(1, len(s) + 1) - 0.5) / len(s), s.mean(), s.std())
        fig = go.Figure()
        fig.add_scatter(x=teoricos, y=s, mode="markers", name="datos",
                        marker=dict(size=4, opacity=0.6))
        fig.add_scatter(x=teoricos, y=teoricos, mode="lines", name="normal teórica")
        fig.update_layout(title=f"Q-Q plot de {var}", xaxis_title="Cuantiles teóricos",
                          yaxis_title="Cuantiles observados")
        st.plotly_chart(fig, use_container_width=True)

    # Detección de outliers por IQR
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    st.info(
        f"**Outliers (regla 1.5·IQR):** {len(outliers):,} observaciones "
        f"({len(outliers)/len(serie):.2%}) fuera de [{lim_inf:,.0f}, {lim_sup:,.0f}]."
    )


# ---------------------------------------------------------------
# TAB 3 — VARIABLES CUALITATIVAS
# ---------------------------------------------------------------
with tab_cat:
    c1, c2 = st.columns(2)
    var_c = c1.selectbox("Variable categórica", cat_cols)
    tipo_g = c2.radio("Tipo de gráfico", ["Barras", "Pastel", "Treemap"], horizontal=True)

    frec = (
        df[var_c].astype(str).replace("nan", "(faltante)")
        .value_counts(dropna=False).rename_axis(var_c).reset_index(name="frecuencia")
    )
    frec["porcentaje"] = (frec["frecuencia"] / frec["frecuencia"].sum() * 100).round(2)
    frec["frec_acumulada"] = frec["frecuencia"].cumsum()
    frec["% acumulado"] = (frec["porcentaje"].cumsum()).round(2)

    g1, g2 = st.columns([1, 1.4])
    with g1:
        st.subheader("Tabla de frecuencias")
        st.dataframe(frec, use_container_width=True, hide_index=True)

        # Entropía normalizada como medida de concentración
        p = frec["frecuencia"] / frec["frecuencia"].sum()
        entropia = -(p * np.log(p)).sum() / np.log(len(p)) if len(p) > 1 else 0
        st.metric("Entropía normalizada", f"{entropia:.3f}",
                  help="0 = toda la masa en una categoría, 1 = distribución uniforme.")

    with g2:
        if tipo_g == "Barras":
            fig = px.bar(frec, x=var_c, y="frecuencia", color=var_c, text="porcentaje",
                         color_discrete_sequence=PALETA)
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
        elif tipo_g == "Pastel":
            fig = px.pie(frec, names=var_c, values="frecuencia", hole=0.45,
                         color_discrete_sequence=PALETA)
        else:
            fig = px.treemap(frec, path=[var_c], values="frecuencia",
                             color_discrete_sequence=PALETA)
        fig.update_layout(title=f"Distribución de {var_c}", showlegend=False, height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Tabla de contingencia + chi-cuadrado
    st.subheader("Tabla de contingencia y prueba χ²")
    otra = st.selectbox("Cruzar con", [c for c in cat_cols if c != var_c])
    tabla = pd.crosstab(df[var_c], df[otra])
    st.dataframe(tabla, use_container_width=True)

    if tabla.shape[0] > 1 and tabla.shape[1] > 1:
        chi2, pval, gl, esperados = stats.chi2_contingency(tabla)
        n_tot = tabla.values.sum()
        cramer = np.sqrt(chi2 / (n_tot * (min(tabla.shape) - 1)))
        c1, c2, c3 = st.columns(3)
        c1.metric("χ²", f"{chi2:,.2f}")
        c2.metric("p-valor", f"{pval:.4f}")
        c3.metric("V de Cramér", f"{cramer:.3f}")
        conclusion = "hay evidencia de asociación" if pval < 0.05 else "no hay evidencia de asociación"
        st.caption(f"Con α = 0.05, {conclusion} entre **{var_c}** y **{otra}** (gl = {gl}).")


# ---------------------------------------------------------------
# TAB 4 — ANÁLISIS BIVARIADO
# ---------------------------------------------------------------
with tab_bi:
    st.subheader("Numérica vs. numérica")
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("Eje X", num_cols, index=num_cols.index("ingreso_mensual"))
    y = c2.selectbox("Eje Y", num_cols, index=num_cols.index("gasto_anual"))
    color_b = c3.selectbox("Color", ["(ninguno)"] + cat_cols, key="color_bi")
    tendencia = c4.checkbox("Línea de tendencia (OLS)", value=True)

    sub = df[[x, y] + ([color_b] if color_b != "(ninguno)" else [])].dropna()
    fig = px.scatter(
        sub, x=x, y=y,
        color=None if color_b == "(ninguno)" else color_b,
        trendline="ols" if tendencia else None,
        opacity=0.6, color_discrete_sequence=PALETA,
        title=f"{y} vs. {x}",
    )
    st.plotly_chart(fig, use_container_width=True)

    r_p = sub[x].corr(sub[y], method="pearson")
    r_s = sub[x].corr(sub[y], method="spearman")
    c1, c2 = st.columns(2)
    c1.metric("Correlación de Pearson", f"{r_p:.3f}")
    c2.metric("Correlación de Spearman", f"{r_s:.3f}")

    st.divider()
    st.subheader("Numérica vs. categórica")
    c1, c2 = st.columns(2)
    var_n = c1.selectbox("Variable numérica", num_cols, key="bi_num")
    var_k = c2.selectbox("Variable categórica", cat_cols, key="bi_cat")

    resumen = (
        df.groupby(var_k, observed=True)[var_n]
        .agg(n="count", media="mean", mediana="median", desv="std", min="min", max="max")
        .round(2)
    )
    g1, g2 = st.columns([1, 1.3])
    g1.dataframe(resumen, use_container_width=True)
    fig = px.box(df.dropna(subset=[var_n, var_k]), x=var_k, y=var_n, color=var_k,
                 points="outliers", color_discrete_sequence=PALETA)
    fig.update_layout(showlegend=False, title=f"{var_n} por {var_k}")
    g2.plotly_chart(fig, use_container_width=True)

    # ANOVA de un factor
    grupos = [g[var_n].dropna().values for _, g in df.groupby(var_k, observed=True)]
    grupos = [g for g in grupos if len(g) > 1]
    if len(grupos) > 1:
        f_stat, p_anova = stats.f_oneway(*grupos)
        st.caption(
            f"ANOVA de un factor — F = {f_stat:,.2f}, p = {p_anova:.4f}. "
            + ("Las medias difieren significativamente (α = 0.05)."
               if p_anova < 0.05 else "No se rechaza la igualdad de medias (α = 0.05).")
        )


# ---------------------------------------------------------------
# TAB 5 — MULTIVARIADO
# ---------------------------------------------------------------
with tab_multi:
    st.subheader("Matriz de correlaciones")
    metodo = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
    corr = df[num_cols].corr(method=metodo)
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
        color_continuous_scale="RdBu_r", title=f"Correlación ({metodo})",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matriz de dispersión")
    sel_vars = st.multiselect(
        "Variables a incluir", num_cols,
        default=["edad", "ingreso_mensual", "gasto_anual", "antiguedad_meses"],
    )
    color_m = st.selectbox("Color", ["(ninguno)"] + cat_cols, key="color_multi")
    if len(sel_vars) >= 2:
        base_m = df.dropna(subset=sel_vars)
        muestra = base_m.sample(min(1500, len(base_m)), random_state=1)
        fig = px.scatter_matrix(
            muestra, dimensions=sel_vars,
            color=None if color_m == "(ninguno)" else color_m,
            opacity=0.5, color_discrete_sequence=PALETA,
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos dos variables.")

    st.subheader("Serie temporal de altas")
    freq = st.select_slider("Agregación", ["Diaria", "Semanal", "Mensual", "Trimestral"], value="Mensual")
    regla = {"Diaria": "D", "Semanal": "W", "Mensual": "ME", "Trimestral": "QE"}[freq]
    serie_t = df.set_index("fecha_alta").resample(regla).size().reset_index(name="clientes")
    fig = px.area(serie_t, x="fecha_alta", y="clientes", title=f"Altas de clientes ({freq.lower()})")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------
# TAB 6 — DATOS Y DESCARGA
# ---------------------------------------------------------------
with tab_datos:
    st.subheader("Explorador de datos")
    cols_ver = st.multiselect("Columnas a mostrar", list(df.columns), default=list(df.columns))
    n_filas = st.slider("Filas a mostrar", 10, min(1000, len(df)), min(100, len(df)))
    st.dataframe(df[cols_ver].head(n_filas), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Descargar CSV (datos filtrados)",
        df[cols_ver].to_csv(index=False).encode("utf-8-sig"),
        file_name="datos_sinteticos.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ Descargar CSV (dataset completo)",
        df_raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="datos_sinteticos_completo.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Diccionario de variables"):
        st.markdown(
            """
| Variable | Tipo | Descripción |
|---|---|---|
| `id_cliente` | ID | Identificador único |
| `fecha_alta` | Fecha | Fecha de vinculación (derivada de la antigüedad) |
| `ciudad` | Cualitativa nominal | Ciudad de residencia |
| `segmento` | Cualitativa ordinal | Básico < Plus < Premium |
| `canal` | Cualitativa nominal | Canal de contacto principal |
| `genero` | Cualitativa nominal | Género declarado |
| `edad` | Cuantitativa continua | Años (normal truncada) |
| `ingreso_mensual` | Cuantitativa continua | Lognormal condicionada al segmento |
| `gasto_anual` | Cuantitativa continua | Función del ingreso, antigüedad y canal |
| `antiguedad_meses` | Cuantitativa discreta | Meses como cliente |
| `n_transacciones` | Cuantitativa discreta | Poisson con λ dependiente del gasto |
| `satisfaccion` | Cualitativa ordinal (1–5) | Escala Likert |
| `churn` | Binaria | Generada por un modelo logístico latente |
            """
        )
