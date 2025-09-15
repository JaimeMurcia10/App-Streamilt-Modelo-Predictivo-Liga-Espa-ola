import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Archivos
archivo_excel = "LaLiga.xlsx"
modelo_guardado = "modelo_laliga.pkl"

# ==============================
# Funciones base
# ==============================
def cargar_datos():
    df = pd.read_excel(archivo_excel)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def preparar_dataset_para_modelo(df):
    registros = []
    for _, row in df.iterrows():
        registros.append({
            "team": row["HomeTeam"],
            "opponent": row["AwayTeam"],
            "venue": "Home",
            "gf": row["FTHG"],
            "ga": row["FTAG"],
            "hs": row["HS"],
            "hst": row["HST"],
            "hc": row["HC"],
            "result": "W" if row["FTR"] == "H" else ("D" if row["FTR"] == "D" else "L")
        })
        registros.append({
            "team": row["AwayTeam"],
            "opponent": row["HomeTeam"],
            "venue": "Away",
            "gf": row["FTAG"],
            "ga": row["FTHG"],
            "hs": row["AS"],
            "hst": row["AST"],
            "hc": row["AC"],
            "result": "W" if row["FTR"] == "A" else ("D" if row["FTR"] == "D" else "L")
        })
    df_modelo = pd.DataFrame(registros)
    def resultado_clase(res):
        if res == "W":
            return 1
        elif res == "D":
            return 0
        else:
            return -1
    df_modelo["target"] = df_modelo["result"].apply(resultado_clase)
    return df_modelo

def entrenar_modelo():
    df = cargar_datos()
    df_modelo = preparar_dataset_para_modelo(df)
    dummies = pd.get_dummies(df_modelo[["team", "opponent", "venue"]], drop_first=True)
    df_final = pd.concat([dummies, df_modelo[["gf", "ga", "hs", "hst", "hc", "target"]]], axis=1)
    X = df_final.drop("target", axis=1)
    y = df_final["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)
    joblib.dump((modelo, X.columns.tolist()), modelo_guardado)

def predecir_partido(local, visitante):
    if not os.path.exists(modelo_guardado):
        entrenar_modelo()
    modelo, columnas_modelo = joblib.load(modelo_guardado)
    df = cargar_datos()
    df_modelo = preparar_dataset_para_modelo(df)
    input_data = pd.DataFrame(columns=columnas_modelo)
    input_data.loc[0] = 0

    if f"team_{local}" in input_data.columns:
        input_data.at[0, f"team_{local}"] = 1
    if f"opponent_{visitante}" in input_data.columns:
        input_data.at[0, f"opponent_{visitante}"] = 1
    if "venue_Home" in input_data.columns:
        input_data.at[0, "venue_Home"] = 1

    estadisticas = df_modelo[df_modelo["team"] == local][["gf", "ga", "hs", "hst", "hc"]].mean()
    for col in ["gf", "ga", "hs", "hst", "hc"]:
        input_data[col] = estadisticas[col] if not pd.isna(estadisticas[col]) else 0

    resultado = modelo.predict(input_data)[0]
    probas = modelo.predict_proba(input_data)[0]

    return resultado, probas

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="⚽ Predictor de LaLiga", layout="wide")
st.title("⚽ Predictor de LaLiga")
st.sidebar.header("📌 Menú de opciones")
menu = st.sidebar.radio("Selecciona una opción", ["Lista de equipos", "Analizar equipo", "Predecir partido"])

df = cargar_datos()
equipos = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

# ---- Opción 1: Lista de equipos
if menu == "Lista de equipos":
    st.subheader("📋 Lista de equipos en LaLiga")
    st.write(equipos)

# ---- Opción 2: Analizar equipo
elif menu == "Analizar equipo":
    equipo = st.selectbox("Selecciona un equipo", equipos, key="equipo_selector")
    partidos_local = df[df["HomeTeam"] == equipo]
    partidos_visitante = df[df["AwayTeam"] == equipo]
    partidos_totales = pd.concat([partidos_local, partidos_visitante])

    if partidos_totales.empty:
        st.error("❌ No se encontraron partidos para este equipo")
    else:
        ganados = empatados = perdidos = 0
        for _, row in partidos_totales.iterrows():
            if row["HomeTeam"] == equipo:
                if row["FTR"] == "H":
                    ganados += 1
                elif row["FTR"] == "A":
                    perdidos += 1
                else:
                    empatados += 1
            else:
                if row["FTR"] == "A":
                    ganados += 1
                elif row["FTR"] == "H":
                    perdidos += 1
                else:
                    empatados += 1

        goles_favor = partidos_local["FTHG"].sum() + partidos_visitante["FTAG"].sum()
        goles_contra = partidos_local["FTAG"].sum() + partidos_visitante["FTHG"].sum()

        # ---- Tarjetas estilo oscuro
        st.markdown("### 📊 Resumen de rendimiento")
        card_style = """
            <div style='background:#1e1e2f;
                        padding:20px;
                        border-radius:15px;
                        text-align:center;
                        color:white;
                        box-shadow:0px 4px 12px rgba(0,0,0,0.5);'>
                <h2 style='font-size:22px;'>{titulo}</h2>
                <h1 style='font-size:36px;margin:0;'>{valor}</h1>
            </div>
        """

        col1, col2, col3, col4 = st.columns(4, gap="large")
        with col1:
            st.markdown(card_style.format(titulo="🏟️ Partidos", valor=len(partidos_totales)), unsafe_allow_html=True)
        with col2:
            st.markdown(card_style.format(titulo="✅ Victorias", valor=ganados), unsafe_allow_html=True)
        with col3:
            st.markdown(card_style.format(titulo="🤝 Empates", valor=empatados), unsafe_allow_html=True)
        with col4:
            st.markdown(card_style.format(titulo="❌ Derrotas", valor=perdidos), unsafe_allow_html=True)

        col5, col6 = st.columns(2, gap="large")
        with col5:
            st.markdown(card_style.format(titulo="⚽ Goles a favor", valor=goles_favor), unsafe_allow_html=True)
        with col6:
            st.markdown(card_style.format(titulo="🥅 Goles en contra", valor=goles_contra), unsafe_allow_html=True)

        # ---- Gráfico de barras
        resultados = pd.DataFrame({
            "Resultado": ["Victorias", "Empates", "Derrotas"],
            "Cantidad": [ganados, empatados, perdidos]
        })
        fig = px.bar(resultados, x="Resultado", y="Cantidad",
                    color="Resultado", text="Cantidad",
                    title=f"📊 Distribución de resultados de {equipo}",
                    color_discrete_sequence=px.colors.sequential.Agsunset)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key="bar_resultados")

        # ---- Gráfico circular efectividad
        efectividad = round((ganados / len(partidos_totales)) * 100, 1)
        fig_pie = px.pie(values=[ganados, len(partidos_totales)-ganados],
                        names=["Victorias", "Otros"],
                        title=f"🔥 Efectividad de {equipo} ({efectividad}%)",
                        hole=0.5,
                        color_discrete_sequence=["#00cc96", "#636efa"])
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_efectividad")

# ---- Opción 3: Predecir partido
elif menu == "Predecir partido":
    local = st.selectbox("🏠 Equipo Local", equipos, key="local_selector")
    visitante = st.selectbox("🚌 Equipo Visitante", [e for e in equipos if e != local], key="visitante_selector")
    if st.button("🔮 Predecir resultado", key="btn_predecir"):
        resultado, probas = predecir_partido(local, visitante)
        st.subheader(f"📊 Predicción: {local} vs {visitante}")
        st.write("🟢 Gana local (1): {:.1f}%".format(probas[2]*100))
        st.write("🟡 Empate (0): {:.1f}%".format(probas[1]*100))
        st.write("🔴 Gana visitante (-1): {:.1f}%".format(probas[0]*100))
        st.success(f"✅ Resultado predicho: {['Visitante','Empate','Local'][resultado+1]}")

        # ---- Gráfica circular probabilidades
        fig_pred = px.pie(values=[probas[2], probas[1], probas[0]],
                        names=["Local", "Empate", "Visitante"],
                        title="📈 Distribución de probabilidades",
                        hole=0.4,
                        color_discrete_sequence=["#00cc96", "#f5c542", "#ef553b"])
        fig_pred.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pred, use_container_width=True, key="pie_predicciones")





