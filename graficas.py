import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


def graphic_line(city):
    # Load and clean data
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )
    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Sort by date
    df = df.sort_values("Fecha")

    # Plot trajectories
    plt.figure()
    plt.plot(df["Fecha"], df[city].astype(float))
    plt.title(f"Trayectoria del precio de gasolina - {city}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return st.pyplot(plt)


def density_plot():
    # --- Cargar datos ---
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    # Limpiar tabla
    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]

    # Convertir a formato largo
    df_long = df.melt(id_vars="Fecha", var_name="Ciudad", value_name="Precio")

    df_long["Precio"] = pd.to_numeric(df_long["Precio"], errors="coerce")

    # --- Colores definidos por ciudad ---
    colores = {
        "Bogota": "#4C72B0",  # azul
        "Medellin": "#55A868",  # verde
        "Cali": "#C44E52",  # rojo
        "Barranquilla": "#8172B3",  # morado
        "Cartagena": "#CCB974",  # dorado
        "Bucaramanga": "#64B5CD",  # turquesa
    }

    # Gráfico
    plt.figure(figsize=(12, 6))
    sns.stripplot(
        data=df_long,
        x="Ciudad",
        y="Precio",
        jitter=True,
        alpha=0.8,
        size=8,
        palette=colores,
    )

    plt.title("Distribuciones de precios por ciudad")
    plt.xlabel("Ciudad")
    plt.ylabel("Precio")
    plt.xticks(rotation=30)
    plt.tight_layout()
    return st.pyplot(plt)


def heatmap(ciudad):

    # -----------------------
    # 1. Cargar y limpiar datos
    # -----------------------
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]

    df = df[df["Fecha"] != "Fecha"]
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    ciudades = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
    ]

    for c in ciudades:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------
    # 2. Crear año y trimestre
    # -----------------------
    df["Año"] = df["Fecha"].dt.year
    df["Trimestre"] = df["Fecha"].dt.quarter

    # -----------------------
    # 3. Función para generar mapa de calor por ciudad
    # -----------------------

    tabla = df.groupby(["Año", "Trimestre"])[ciudad].mean().unstack()

    plt.figure(figsize=(10, 4))

    plt.imshow(tabla.T, cmap="viridis", aspect="auto")

    plt.title(ciudad, fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Quarter")

    plt.xticks(ticks=np.arange(len(tabla.index)), labels=tabla.index, rotation=45)
    plt.yticks(ticks=[0, 1, 2, 3], labels=["1", "2", "3", "4"])

    cbar = plt.colorbar()
    cbar.set_label("Precio promedio")

    plt.tight_layout()
    return st.pyplot(plt)

    # -----------------------
    # 4. Generar heatmap para cada ciudad
    # -----------------------


def heatmap2(ciudad):
    # -----------------------
    # 1. Cargar y preparar datos
    # -----------------------
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    ciudades = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
    ]

    for c in ciudades:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Año"] = df["Fecha"].dt.year
    df["Trimestre"] = df["Fecha"].dt.quarter

    # -----------------------
    # 2. Función con estética idéntica a tu imagen
    # -----------------------

    tabla = df.groupby(["Año", "Trimestre"])[ciudad].mean().unstack()

    plt.figure(figsize=(10, 4))

    # heatmap estilo cuadrado
    plt.imshow(
        tabla.T,
        cmap="viridis",  # Colormap igual a ejemplo
        aspect="equal",  # Cuadrados perfectos
        interpolation="none",  # Sin suavizado
    )

    # Título similar al ejemplo
    plt.title(ciudad.lower(), fontsize=22, fontweight="semibold", pad=20)

    # Etiquetas estéticas
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Quarter", fontsize=12)

    # Eje X (años)
    plt.xticks(ticks=np.arange(len(tabla.index)), labels=tabla.index, fontsize=11)

    # Eje Y (trimestres)
    plt.yticks(ticks=[0, 1, 2, 3], labels=["1", "2", "3", "4"], fontsize=11)

    # Cuadrícula estilo suave
    plt.grid(color="white", linestyle="-", linewidth=1, alpha=0.8)

    # Barra de color estilo foto
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    return st.pyplot(plt)


def city_bars():
    # -----------------------
    # 1. Cargar datos
    # -----------------------
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    ciudades = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
    ]

    for c in ciudades:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Año"] = df["Fecha"].dt.year
    df["Trimestre"] = df["Fecha"].dt.quarter

    # -----------------------
    # 2. Seleccionar trimestre
    # -----------------------
    trimestre = 1  # ← Cambia aquí 1, 2, 3 o 4

    # -----------------------
    # 3. Calcular promedios por ciudad
    # -----------------------
    promedios = {}

    for ciudad in ciudades:
        promedios[ciudad] = df[df["Trimestre"] == trimestre][ciudad].mean()

    # Ordenar por valor (opcional)
    ciudades_ord = sorted(promedios, key=promedios.get)
    valores_ord = [promedios[c] for c in ciudades_ord]

    # -----------------------
    # 4. Gráfica de barras estilo la foto
    # -----------------------
    plt.figure(figsize=(10, 4))

    plt.bar(
        ciudades_ord,
        valores_ord,
        color=plt.cm.viridis(
            np.linspace(0.2, 0.8, len(ciudades_ord))
        ),  # tonos azul/verde suaves
    )

    plt.title(
        f"Comparación trimestral (Trimestre {trimestre})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.ylabel("Precio promedio", fontsize=12)

    # Rotar etiquetas como en la imagen
    plt.xticks(rotation=55, ha="right", fontsize=10)

    # Cuadrícula suave en eje Y
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    return st.pyplot(plt)


def acf_pacf():
    # -----------------------
    # 1. Cargar y preparar datos
    # -----------------------
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values("Fecha")

    cities = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena", "Bucaramanga"]
    for c in cities:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------
    # 2. Serie mensual promedio
    # -----------------------
    df["Promedio"] = df[cities].mean(axis=1)
    monthly = df.groupby(pd.Grouper(key="Fecha", freq="ME"))["Promedio"].mean()

    # -----------------------
    # 3. Segunda diferencia (serie estacionaria)
    # -----------------------
    monthly_diff2 = monthly.diff().diff().dropna()

    # -----------------------
    # 4. ACF y PACF
    # -----------------------
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plot_acf(monthly_diff2, ax=plt.gca(), lags=20)
    plt.title("ACF - Segunda diferencia")

    plt.subplot(1, 2, 2)
    plot_pacf(monthly_diff2, ax=plt.gca(), lags=20, method="ywm")
    plt.title("PACF - Segunda diferencia")

    plt.tight_layout()
    return st.pyplot(plt)


def forecast():
    # -----------------------
    # 1. Cargar y preparar datos
    # -----------------------
    df = pd.read_excel(
        "./content/BASE DE DATOS DE PRECIOS DE GASOLINA (1).xlsx", header=1
    )

    df = df.drop(columns=[df.columns[0]])
    df.columns = [
        "Bogota",
        "Medellin",
        "Cali",
        "Barranquilla",
        "Cartagena",
        "Bucaramanga",
        "Fecha",
    ]
    df = df[df["Fecha"] != "Fecha"]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values("Fecha")

    cities = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena", "Bucaramanga"]
    for c in cities:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------
    # 2. Serie mensual promedio
    # -----------------------
    df["Promedio"] = df[cities].mean(axis=1)
    monthly = df.groupby(pd.Grouper(key="Fecha", freq="ME"))["Promedio"].mean()
    # ================================
    # 1. Ajustar modelo ARIMA(2,2,0)
    # ================================
    model_220 = ARIMA(monthly, order=(2, 2, 0))
    result_220 = model_220.fit()

    # Mostrar resumen del modelo
    print(result_220.summary())

    # ================================
    # 2. Diagnóstico de residuos
    # ================================

    one_g = result_220.plot_diagnostics(figsize=(12, 8))

    # ================================
    # 3. Pronóstico
    # ================================
    n_periods = 12  # cambiar si quieres más meses
    forecast = result_220.get_forecast(steps=n_periods)

    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Crear índice futuro
    last_date = monthly.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1), periods=n_periods, freq="M"
    )

    pred_mean.index = future_idx
    conf_int.index = future_idx

    # ================================
    # 4. Gráfica del pronóstico
    # ================================
    plt.figure(figsize=(12, 6))
    plt.plot(monthly, label="Datos históricos")
    plt.plot(pred_mean, label="Pronóstico ARIMA(2,2,0)")
    plt.fill_between(future_idx, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3)
    plt.legend()
    plt.title("Pronóstico con ARIMA(2,2,0)")
    two_g = plt
    return one_g, two_g
