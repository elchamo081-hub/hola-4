import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from graficas import (
    graphic_line,
    density_plot,
    city_bars,
    heatmap,
    heatmap2,
    acf_pacf,
    forecast,
)


st.set_page_config(
    page_title="Precios de Gasolina de Colombia",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

cities = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena", "Bucaramanga"]


st.title("Precios de Gasolina de Colombia")

selected_city = st.selectbox("Selecciona una ciudad", cities)
col = st.columns((2, 4, 4), gap="medium")

with col[0]:
    st.markdown(f"#### Serie de Tiempo {selected_city}")
    density_plot()
    heatmap2(selected_city)
    graphic_line(selected_city)

with col[1]:
    st.markdown("### Análisis Comparativo de Precios de Gasolina")
    city_bars()
    acf_pacf()

with col[2]:
    st.markdown("#### Pronóstico de Precios de Gasolina")
    one_g, two_g = forecast()
    st.pyplot(two_g)
    st.pyplot(one_g)
