import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.express as px

# --- Funciones Auxiliares ---
def redondear_preciso(valor, precision):
    return round(valor + 1e-12, precision)

def calcular_amplitud(rango_val, num_intervalos, precision_datos):
    if rango_val == 0:
        return 1 / (10 ** precision_datos) if precision_datos > 0 else 1
    amplitud_calc = rango_val / num_intervalos
    amplitud_red = math.ceil(amplitud_calc * (10**(precision_datos + 1))) / (10**(precision_datos + 1))
    return redondear_preciso(amplitud_red, precision_datos + 2)

def contar_frecuencias(datos, clases, precision):
    frec_abs = [0] * len(clases)
    epsilon = 1 / (10 ** (precision + 5))
    for dato in datos:
        for i, (li, ls) in enumerate(clases):
            if (i < len(clases) - 1 and li <= dato < ls) or (i == len(clases) - 1 and li <= dato <= ls + epsilon):
                frec_abs[i] += 1
                break
    return frec_abs

# --- Función Principal de Análisis ---
def calcular_estadisticas_agrupadas(datos_str):
    try:
        datos = sorted([float(x.strip()) for x in datos_str.split(',') if x.strip()])
    except ValueError:
        st.error("Asegúrese de que todos los datos sean números válidos separados por comas.")
        return None

    n = len(datos)
    if n == 0:
        st.error("No se ingresaron datos.")
        return None

    rango_val = max(datos) - min(datos)
    precision_datos = max(len(str(d).split('.')[-1]) if '.' in str(d) else 0 for d in datos)
    num_intervalos = int(round(1 + math.log2(n))) if n > 1 else 1
    amplitud = calcular_amplitud(rango_val, num_intervalos, precision_datos)

    # Generar clases y marcas
    clases_limites = []
    marcas_clase = []
    li = min(datos)
    for _ in range(num_intervalos):
        ls = li + amplitud
        li_r = redondear_preciso(li, precision_datos + 3)
        ls_r = redondear_preciso(ls, precision_datos + 3)
        clases_limites.append((li_r, ls_r))
        marcas_clase.append(redondear_preciso((li + ls) / 2, precision_datos + 4))
        li = ls

    frec_abs = contar_frecuencias(datos, clases_limites, precision_datos)

    df = pd.DataFrame({
        'Clase': [f"{li:.{precision_datos+2}f} - {ls:.{precision_datos+2}f}" for li, ls in clases_limites],
        'Límite Inferior': [li for li, _ in clases_limites],
        'Límite Superior': [ls for _, ls in clases_limites],
        'Marca de Clase (xi)': marcas_clase,
        'Frecuencia Absoluta (fi)': frec_abs
    })

    df['Frecuencia Acumulada (Fi)'] = df['Frecuencia Absoluta (fi)'].cumsum()
    df['Frecuencia Relativa (hi)'] = df['Frecuencia Absoluta (fi)'] / n
    df['Frecuencia Relativa Acumulada (Hi)'] = df['Frecuencia Relativa (hi)'].cumsum()
    df['xi * fi'] = df['Marca de Clase (xi)'] * df['Frecuencia Absoluta (fi)']

    # Media
    media = df['xi * fi'].sum() / n

    # Mediana
    pos_mediana = n / 2
    idx_mediana = df[df['Frecuencia Acumulada (Fi)'] >= pos_mediana].index[0]
    Li_med = df.loc[idx_mediana, 'Límite Inferior']
    Fi_ant = df['Frecuencia Acumulada (Fi)'].iloc[idx_mediana - 1] if idx_mediana > 0 else 0
    fi_med = df.loc[idx_mediana, 'Frecuencia Absoluta (fi)']
    mediana = Li_med + ((pos_mediana - Fi_ant) / fi_med) * amplitud if fi_med > 0 else np.nan

    # Moda
    idx_moda = df['Frecuencia Absoluta (fi)'].idxmax()
    fi_mod = df.loc[idx_moda, 'Frecuencia Absoluta (fi)']
    Li_mod = df.loc[idx_moda, 'Límite Inferior']
    fi_ant = df['Frecuencia Absoluta (fi)'].iloc[idx_moda - 1] if idx_moda > 0 else 0
    fi_sig = df['Frecuencia Absoluta (fi)'].iloc[idx_moda + 1] if idx_moda < len(df) - 1 else 0
    denom = (fi_mod - fi_ant + fi_mod - fi_sig)
    moda = Li_mod + ((fi_mod - fi_ant) / denom) * amplitud if denom != 0 else df.loc[idx_moda, 'Marca de Clase (xi)']

    # Varianza y Desviación Estándar
    df['(xi - media)^2 * fi'] = ((df['Marca de Clase (xi)'] - media) ** 2) * df['Frecuencia Absoluta (fi)']
    varianza = df['(xi - media)^2 * fi'].sum() / (n - 1) if n > 1 else 0
    desviacion_estandar = math.sqrt(varianza)
    coef_variacion = (desviacion_estandar / abs(media)) * 100 if media != 0 else float('inf')

    return {
        "n": n,
        "rango": rango_val,
        "num_intervalos_sugerido": num_intervalos,
        "amplitud": amplitud,
        "tabla_frecuencias": df,
        "media": media,
        "mediana": mediana,
        "moda": moda,
        "varianza": varianza,
        "desviacion_estandar": desviacion_estandar,
        "coeficiente_variacion": coef_variacion,
        "datos_originales_ordenados": datos,
        "precision_datos": precision_datos
    }


# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Análisis de Datos Agrupados", layout="wide")

st.title("📊 PRÁCTICA 1: Tablas de Frecuencia con Datos Agrupados")
st.markdown("""
**DESARROLLO DE PRÁCTICA 3 DEL CURSO: CÁLCULO DE LA TABLA DE FRECUENCIAS CON DATOS AGRUPADOS**

Esta aplicación calcula:
- Intervalo, Amplitud, Rango
- Clases, Frecuencias, Marca de Clase, etc.
- Medidas de Tendencia Central
- Medidas de Variabilidad (para valor extra)
- Gráfica (Histograma)

**Instrucciones:** Ingrese los datos numéricos separados por comas (ej: `23, 45.5, 33, 21, 50, ...`).
""")

# Entrada de datos
sample_data = "71,81,65,70,84,68,75,77,62,79,88,60,73,80,66,70,85,69,76,72,83,61,74,82,67,70,86,63,78,89,66,71,84,67,75,80,62,79,87,64,73,81,60,77,85,68,72,88,65,76"
datos_input = st.text_area("Ingrese sus datos aquí:", value=sample_data, height=100, placeholder="Ej: 10, 12.5, 15, 10, 18, 20.2, 22, 15, 17, 12.5")

if st.button("Calcular y Generar Tabla"):
    if datos_input:
        resultados = calcular_estadisticas_agrupadas(datos_input)

        if resultados:
            st.header("Resultados del Análisis")
            precision_display = resultados['precision_datos'] + 2 # Para mostrar resultados

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Estadísticas Iniciales")
                st.write(f"**Cantidad de Datos (N):** {resultados['n']}")
                st.write(f"**Rango (R):** {resultados['rango']:.{precision_display}f}")
            with col2:
                st.write(f"**Número de Intervalos (k - Sturges):** {resultados['num_intervalos_sugerido']}")
                st.write(f"**Amplitud del Intervalo (A):** {resultados['amplitud']:.{precision_display}f}")


            st.subheader("Tabla de Frecuencias")
            df_display = resultados['tabla_frecuencias'].copy()

            # Formatear columnas numéricas para mejor visualización
            format_dict = {
                'Marca de Clase (xi)': f"{{:.{precision_display}f}}",
                'Frecuencia Relativa (hi)': "{:.4f}",
                'Frecuencia Relativa Acumulada (Hi)': "{:.4f}",
                'xi * fi': f"{{:.{precision_display}f}}",
                '(xi - media)^2 * fi': f"{{:.{precision_display}f}}"
            }
            # Las columnas de límites ya están en el DataFrame como strings formateados
            # df_display['Límite Inferior'] = df_display['Límite Inferior'].apply(lambda x: f"{x:.{precision_display}f}")
            # df_display['Límite Superior'] = df_display['Límite Superior'].apply(lambda x: f"{x:.{precision_display}f}")


            columnas_a_mostrar = ['Clase', 'Marca de Clase (xi)', 'Frecuencia Absoluta (fi)',
                                  'Frecuencia Acumulada (Fi)', 'Frecuencia Relativa (hi)',
                                  'Frecuencia Relativa Acumulada (Hi)']
            st.dataframe(df_display[columnas_a_mostrar].style.format(format_dict, na_rep="-"))

            with st.expander("Ver columnas auxiliares para cálculos (Límites, xi*fi, (xi-media)^2*fi)"):
                 st.dataframe(df_display[['Clase','Límite Inferior', 'Límite Superior', 'xi * fi', '(xi - media)^2 * fi']].style.format(format_dict, na_rep="-"))


            st.markdown("---")
            st.subheader("Medidas de Tendencia Central")
            col_mtc1, col_mtc2, col_mtc3 = st.columns(3)
            with col_mtc1:
                st.metric(label="Media ($\\bar{x}$)", value=f"{resultados['media']:.{precision_display}f}" if not np.isnan(resultados['media']) else "N/A")
            with col_mtc2:
                st.metric(label="Mediana ($Me$)", value=f"{resultados['mediana']:.{precision_display}f}" if not np.isnan(resultados['mediana']) else "N/A")
            with col_mtc3:
                st.metric(label="Moda ($Mo$)", value=f"{resultados['moda']:.{precision_display}f}" if not np.isnan(resultados['moda']) else "N/A")

            st.markdown("---")
            st.subheader("Medidas de Variabilidad")
            col_mv1, col_mv2, col_mv3 = st.columns(3)
            with col_mv1:
                st.metric(label="Varianza ($s^2$)", value=f"{resultados['varianza']:.{precision_display}f}" if not np.isnan(resultados['varianza']) else "N/A")
            with col_mv2:
                st.metric(label="Desviación Estándar ($s$)", value=f"{resultados['desviacion_estandar']:.{precision_display}f}" if not np.isnan(resultados['desviacion_estandar']) else "N/A")
            with col_mv3:
                cv_value = resultados['coeficiente_variacion']
                cv_display = f"{cv_value:.2f}%" if not (np.isnan(cv_value) or np.isinf(cv_value)) else "N/A"
                st.metric(label="Coeficiente de Variación ($CV$)", value=cv_display)

            st.markdown("---")
            st.subheader("Gráfica: Histograma de Frecuencias")

            df_hist = resultados['tabla_frecuencias'].copy()
            # Para Plotly, es mejor si el eje x es categórico o si se define explícitamente el binning.
            # Aquí usaremos las etiquetas de clase como categorías.

            fig = px.bar(df_hist,
                         x='Clase',
                         y='Frecuencia Absoluta (fi)',
                         title='Histograma de Frecuencias',
                         labels={'Clase': 'Clases', 'Frecuencia Absoluta (fi)': 'Frecuencia Absoluta'},
                         text_auto=True) # Mostrar valor de frecuencia en cada barra

            fig.update_layout(xaxis_title="Clases",
                              yaxis_title="Frecuencia Absoluta",
                              bargap=0.05) # Espacio entre barras
            # Rotar etiquetas del eje X si son muchas clases para evitar superposición
            if len(df_hist['Clase']) > 8:
                fig.update_xaxes(tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver datos originales ordenados"):
                st.write(resultados['datos_originales_ordenados'])
    else:
        st.warning("Por favor, ingrese datos antes de calcular.")

st.markdown("---")
st.markdown("Desarrollado como parte de la Práctica del curso. 👨‍💻")