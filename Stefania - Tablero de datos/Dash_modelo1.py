import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
#Importar librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Cargar datos 
df = pd.read_csv("df_limpio.csv")

# Inicializar app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H2("Distribución del puntaje de matemáticas segun género del colegio"),

    html.Label("Género/Tipo de colegio"),
    dcc.Dropdown(
        id='COLE_GENERO',
        options=[{'label': x, 'value': x} for x in [
            'MIXTO', 'FEMENINO', 'MASCULINO'
        ]],
        value='Mixto',
        multi=True
    ),

        html.Br(),
    dcc.Graph(id='grafico_genero'),
    
    html.H2("Box-Plot del puntaje de matemáticas segun calendario escolar"),

    html.Label("Calendario Escolar"),
    dcc.Dropdown(
        id='COLE_CALENDARIO',
        options=[{'label': x, 'value': x} for x in [
            'A', 'B','OTRO'
        ]],
        value=['A'],
        multi=True
    ),

        html.Br(),
    dcc.Graph(id='grafico_calendario'),

    html.H2("Diagrama de violín del puntaje de matemáticas por jornada escolar "),

    html.Label("Jornada del colegio"),
    dcc.Dropdown(
        id="COLE_JORNADA",
        options=[{'label': x, 'value': x} for x in [
            'MAÑANA', 'TARDE', 'SABATINA', 'NOCHE', 'COMPLETA'
        ]],
        value=['MAÑANA'],
        multi = True
    ),

        html.Br(),
    dcc.Graph(id='grafico_jornada'),

    html.H2("Diagrama de barras del puntaje de matemáticas por municipio"),

    html.Label("Municipio del colegio"),
    dcc.Dropdown(
        id='COLE_MCPIO_UBICACION',
        options=[{'label': x, 'value': x} for x in df['COLE_MCPIO_UBICACION'].dropna().unique()],
        value=['MEDELLÍN'],
        multi = True
    ),

        html.Br(),
    dcc.Graph(id='grafico_municipio'),

])

#-----------Gráfico KDE por género del coelgio-----------------------
@app.callback(
    Output("grafico_genero", 'figure'),
    Input("COLE_GENERO", 'value')
)


def grafico_genero(lista_generos):
    fig = go.Figure()

    if not lista_generos:
        return fig  

    for genero in lista_generos:
        datos = df[df['COLE_GENERO'] == genero]['PUNT_MATEMATICAS'].dropna()
        if len(datos) > 1:
            try:
                kde = gaussian_kde(datos)
                x_vals = np.linspace(datos.min(), datos.max(), 200)
                y_vals = kde(x_vals)

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    fill='tozeroy',
                    name=genero,
                    mode='lines'
                ))
            except Exception as e:
                print(f"Error procesando {genero}: {e}")

    fig.update_layout(
        title="Distribución del puntaje de matemáticas por tipo de colegio",
        xaxis_title="Puntaje de Matemáticas",
        yaxis_title="Densidad",
        template='simple_white'
    )
    return fig

#-----------Box Plot por calendario del coelgio-----------------------
@app.callback(
    Output("grafico_calendario", 'figure'),
    Input("COLE_CALENDARIO", 'value')
)
def grafico_calendario(lista_calendario):
    if not lista_calendario:
        return go.Figure()

    df_filtrado = df[df['COLE_CALENDARIO'].isin(lista_calendario)]
    if df_filtrado.empty:
        return go.Figure()

    fig = go.Figure()

    for calendario in lista_calendario:
        subset = df_filtrado[df_filtrado['COLE_CALENDARIO'] == calendario]

        fig.add_trace(go.Box(
            y=subset['PUNT_MATEMATICAS'],
            name=calendario,
            boxpoints='outliers',
            jitter=0.4,
            pointpos=0,
            marker=dict(opacity=0.6),
            line=dict(width=1),
            boxmean='sd'  
        ))

    fig.update_layout(
        title="Box-Plot del puntaje de matemáticas por calendario del colegio",
        yaxis_title="Puntaje de Matemáticas",
        xaxis_title="Calendario escolar",
        template='simple_white',
        boxmode='group',
        height=1000
    )

    return fig


#--------- Gráfico de violin jornada---------
# Callback
@app.callback(
    Output('grafico_jornada', 'figure'),
    Input('COLE_JORNADA', 'value')
)
def grafico_jornada(lista_jornada):
    if not lista_jornada:
        return go.Figure()

    df_filtrado = df[df['COLE_JORNADA'].isin(lista_jornada)]
    fig = go.Figure()

    colores = [
    'lightblue', 'lightgreen', 'salmon', 'violet', 'gold',
    
    ]

    for i, j in enumerate(lista_jornada):
        datos = df_filtrado[df_filtrado['COLE_JORNADA'] == j]
        fig.add_trace(go.Violin(
            y=datos['PUNT_MATEMATICAS'],
            name=j,
            box_visible=True,
            meanline_visible=True,
            line=dict(color='black'),
            fillcolor=colores[i % len(colores)],
            opacity=0.7,
            points='outliers',
            jitter=0.3,
            scalemode='count'
        ))

    fig.update_layout(
        title="Distribución de puntajes de matemáticas por nivel de educación de la madre)",
        yaxis_title="Puntaje de Matemáticas",
        xaxis_title="Nivel de educación",
        template='simple_white',
        height=500
    )

    return fig

#-------------Grafico de barras por municipio---------
import plotly.express as px

@app.callback(
    Output('grafico_municipio', 'figure'),
    Input('COLE_MCPIO_UBICACION', 'value')
)
def grafico_municipio(lista_municipio):
    if not lista_municipio:
        return go.Figure()

    # Filtrar y agrupar
    df_filtrado = df[df['COLE_MCPIO_UBICACION'].isin(lista_municipio)]
    resumen = df_filtrado.groupby('COLE_MCPIO_UBICACION')['PUNT_MATEMATICAS'].mean().reset_index()

    # Ordenar por promedio (opcional)
    resumen = resumen.sort_values(by='PUNT_MATEMATICAS', ascending=False)

    # Crear gráfico de barras verticales
    fig = px.bar(
        resumen,
        x='PUNT_MATEMATICAS',
        y='COLE_MCPIO_UBICACION',
        orientation='h',
        labels={
            'COLE_MCPIO_UBICACION': 'Municipio',
            'PUNT_MATEMATICAS': 'Promedio de matemáticas'
        },
        title='Promedio de puntaje de matemáticas por educación de la madre',
        color='COLE_MCPIO_UBICACION'  
    )

    fig.update_layout(
        template='simple_white',
        xaxis_tickangle=-45,
        height=500
    )

    return fig


# Ejecutar
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8051)




