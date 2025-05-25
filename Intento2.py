import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Cargar modelo y preprocesador
modelo = tf.keras.models.load_model("modelo2.keras")
with open("preprocesador.pkl", "rb") as f:
    preprocessor = pickle.load(f)

app = dash.Dash(__name__)
app.title = "Predicción Saber 11"

# Layout
app.layout = html.Div([
    html.H2("Predicción de Probabilidad de Alto Rendimiento en Matemáticas"),

    html.Label("¿Tiene lavadora? (1=Sí, 0=No)"),
    dcc.Input(id='lavadora', type='number', value=1),

    html.Label("¿Tiene automóvil? (1=Sí, 0=No)"),
    dcc.Input(id='auto', type='number', value=0),

    html.Label("¿Tiene computador? (1=Sí, 0=No)"),
    dcc.Input(id='computador', type='number', value=1),

    html.Label("Área de ubicación del colegio"),
    dcc.Dropdown(
        id='area',
        options=[
            {'label': 'Urbano', 'value': 1},
            {'label': 'Rural', 'value': 0}
        ],
        value=1
    ),

    html.Label("Municipio de residencia (código)"),
    dcc.Input(id='reside', type='number', value=5001),  # Medellín como ejemplo

    html.Label("Municipio presentación (código)"),
    dcc.Input(id='presentacion', type='number', value=5001),

    html.Label("Tipo de documento"),
    dcc.Dropdown(
        id='documento',
        options=[{'label': x, 'value': x} for x in ['CC', 'TI', 'CE']],
        value='TI'
    ),

    html.Label("Personas en el hogar"),
    dcc.Input(id='hogar', type='text', value='Cuatro'),

    html.Label("Número de cuartos en el hogar"),
    dcc.Input(id='cuartos', type='text', value='Tres'),

    html.Label("Educación madre"),
    dcc.Dropdown(
        id='educacion',
        options=[{'label': x, 'value': x} for x in [
            'Ninguno', 'Primaria', 'Secundaria', 'Universitaria', 'Postgrado'
        ]],
        value='Primaria'
    ),

    html.Label("Estrato de vivienda"),
    dcc.Dropdown(
        id='estrato',
        options=[{'label': f'Estrato {i}', 'value': f'Estrato {i}'} for i in range(1, 7)],
        value='Estrato 3'
    ),

    html.Br(),
    html.Button('Predecir', id='boton'),
    html.Div(id='resultado')
])

# Callback
@app.callback(
    Output('resultado', 'children'),
    Input('boton', 'n_clicks'),
    State('lavadora', 'value'),
    State('auto', 'value'),
    State('computador', 'value'),
    State('area', 'value'),
    State('reside', 'value'),
    State('presentacion', 'value'),
    State('documento', 'value'),
    State('hogar', 'value'),
    State('cuartos', 'value'),
    State('educacion', 'value'),
    State('estrato', 'value'),
)
def predecir(n, lavadora, auto, computador, area, reside, presentacion, documento, hogar, cuartos, educacion, estrato):
    if n is None:
        return ""

    # Armar input como DataFrame
    df_input = pd.DataFrame([{
        'FAMI_TIENELAVADORA_Si': lavadora,
        'FAMI_TIENEAUTOMOVIL_Si': auto,
        'FAMI_TIENECOMPUTADOR_Si': computador,
        'COLE_AREA_UBICACION_URBANO': area,
        'ESTU_COD_RESIDE_MCPIO': reside,
        'ESTU_COD_MCPIO_PRESENTACION': presentacion,
        'ESTU_TIPODOCUMENTO': documento,
        'FAMI_PERSONASHOGAR': hogar,
        'FAMI_CUARTOSHOGAR': cuartos,
        'FAMI_EDUCACIONMADRE': educacion,
        'FAMI_ESTRATOVIVIENDA': estrato
    }])

    try:
        # Preprocesamiento y predicción
        X_input = preprocessor.transform(df_input)
        prob = modelo.predict(X_input)[0][0]
        return f"🔮 Probabilidad estimada de alto rendimiento: {prob:.2%}"
    except Exception as e:
        return f"⚠️ Error en la predicción: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)

