import dash
from dash import html, dcc, Output, Input
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
import pickle
import os




# Obtener el directorio del archivo actual
BASE_DIR = os.path.dirname(__file__)

# Cargar el modelo
ruta_modelo = os.path.join(BASE_DIR, "modelo2.keras")
modelo = tf.keras.models.load_model("modelo2.keras")

# Cargar el preprocesador
ruta_preprocesador = os.path.join(BASE_DIR, "preprocesador.pkl")
with open(ruta_preprocesador, "rb") as f:
    preprocessor = pickle.load(f)

# Cargar el CSV
ruta_csv = os.path.join(BASE_DIR, "df_limpio.csv")

df = pd.read_csv(ruta_csv)
# Inicializar app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Columnas esperadas por el modelo
columnas_modelo = [
    'FAMI_TIENELAVADORA_Si', 'FAMI_TIENEAUTOMOVIL_Si', 'FAMI_TIENECOMPUTADOR_Si',
    'ESTU_COD_RESIDE_MCPIO', 'ESTU_COD_MCPIO_PRESENTACION', 'COLE_AREA_UBICACION_URBANO',
    'FAMI_PERSONASHOGAR_Cuatro', 'FAMI_PERSONASHOGAR_Diez', 'FAMI_PERSONASHOGAR_Doce o más',
    'FAMI_PERSONASHOGAR_Dos', 'FAMI_PERSONASHOGAR_Nueve', 'FAMI_PERSONASHOGAR_Ocho',
    'FAMI_PERSONASHOGAR_Once', 'FAMI_PERSONASHOGAR_Seis', 'FAMI_PERSONASHOGAR_Siete',
    'FAMI_PERSONASHOGAR_Tres', 'FAMI_PERSONASHOGAR_Una',
    'ESTU_TIPODOCUMENTO_CE', 'ESTU_TIPODOCUMENTO_CR', 'ESTU_TIPODOCUMENTO_PC',
    'ESTU_TIPODOCUMENTO_PE', 'ESTU_TIPODOCUMENTO_PV', 'ESTU_TIPODOCUMENTO_RC',
    'ESTU_TIPODOCUMENTO_TI', 'ESTU_TIPODOCUMENTO_V',
    'FAMI_CUARTOSHOGAR_Cuatro', 'FAMI_CUARTOSHOGAR_Diez o más', 'FAMI_CUARTOSHOGAR_Dos',
    'FAMI_CUARTOSHOGAR_Nueve', 'FAMI_CUARTOSHOGAR_Ocho', 'FAMI_CUARTOSHOGAR_Seis',
    'FAMI_CUARTOSHOGAR_Siete', 'FAMI_CUARTOSHOGAR_Tres', 'FAMI_CUARTOSHOGAR_Uno','FAMI_EDUCACIONMADRE_Educación profesional incompleta',
    'FAMI_EDUCACIONMADRE_Ninguno', 'FAMI_EDUCACIONMADRE_No sabe',
    'FAMI_EDUCACIONMADRE_Postgrado',
    'FAMI_EDUCACIONMADRE_Primaria completa',
    'FAMI_EDUCACIONMADRE_Primaria incompleta',
    'FAMI_EDUCACIONMADRE_Secundaria (Bachillerato) completa',
    'FAMI_EDUCACIONMADRE_Secundaria (Bachillerato) incompleta',
    'FAMI_EDUCACIONMADRE_Técnica o tecnológica completa',
    'FAMI_EDUCACIONMADRE_Técnica o tecnológica incompleta',
    'FAMI_ESTRATOVIVIENDA_Estrato 2', 'FAMI_ESTRATOVIVIENDA_Estrato 3',
    'FAMI_ESTRATOVIVIENDA_Estrato 4', 'FAMI_ESTRATOVIVIENDA_Estrato 5',
    'FAMI_ESTRATOVIVIENDA_Estrato 6'
]

#----------------------------
#LAYOUT
#----------------------------

#--------------- Modelo Pregunta 2 ------------

def layout_modelo():
    return html.Div([
        html.H2("Formulario de variables para predicción"),

        html.Label("¿Tiene lavadora?"),
        dcc.RadioItems(
            id='FAMI_TIENELAVADORA_Si',
            options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
            value=1
        ),

        html.Label("¿Tiene automóvil?"),
        dcc.RadioItems(
            id='FAMI_TIENEAUTOMOVIL_Si',
            options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
            value=1
        ),

        html.Label("¿Tiene computador?"),
        dcc.RadioItems(
            id='FAMI_TIENECOMPUTADOR_Si',
            options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
            value=1
        ),

        html.Label("¿La ubicación de la sede es urbana?"),
        dcc.RadioItems(
            id='COLE_AREA_UBICACION_URBANO',
            options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
            value=1
        ),

        html.Label("Código del municipio de residencia del examinando"),
        dcc.Input(id='ESTU_COD_RESIDE_MCPIO', type='text', value=5001),

        html.Label("Código del municipio de presentación del examen"),
        dcc.Input(id='ESTU_COD_MCPIO_PRESENTACION', type='text', value=5001),

        html.Label("¿Con cuántas personas vive?"),
        dcc.Dropdown(
            id='FAMI_PERSONASHOGAR',
            options=[{'label': x, 'value': x} for x in [
                'Una', 'Dos', 'Tres', 'Cuatro', 'Seis', 'Siete', 'Ocho', 'Nueve',
                'Diez', 'Once', 'Doce o más'
            ]],
            value='Cuatro'
        ),

        html.Label("Tipo de documento"),
        dcc.Dropdown(
            id='ESTU_TIPODOCUMENTO',
            options=[{'label': x, 'value': x} for x in [
                'CC', 'CE', 'CR', 'PC', 'PE', 'PV', 'RC', 'TI', 'V'
            ]],
            value='CC'
        ),

        html.Label("Número de cuartos en el hogar"),
        dcc.Dropdown(
            id='FAMI_CUARTOSHOGAR',
            options=[{'label': x, 'value': x} for x in [
                'Uno', 'Dos', 'Tres', 'Cuatro', 'Seis', 'Siete', 'Ocho', 'Nueve', 'Diez o más'
            ]],
            value='Cuatro'
        ),

        html.Label("Educacón de la madre"),
        dcc.Dropdown(
            id='FAMI_EDUCACIONMADRE',
            options=[{'label': x, 'value': x} for x in [
                'Ninguno', 'No sabe', 'Postgrado', 'Primaria completa', 'Primaria incompleta', 'Secundaria (Bachillerato) completa', 'Secundaria (Bachillerato) incompleta', 'Técnica o tecnológica completa', 'Técnica o tecnológica incompleta'
            ]],
            value='Ninguno'
        ),

        html.Label("Estrado de la vivienda"),
        dcc.Dropdown(
            id='FAMI_ESTRATOVIVIENDA',
            options=[{'label': x, 'value': x} for x in [
                'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6'
            ]],
            value='Estrato 2'
        ),

        html.Br(), html.Div(id='salida_prediccion', style={
            'textAlign': 'center', 'fontSize': '22px',
            'fontWeight': 'bold', 'color': '#2e7d32'
        }),

            html.Br(),
        dcc.Graph(id='grafico_gauge'),
    ])

#--------------Gráfica Pregunta 1 ----------------

def layout_graficos():
    return html.Div([
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

#----------------------------
#PESTAÑAS
#----------------------------
app.layout = html.Div([
    dcc.Tabs(id='tabs-principal', value='tab-modelo', children=[
        dcc.Tab(label='Predicción', value='tab-modelo'),
        dcc.Tab(label='Visualización', value='tab-graficos')
    ]),
    html.Div(id='contenido-tabs')
])

#------------- Cambiar de pestañas -------------
@app.callback(
    Output('contenido-tabs', 'children'),
    Input('tabs-principal', 'value')
)
def render_tab(tab):
    if tab == 'tab-modelo':
        return layout_modelo()
    elif tab == 'tab-graficos':
        return layout_graficos()

#----------------------------
#CALLBACKS
#----------------------------
#------------------Modelo------------------
@app.callback(
    [
        Output('salida_prediccion', 'children'),
        Output('grafico_gauge', 'figure')
    ],
    [
        Input('FAMI_TIENELAVADORA_Si', 'value'),
        Input('FAMI_TIENEAUTOMOVIL_Si', 'value'),
        Input('FAMI_TIENECOMPUTADOR_Si', 'value'),
        Input('COLE_AREA_UBICACION_URBANO', 'value'),
        Input('ESTU_COD_RESIDE_MCPIO', 'value'),
        Input('ESTU_COD_MCPIO_PRESENTACION', 'value'),
        Input('FAMI_PERSONASHOGAR', 'value'),
        Input('ESTU_TIPODOCUMENTO', 'value'),
        Input('FAMI_CUARTOSHOGAR', 'value'),
        Input('FAMI_EDUCACIONMADRE', 'value'),
        Input('FAMI_ESTRATOVIVIENDA', 'value')
    ]
)


def predecir(lavadora, carro, computador, sede, cod_res, cod_pres, personas, doc, cuartos, madre, estrato):
    try:
        # Validación de los códigos
        if not str(cod_res).isdigit() or not str(cod_pres).isdigit():
            return "Error: los códigos de municipio deben ser números enteros.", go.Figure()
        # Convertir a int
        cod_res_int = int(cod_res)
        cod_pres_int = int(cod_pres)
        base = {
            'FAMI_TIENELAVADORA_Si': lavadora,
            'FAMI_TIENEAUTOMOVIL_Si': carro,
            'FAMI_TIENECOMPUTADOR_Si': computador,
            'COLE_AREA_UBICACION_URBANO': sede,
            'ESTU_COD_RESIDE_MCPIO': cod_res_int,
            'ESTU_COD_MCPIO_PRESENTACION': cod_pres_int,
            'FAMI_PERSONASHOGAR': personas,
            'ESTU_TIPODOCUMENTO': doc,
            'FAMI_CUARTOSHOGAR': cuartos,
            'FAMI_EDUCACIONMADRE': madre,
            'FAMI_ESTRATOVIVIENDA': estrato
        }

        df_input = pd.DataFrame([base])
        X_final = preprocessor.transform(df_input)
        pred = modelo.predict(X_final)[0][0]

        
        texto = f"Probabilidad Predicha: {pred:.2f}"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Probabilidad Predicha"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#2e7d32"},
                'steps': [
                    {'range': [0, 0.33], 'color': "#e57373"},
                    {'range': [0.33, 0.66], 'color': "#fff176"},
                    {'range': [0.66, 1.0], 'color': "#81c784"}
                ],
            }
        ))

        fig.update_layout(height=300, margin={'t': 40, 'b': 0, 'l': 0, 'r': 0})

        return texto, fig

    except Exception as e:
        return f"Error al predecir: {str(e)}", go.Figure()

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
            points=False,
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

#----------------------------
#EJECUTAR
#----------------------------
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8050)


