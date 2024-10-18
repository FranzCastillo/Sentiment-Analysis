import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State

from utils.data import Data

PURPLE = '#7b72aa'
CYAN = '#72aaa8'
RED = '#e37353'
YELLOW = '#fada43'
BG = '#f1eae1'
DISABLED = '#d3d3d3'
BLACK = '#2d2d2d'

data = Data('data/disaster.csv')

app = Dash()

app.layout = html.Div(
    style={
        'backgroundColor': RED,
        'width': '95vw',
        'margin': '0',
        'padding': '2rem',
        'fontFamily': 'Arial, sans-serif',
        'border': '1px solid black'
    },
    children=[
        html.H1(  # Title
            'Desastres... ¿Naturales o metáforas?',
            style={
                'textAlign': 'center',
                'color': BLACK,
                'backgroundColor': CYAN,
                'padding': '1rem',
                'border': '2px solid black',
                'shadow': '2px 2px 2px black',
            }
        ),
        html.Div(  # Frecuencia del tipo de desastres
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
            },
            children=[
                html.H2(
                    'Frecuencia del tipo de desastres',
                ),
                html.Div(
                    children=[
                        html.P(
                            'Seleccione el tipo de desastre:',
                        ),
                        dcc.Dropdown(
                            id='disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            value='',
                            style={'width': '50%', }
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'flexDirection': 'row',
                                'justifyContent': 'space-between',
                                'height': '50vh',
                            },
                            children=[
                                dcc.Graph(id='bar-plot'),
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'flexDirection': 'column',
                                        'justifyContent': 'center',
                                        'padding': '1rem',
                                        'backgroundColor': YELLOW,
                                        'border': '1px solid black',
                                    },
                                    children=[
                                        html.P('Tweets', style={'fontSize': '1.5rem', 'fontWeight': 'bold'}),
                                        html.Div(
                                            id='tweets-container',
                                            style={
                                                'overflowY': 'scroll',
                                                'border': '1px solid black',
                                                'padding': '1rem',
                                                'backgroundColor': BG,
                                            }
                                        )
                                    ]
                                )

                            ]
                        )
                    ]
                ),
            ]
        ),
        html.Div(
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
            },
            children=[
                html.H2(
                    'Predicción de desastres',
                ),
                html.P(
                    'Ingrese un texto para obtener la predicción',
                ),
                html.Div(
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'justifyContent': 'space-between',
                    },
                    children=[
                        html.Div(
                            style={'width': '50%'},
                            children=[
                                dcc.Textarea(
                                    id='input-text',
                                    placeholder="Escribe tu 'tweet' aquí...",
                                    style={'width': '100%', 'height': '10rem'}
                                ),
                                html.Button('Submit', id='submit-button', n_clicks=0, style={'marginTop': '1rem', 'width': '100%', 'backgroundColor': CYAN})
                            ]
                        ),
                        html.P(id='prediction-result', style={'width': '50%', 'fontSize': '2rem', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ]
                )
            ]
        )
    ]
)


@app.callback(
    Output('bar-plot', 'figure'),
    Input('disaster-type', 'value')
)
def update_bar_plot(selected_type):
    frequencies = data.get_frequencies()
    colors = [CYAN if target == selected_type else DISABLED for target in frequencies['target']]

    fig = px.bar(
        frequencies.assign(target=frequencies['target'].map({1: 'Natural', 0: 'Metáfora'})),
        x='target',
        y='count',
        color='target',
        color_discrete_sequence=colors,
        labels={'target': 'Tipo de desastre', 'count': 'Frecuencia'}
    ).update_layout(
        showlegend=False,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )

    return fig


@app.callback(
    Output('prediction-result', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def update_prediction(n_clicks, input_text):
    if n_clicks > 0:
        if input_text:
            prediction = data.predict([input_text])[0]
            prediction_result = 'Natural' if prediction == 1 else 'Metáfora'
            color = CYAN if prediction == 1 else RED
            return html.P([
                "El desastre es... ",
                html.Span(prediction_result, style={'color': color})
            ])
        return 'Enter text to get prediction'
    return ''

if __name__ == '__main__':
    app.run(debug=True)
