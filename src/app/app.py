import plotly.express as px
from dash import Dash, html, dcc, Input, Output

from utils.data import Data

PURPLE = '#7b72aa'
CYAN = '#72aaa8'
RED = '#e37353'
YELLOW = '#fada43'
BG = '#f1eae1'
DISABLED = '#d3d3d3'
BLACK = '#2d2d2d'

# Assuming the Data class is correctly implemented in utils.data and the CSV file is available
data = Data('data/disaster.csv')

app = Dash()

app.layout = html.Div(
    style={
        'backgroundColor': RED,
        'height': '95vh',
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
                'shadow': '2px 2px 2px black'
            }
        ),
        html.Div(  # Frecuencia del tipo de desastres
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
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
                                    },
                                    children=[
                                        html.P('Tweets'),
                                        html.Div(
                                            id='tweets-container',
                                            style={
                                                'overflowY': 'scroll',
                                                'height': '50vh',
                                                'border': '1px solid black',
                                                'padding': '1rem 1rem 1rem 0',
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
    ]
)


# Define the callback function to update the prediction result
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
    Output('tweets-container', 'children'),
    Input('disaster-type', 'value')
)
def update_tweets(selected_type):
    if selected_type is not None:
        filtered_df = data.df[data.df['target'] == selected_type]
        tweets = filtered_df['text'].tolist()
        return [html.P(tweet) for tweet in tweets]
    else:
        return [html.P('Seleccione un tipo de desastre')]




if __name__ == '__main__':
    app.run(debug=True)
