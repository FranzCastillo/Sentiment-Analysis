import base64
from io import BytesIO
import random

import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from wordcloud import WordCloud
import plotly.graph_objects as go

from utils.data import Data

PURPLE = '#7b72aa'
CYAN = '#72aaa8'
RED = '#e37353'
YELLOW = '#fada43'
BG = '#f1eae1'
DISABLED = '#d3d3d3'
BLACK = '#2d2d2d'

data = Data('./src/app/data/disaster.csv')

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
        html.Div(  # Frecuencia del sentimiento
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
            },
            children=[
                html.H2(
                    'Frecuencia del tipo de sentimiento',
                ),
                html.Div(
                    children=[
                        html.P(
                            'Seleccione el tipo de sentimiento:',
                        ),
                        dcc.Dropdown(
                            id='sentiment-type',
                            options=[
                                {'label': 'Negativo', 'value': 'negative'},
                                {'label': 'Neutral', 'value': 'neutral'},
                                {'label': 'Positivo', 'value': 'positive'},
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
                                dcc.Graph(id='bar-plot-sentiment'),
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
                                            id='tweets-container-sentiments',
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
        html.Div(  # Wordcloud
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
            },
            children=[
                html.H2('Wordcloud según tipo de desastre'),
                html.Div(
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'justifyContent': 'space-between',
                    },
                    children=[
                        dcc.Input(
                            id='wordcloud-number',
                            type='number',
                            placeholder='Número de palabras',
                            style={'width': '30%'}
                        ),
                        dcc.Dropdown(
                            id='wordcloud-disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            placeholder='Seleccione el tipo de desastre',
                            style={'width': '30%'}
                        ),
                        html.Button('Generar Wordcloud', id='generate-wordcloud', n_clicks=0, style={'width': '30%'})
                    ]
                ),
                html.Div(
                    id='wordcloud-container',
                    style={
                        'marginTop': '2rem',
                        'width': '40%',
                        # Align in center
                        'marginLeft': 'auto',
                        'marginRight': 'auto'
                    }
                )
            ]
        ),
        html.Div(  # Predicción de desastres
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
                                html.Button('Submit', id='submit-button', n_clicks=0,
                                            style={'marginTop': '1rem', 'width': '100%', 'backgroundColor': CYAN})
                            ]
                        ),
                        html.P(id='prediction-result',
                               style={'width': '50%', 'fontSize': '2rem', 'fontWeight': 'bold', 'textAlign': 'center'})
                    ]
                )
            ]
        ),
        html.Div( # Word frequency
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
                'marginTop': '2rem',
            },
            children=[
                html.H2('Palabras según tipo de desastre'),
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'},
                    children=[
                        dcc.Input(
                            id='word-number',
                            type='number',
                            placeholder='Número de palabras',
                            style={'width': '30%'},
                            value=10
                        ),
                        dcc.Dropdown(
                            id='word-disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            placeholder='Seleccione el tipo de desastre',
                            style={'width': '30%'},
                            value=1
                        ),
                    ]
                ),
                dcc.Graph(id='word-frequency-graph')
            ]
        ),
        html.Div( # Keyword frequency
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
                'marginTop': '2rem',
            },
            children=[
                html.H2('Keywords según tipo de desastre'),
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'},
                    children=[
                        dcc.Input(
                            id='keyword-number',
                            type='number',
                            placeholder='Número de palabras',
                            style={'width': '30%'},
                            value=10
                        ),
                        dcc.Dropdown(
                            id='keyword-disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            placeholder='Seleccione el tipo de desastre',
                            style={'width': '30%'},
                            value=1
                        ),
                    ]
                ),
                dcc.Graph(id='keyword-frequency-graph')
            ]
        ),
        html.Div( # Sentiment Distribution
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
                'marginTop': '2rem',
            },
            children=[
                html.H2('Distribución de sentimiento según tipo de desastre'),
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'},
                    children=[
                        dcc.Dropdown(
                            id='violin-disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            placeholder='Seleccione el tipo de desastre',
                            style={'width': '30%'},
                            value=1
                        ),
                    ]
                ),
                dcc.Graph(id='violin-plot')
            ]
        ),
        html.Div( # Sentiment pie
            style={
                'backgroundColor': BG,
                'border': '2px solid black',
                'padding': '1rem',
                'marginBottom': '2rem',
                'marginTop': '2rem',
            },
            children=[
                html.H2('Distribución de sentimiento según tipo de desastre'),
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'},
                    children=[
                        dcc.Dropdown(
                            id='pie-disaster-type',
                            options=[
                                {'label': 'Natural', 'value': 1},
                                {'label': 'Metáfora', 'value': 0},
                            ],
                            placeholder='Seleccione el tipo de desastre',
                            style={'width': '30%'},
                            value=1
                        ),
                    ]
                ),
                dcc.Graph(id='pie-plot')
            ]
        ),
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


def _get_sentiment_color(sentiment: str) -> str:
    if sentiment == 'positive':
        return CYAN
    elif sentiment == 'neutral':
        return YELLOW
    return RED  # Negative


@app.callback(
    Output('bar-plot-sentiment', 'figure'),
    Input('sentiment-type', 'value')
)
def update_bar_plot_sentiment(selected_type):
    frequencies = data.get_sentiment_frequency()
    colors = [_get_sentiment_color(sentiment) if sentiment == selected_type else DISABLED for sentiment in
              frequencies['sentiment']]

    fig = px.bar(
        frequencies.assign(sentiment=frequencies['sentiment'].map(
            {'positive': 'Positivo', 'neutral': 'Neutral', 'negative': 'Negativo'})),
        x='sentiment',
        y='count',
        color='sentiment',
        color_discrete_sequence=colors,
        labels={'sentiment': 'Tipo de sentimiento', 'count': 'Frecuencia'}
    ).update_layout(
        showlegend=False,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )

    return fig


@app.callback(
    Output('tweets-container-sentiments', 'children'),
    Input('sentiment-type', 'value')
)
def update_tweets_sentiment(selected_type):
    if selected_type is not None:
        filtered_df = data.df[data.df['sentiment'] == selected_type]
        tweets = filtered_df['text'].tolist()
        return [html.P(tweet) for tweet in tweets]
    else:
        return [html.P('Seleccione un tipo de sentimiento')]


@app.callback(
    Output('wordcloud-container', 'children'),
    Input('generate-wordcloud', 'n_clicks'),
    State('wordcloud-number', 'value'),
    State('wordcloud-disaster-type', 'value')
)
def generate_wordcloud(n_clicks, num_words, disaster_type):
    if n_clicks > 0 and num_words and disaster_type is not None:
        filtered_df = data.df[data.df['target'] == disaster_type]
        text = ' '.join(filtered_df['text_clean'].tolist())

        if disaster_type == 1:
            colors = ['#e37353', '#f18c6a', '#cc6042', '#d97b4e', '#b34f36']
        else:
            colors = ['#72aaa8', '#89c1bf', '#5d9492', '#68a1a0', '#4f8381']

        # Define a custom color function to select colors from the list
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return random.choice(colors)

        # Generate the word cloud with the custom color function
        wordcloud = WordCloud(max_words=num_words, background_color=BG, color_func=color_func).generate(text)

        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        encoded_image = base64.b64encode(img.getvalue()).decode()

        return html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%'})

    return html.P('Ingrese un número de palabras y seleccione un tipo de desastre para generar el wordcloud.')


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

@app.callback(
    Output('word-frequency-graph', 'figure'),
    [Input('word-number', 'value'),
     Input('word-disaster-type', 'value')]
)
def update_word_freq(word_number, disaster_type):

    if word_number is None or disaster_type is None:
        return {}

    # Filter the dataframe by disaster type
    filtered_df = data.df[data.df['target'] == disaster_type]

    # Get the word frequencies
    word_frequencies = {}
    for text in filtered_df['text_clean']:
        for word in text.split():
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1

    # Order the word frequencies
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Select the most common words
    most_common_words = dict(sorted_word_frequencies[:word_number])

    color = CYAN if disaster_type == 1 else RED

    # Create the figure
    fig = px.bar(
        x=list(most_common_words.keys()),
        y=list(most_common_words.values()),
        labels={'x': 'Palabra', 'y': 'Frecuencia'},
        color_discrete_sequence=[color] * word_number
    ).update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )
    
    return fig


@app.callback(
    Output('keyword-frequency-graph', 'figure'),
    [Input('keyword-number', 'value'),
     Input('keyword-disaster-type', 'value')]
)
def update_keyword_freq(word_number, disaster_type):

    if word_number is None or disaster_type is None:
        return {}

    # Filter the dataframe by disaster type
    filtered_df = data.df[data.df['target'] == disaster_type]

    # Get the word frequencies
    word_frequencies = {}
    for keyword in filtered_df['keyword']:
        try:
            keyword_parsed = str(keyword).lower().replace('%20', '_')
            if keyword_parsed in word_frequencies:
                word_frequencies[keyword_parsed] += 1
            else:
                word_frequencies[keyword_parsed] = 1
        except:
            continue

    # Order the word frequencies
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Select the most common words
    most_common_words = dict(sorted_word_frequencies[:word_number])

    color = CYAN if disaster_type == 1 else RED

    # Create the figure
    fig = px.bar(
        x=list(most_common_words.keys()),
        y=list(most_common_words.values()),
        labels={'x': 'Keyword', 'y': 'Frecuencia'},
        color_discrete_sequence=[color] * word_number
    ).update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )
    
    return fig

@app.callback(
    Output('violin-plot', 'figure'),
    Input('violin-disaster-type', 'value')
)
def update_violin_plot(disaster_type):

    if disaster_type is None:
        return {}

    filtered_df = data.df[data.df['target'] == disaster_type]

    fig = px.violin(
        filtered_df,
        y='sentiment_score',
        x='sentiment',
        color='sentiment',
        box=True,
        points="all",
        labels={'sentiment_score': 'Sentiment Score', 'sentiment': 'Sentiment Type'},
        title='Distribución del Sentiment Score por tipo de sentimiento'
    ).update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )

    return fig

@app.callback(
    Output('pie-plot', 'figure'),
    Input('pie-disaster-type', 'value')
)
def update_pie_plot(disaster_type):
    
    if disaster_type is None:
        return {}
    
    filtered_df = data.df[data.df['target'] == disaster_type]

    fig = px.pie(
        filtered_df,
        names='sentiment',
        title='Distribución de sentimiento',
        labels={'sentiment': 'Sentimiento'},
        color='sentiment',
        color_discrete_map={'positive': CYAN, 'neutral': YELLOW, 'negative': RED}
    ).update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=BLACK,
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
