from dash import Dash, html, dcc, Input, Output

from utils.data import Data

PURPLE = '#7b72aa'
CYAN = '#72aaa8'
RED = '#e37353'
YELLOW = '#fada43'
BG = '#f1eae1'
BLACK = '#2d2d2d'

# Assuming the Data class is correctly implemented in utils.data and the CSV file is available
data = Data('data/disaster.csv')

app = Dash()

app.layout = html.Div(
    style={
        'backgroundColor': BG,
        'height': '100vh',
        'width': '100vw',
        'margin': '0',
        'padding': '0',
        'fontFamily': 'Arial, sans-serif',
        'border': '1px solid black'
    },
    children=[
        html.H1(
            'Desastres... ¿Naturales o metáforas?',
            style={
                'textAlign': 'center',
                'color': BLACK,
                'backgroundColor': CYAN,
                'padding': '1rem',
                'border': '1px solid black'
            }
        ),
        dcc.Input(
            id='input-text',
            type='text',
            placeholder='Enter text here...',
            style={'width': '80%', 'padding': '0.5rem', 'margin': '1rem auto', 'display': 'block'}
        ),
        html.Div(id='prediction-result', style={'textAlign': 'center', 'color': BLACK, 'padding': '1rem'})
    ]
)


# Define the callback function to update the prediction result
@app.callback(
    Output('prediction-result', 'children'),
    Input('input-text', 'value')
)
def update_prediction(input_text):
    if input_text:
        # Use the predict method from the Data class to get the prediction
        prediction = data.predict([input_text])[0]
        return f'Prediction: {prediction}'
    return 'Enter text to get prediction'


if __name__ == '__main__':
    app.run(debug=True)
