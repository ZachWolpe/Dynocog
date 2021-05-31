import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np


np.random.seed(2020)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__)


markdown_text = '''
    ### Dash and Markdown

    Dash apps can be written in Markdown.
    Dash uses the [CommonMark](http://commonmark.org/)
    specification of Markdown.
    Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
    if this is your first introduction to Markdown!
'''


app.layout = html.Div([
    dcc.Markdown(children=markdown_text),
    dcc.Dropdown(
    options=[
        {'label': 'New York City', 'value': 'NYC'},
        {'label': 'Montreal', 'value': 'MTL'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    searchable=False
    ),
    dcc.Graph(id="graph"),
    html.P("Mean:"),
    dcc.Slider(id="mean", min=-3, max=3, value=0, 
               marks={-3: '-3', 3: '3'}),
    html.P("Standard Deviation:"),
    dcc.Slider(id="std", min=1, max=3, value=1, 
               marks={1: '1', 3: '3'}),
], style={'columnCount': 2})

@app.callback(
    Output("graph", "figure"), 
    [Input("mean", "value"), 
     Input("std", "value")])
def display_color(mean, std):
    data = np.random.normal(mean, std, size=500)
    fig = px.histogram(data, nbins=30, range_x=[-10, 10])
    return fig

app.run_server(host = '127.0.0.1', debug=True)