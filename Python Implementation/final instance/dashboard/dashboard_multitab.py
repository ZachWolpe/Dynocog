


from typing import Text
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
from future.utils import with_metaclass

from pandas.io.formats import style
from dash_core_components.Dropdown import Dropdown
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import re
import sys
sys.path.append('../process data/')
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as pyo
import plotly.express as px
from encode_processed_data import encode_data

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from summary_plots_and_figures import summary_plots_and_figures




app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    html.Div('Neuropsychological Data Analysis', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'center'}),
    dbc.Row([html.Div('We\'re a cognitive neuroscience research unit comparing human & Artificial Intelligence learning in dynamic decision making tasks under uncertainty. We\'re utilising Statistical Learning to reverse engineer neurological executive functions. This interactive tools aims to aid in one\'s intuitive understanding about relationships in the data. In affliantion with University of Helsinki & University of Cape Town.', style={'color': '#555555', 'fontSize': 12, 'margin-right':'20px', 'text-align': 'center'})], justify="center", align="center"),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Col(dcc.Location(id='url', refresh=False)),
            dbc.Col(dcc.Link('Model Free Analysis', href='/model-free')),
            dbc.Col(dcc.Link('Theoretical Statistics & Neuroscience', href='/theory')),
            dbc.Col(html.Label([html.A('Connect with us!',
                href='https://raw.githubusercontent.com/ZachWolpe/Dynocog/main/images/description.png')],
                style={'color': '#555555', 'fontSize': 12, 'margin-right':'20px', 'text-align': 'center'}))
        ], width=8),
        dbc.Col([
        html.Img(src=app.get_asset_url('uct-uhs.png'),style={'width':'60%'}
        )], width=4),
    ]),
    html.Div('detla'),
    html.Img(src=app.get_asset_url('uct-uhs.png'), style={'height':'25%', 'width':'25%'}),
    dbc.Row([
        dbc.Col([html.Div('delta'), html.Div('delta2'), html.Div('delta3')], width=6),
        dbc.Col([html.Div('delta')], width=6)
    ]),
   

    
    


    



    # content will be rendered in this element
    html.Div(id='page-content')
])


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    return dbc.Container([
        html.H3('You are on page {}'.format(pathname)), 
    ])



if __name__=='__main__':
    app.run_server(debug=True, host = '127.0.0.1') 