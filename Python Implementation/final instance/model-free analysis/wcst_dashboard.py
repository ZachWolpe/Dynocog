
from typing import Text
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
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from summary_plots_and_figures import summary_plots_and_figures


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    

# ---- fetch data object ----x
with open('../data objects/batch_processing_object_with_encodings.pkl', 'rb') as file2:
    ed = pickle.load(file2)
spf = summary_plots_and_figures(ed)
spd = ed.summary_table




app.layout = dbc.Container([
    html.Div('Neuropsychological Data Analysis', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'center'}),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div('Variable X'), width=4),
                dbc.Col([dcc.Dropdown(
                    id='varx', 
                    options=spf.continuous_vars, 
                    value='wcst_accuracy')]
                )
                ]),
            dbc.Row([
                dbc.Col(html.Div('Variable Y'), width=4),
                dbc.Col([dcc.Dropdown(
                        id='vary',
                        options=spf.continuous_vars,
                        value='navon_perc_correct'
                    )]
                )
                ]),
            dbc.Row([
                dbc.Col(html.Div('Group'), width=4),
                dbc.Col([dcc.Dropdown(
                        id='group',
                        options=spf.categorical_vars,
                        value='navon_level_of_target'
                    )]
                )
                ]),        
        ], width=7),
        dbc.Col([
            html.Div('Dynocog Research Unit', style={'color': '#555555', 'fontSize': 15, 'margin-bottom':'10px'}),
            dbc.Row([html.Div('We\'re a cognitive neuroscience research unit comparing human vs Artificial Intelligence learning in dynamic task. We\'re working to use Statistical Learning to reverse engineer human cognition.', style={'color': '#555555', 'fontSize': 10, 'margin-right':'20px', 'text-align': 'center'})], justify="center", align="center")])
    ]),
    

        dbc.Col(dcc.Graph(id='summary_plot_1')),
        dbc.Col(dcc.Graph(id='summary_plot_2')),



    html.Div('Demographics of Participants', style={'color': '#555555', 'fontSize': 25, 'margin':'10px'}),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div('Grouping Variable', style={'color': '#555555', 'fontSize': 15, 'margin':'10px'}), width=4),
                 
                dbc.Col([dcc.Dropdown(
                    id='demographic_categorical_var',
                    options=spf.demographic_groups,
                    value='gender_a'
                )]
                )
                ]),
            dbc.Row([
                dbc.Col(html.Div('Continuous Variable', style={'color': '#555555', 'fontSize': 15, 'margin':'10px'}), width=4),
            dbc.Col([dcc.Dropdown(
                    id='demographic_continuous_var',
                    options=spf.demographic_cont_vars,
                    value='age_a'
                )]
                )
                ]),       
        ], width=12),
    ]),

    
   

    

 


        dbc.Row([
        dbc.Col(dcc.Graph(id='demographic_plot_1'), width=4),
        dbc.Col(dcc.Graph(id='demographic_plot_2'))
    ]),


])


@app.callback(
    Output(component_id='summary_plot_1',               component_property='figure'),
    Output(component_id='summary_plot_2',               component_property='figure'),
    Output(component_id='demographic_plot_1',           component_property='figure'),
    Output(component_id='demographic_plot_2',           component_property='figure'),
    Input(component_id='varx',                          component_property='value'),
    Input(component_id='vary',                          component_property='value'),
    Input(component_id='group',                         component_property='value'),
    Input(component_id='demographic_categorical_var',   component_property='value'),
    Input(component_id='demographic_continuous_var',    component_property='value'),
    )
def update_output_div(varx, vary, group, cat_var, cont_var):
    # ---- plots ----x
    sm1  = spf.scatter_plot(data=spd, group_var=group, xvar=varx, yvar=vary, xlab=varx, ylab=vary, title='Raw Data')
    sm2  = spf.distribution_plot(data=spd, nbinsx=50, xvar=varx, group_var=group, title='Distributions over: ' + varx)
    demo = spf.demo_pie_map[cat_var]; cv = spf.demo_continuous_naming[cont_var]
    cht = spf.basic_pie_chart(df=ed.demographics, dummy_var=demo['dummy_var'], labels=demo['labels'], colors=demo['colors'], title=demo['title'])
    f1  = spf.basic_distributional_plots(df=ed.demographics, group_var=demo['dummy_var'], continuous_var=cont_var, xlab=cv['xlab'], ylab=cv['ylab'], title=cv['name'] + demo['name'])
    return sm1, sm2, cht, f1


if __name__=='__main__':
    app.run_server(debug=True, host = '127.0.0.1') 