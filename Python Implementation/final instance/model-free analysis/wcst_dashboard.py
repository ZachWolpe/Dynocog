
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
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    
np.random.seed(42)
random_x = np.random.randint(1,101,1000)
random_y = np.random.randint(1,101,1000)

# ---- fetch data object ----x
with open('../data objects/batch_processing_object_with_encodings.pkl', 'rb') as file2:
    ed = pickle.load(file2)



# ============================================== Demographics ==============================================
demographic_groups = [
    {'label': 'gender', 'value': 'gender_a'}, {'label': 'handedness', 'value': 'handedness_a'}, 
    {'label': 'education', 'value': 'education_a'}, {'label': 'age', 'value': 'age_group'}]

demographic_cont_vars = [
    {'label': 'age', 'value': 'age_a'}, {'label': 'income', 'value': 'income_a'}, 
    {'label': 'computer_hours', 'value': 'computer_hours_a'}, {'label': 'reaction time (ms)', 'value': 'mean_reation_time_ms'}
]

demo_pie_map = {
    'gender_a':     {'dummy_var':'gender_a',        'labels':['male', 'female', 'other'],                       'colors':['steelblue', 'darkred', 'cyan'],                                          'title':'Gender Distribution',     'name':'gender'},
    'education_a':  {'dummy_var':'education_a',     'labels':['university', 'graduate school', 'high school'],  'colors':['rgb(177, 127, 38)', 'rgb(129, 180, 179)', 'rgb(205, 152, 36)'],  'title':'Education Distribution',   'name':'education'},
    'handedness_a': {'dummy_var':'handedness_a',    'labels':['right', 'left', 'ambidextrous'],                 'colors':px.colors.sequential.RdBu,                                         'title':'Handedness Distribution',  'name':'handedness'},
    'age_group':    {'dummy_var':'age_group',       'labels':np.unique(ed.demographics[['age_group']]).tolist(),'colors':px.colors.sequential.GnBu,                                         'title':'Age Distribution',         'name':'age'}
    }
    
demo_continuous_naming = {
     'age_a':                   {'xlab':'Age',                      'ylab':'Count', 'name':'Age Distribution by '},
     'income_a':                {'xlab':'Income',                   'ylab':'Count', 'name':'Income Distribution by '},
     'computer_hours_a':        {'xlab':'Computer hours',           'ylab':'Count', 'name':'Computer Hours Distribution by '},
     'mean_reation_time_ms':    {'xlab':'RT (reaction time (ms))',  'ylab':'Count', 'name':'RT Distribution by '},
}

def pie_chart(dummy_var, labels, colors, title, df=ed.demographics):
    sub    = df[[dummy_var]].value_counts()
    values = sub.tolist()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    fig.update_traces(textfont_size=15, marker=dict(colors=colors, line=dict(color='white', width=0)))
    fig.update(layout_title_text=title)
    fig.update_layout(showlegend=False)
    return fig

def distributional_plots(group_var, continuous_var, xlab, ylab, title, df=ed.demographics):
  
    group_var = demo_pie_map[group_var]
    fig = go.Figure()
    for c in range(len(group_var['labels'])):
        fig.add_trace(go.Histogram(
            x           =   df[continuous_var][df[group_var['dummy_var']] == group_var['labels'][c]],            
            # histnorm  ='percent',
            name        = group_var['labels'][c], 
            marker_color= group_var['colors'][c],
            opacity     = 1
        ))
    fig.update_layout(
        barmode         = 'overlay',
        title_text      = title, 
        xaxis_title_text= xlab, 
        yaxis_title_text= ylab, 
        bargap          = 0.05, 
        bargroupgap     = 0.1 
    )
    fig.update_layout(barmode='group', template='none')
    return fig
# ============================================== Demographics ==============================================


app.layout = dbc.Container([
    html.Div('Demographics of Participants', style={'color': '#555555', 'fontSize': 25, 'margin-top':'10px'}),
    html.Div('Grouping Variable', style={'color': '#555555', 'fontSize': 15, 'margin':'10px'}),
    dcc.Dropdown(
        id='demographic_categorical_var',
        options=demographic_groups,
        value='gender_a'
    ),
    html.Div('Continuous Variable', style={'color': '#555555', 'fontSize': 15, 'margin':'10px'}),
    dcc.Dropdown(
        id='demographic_continuous_var',
        options=demographic_cont_vars,
        value='age_a'
    ),
 
  
    dbc.Row([
        dbc.Col(dcc.Graph(id='demographic_plot_1'), width=4),
        dbc.Col(dcc.Graph(id='demographic_plot_2'))
        
    ])

    
   ,
    # html.Div([
    #     dcc.Graph(id='demographic_plot_1', stwidth=4yle={'width': '90%', 'height': '9%'}),
    #     dcc.Graph(id='demographic_plot_2')
    # ], #style={'columnCount': 2})
    
    ]
    )


@app.callback(
    Output(component_id='demographic_plot_1', component_property='figure'),
    Output(component_id='demographic_plot_2', component_property='figure'),
    Input(component_id='demographic_categorical_var',   component_property='value'),
    Input(component_id='demographic_continuous_var',    component_property='value'),
    )
def update_output_div(cat_var, cont_var):
    demo = demo_pie_map[cat_var]; cv = demo_continuous_naming[cont_var]
    cht = pie_chart(dummy_var=demo['dummy_var'], labels=demo['labels'], colors=demo['colors'], title=demo['title'])
    f1  = distributional_plots(group_var=demo['dummy_var'], continuous_var=cont_var, xlab=cv['xlab'], ylab=cv['ylab'], title=cv['name'] + demo['name'])
    return cht, f1


if __name__=='__main__':
    app.run_server(debug=True, host = '127.0.0.1') 