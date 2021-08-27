from inspect import FrameInfo
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


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


# ================================================== https://dash.plotly.com/urls ==================================================


# ---- fetch data object ----x
with open('../data objects/batch_processing_object_with_encodings.pkl', 'rb') as file2:
    # ed = pickle.load(file2)
    ed = pd.read_pickle(file2)
spf = summary_plots_and_figures(ed)
spf.create_performance_groupings(10)
spf.random_participants_sample(10)
# ---- fetch data object ----x

# # ---- colour schemes ----x
colours_choice = [
            {'label': 'Plasma',     'value': 'Plasma'},
            {'label': 'Plotly3',    'value': 'Plotly3'},
            {'label': 'viridis',    'value': 'viridis'},
            {'label': 'Inferno',    'value': 'Inferno'},
            {'label': 'Turbo',      'value': 'Turbo'},
            {'label': 'Blackbody',  'value': 'Blackbody'},
            {'label': 'RdBu',       'value': 'RdBu'},
            {'label': 'Aggrnyl',    'value': 'Aggrnyl'},
            {'label': 'RdYlBu',     'value': 'RdYlBu'},
            {'label': 'Portland',   'value': 'Portland'},
            {'label': 'Tropic',     'value': 'Tropic'},
        ]
clrs = {
    'Plasma':      sum([px.colors.sequential.Plasma*3], []),
    'Plotly3':     sum([px.colors.sequential.Plotly3*3], []),
    'viridis':     sum([px.colors.sequential.Viridis*3], []),
    'Inferno':     sum([px.colors.sequential.Inferno*3], []),
    'Turbo':       sum([px.colors.sequential.Turbo*3], []),
    'Blackbody':   sum([px.colors.sequential.Blackbody*3], []),
    'RdBu':        sum([px.colors.sequential.RdBu*3], []),
    'Aggrnyl':     sum([px.colors.sequential.Aggrnyl*3], []),
    'RdYlBu':      sum([px.colors.diverging.RdYlBu*3], []),
    'Portland':    sum([px.colors.diverging.Portland*3], []),
    'Tropic':      sum([px.colors.diverging.Tropic*3], [])}
# # ---- colour schemes ----x



app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# =================================================== Universal Elements ==================================================
navigation_menu = dbc.Col([ 
    html.Br(), html.Hr(),
    dbc.Row([
    dbc.Col([
        # html.Br(),
        dbc.Col(dcc.Link('Home page', href='/')),
        dbc.Col(dcc.Link('Model Free Analysis', href='/page-1')),
        dbc.Col(dcc.Link('Theoretical Statistics & Neuroscience', href='/page-2'))], width=8),
        dbc.Col([html.Img(src=app.get_asset_url('uct-uhs.png'),style={'width':'60%'})], width=4),
    ]),
    html.Hr(), html.Br()
])
# =================================================== Universal Elements ==================================================



# ====================================================== Landing Page =====================================================
index_page = dbc.Container([
    html.Div('Neuropsychological Data Analysis', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'center'}),
    dbc.Row([html.Div('We\'re a cognitive neuroscience research unit comparing human & Artificial Intelligence learning in dynamic decision making tasks under uncertainty. We\'re utilising Statistical Learning to reverse engineer neurological executive functions. This interactive tools aims to aid in one\'s intuitive understanding about relationships in the data. In affliantion with University of Helsinki & University of Cape Town.', style={'color': '#555555', 'fontSize': 14, 'margin-right':'20px', 'text-align': 'center'})], justify="center", align="center"),
    navigation_menu,
    dbc.Row(html.Iframe(id="embedded-pdf", src="assets/description.pdf", height='900', width='90%'), justify="center", align="center")
])
# ====================================================== Landing Page =====================================================



# ================================================== Model-Free Analysis ==================================================
page_1_layout = dbc.Container([
    html.Div('Model Free Analysis', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'left'}),
    html.Div('This dashboard allows one to build a statistical intuition about the relationships in the data.', style={'color': '#555555', 'fontSize': 14, 'margin-right':'20px', 'text-align': 'left'}),
    navigation_menu,
    dbc.Row([
        dbc.Col([
            html.Div('', style={'margin-top':'30px'}),
            dbc.Row([
                dbc.Col(html.Div('Variable X'), width=4),
                dbc.Col([dcc.Dropdown(id='varx', options=spf.continuous_vars, value='wcst_accuracy')])
                ]),
            dbc.Row([
                dbc.Col(html.Div('Variable Y'), width=4),
                dbc.Col([dcc.Dropdown(id='vary', options=spf.continuous_vars,value='navon_perc_correct')])
                ]),
            dbc.Row([
                dbc.Col(html.Div('Group'), width=4),
                dbc.Col([dcc.Dropdown(id='group', options=spf.categorical_vars, value='')])
                ])      
        ], width=4),
        dbc.Col(dcc.Graph(id='summary_plot_1'))
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div('Colour scheme'), width=4),
                dbc.Col([dcc.Dropdown(id='colours', options=colours_choice, value='Plasma')])
                            ]),
            dbc.Row([
                dbc.Col(html.Div('Trial batch size'), width=4),
                dbc.Col(dcc.Slider(id='wcst_groups', min=1, max=10, step=1, value=1)) # options: [1,2,3,4,5]
                        ]),
        ], width=6),
        dbc.Col([
                daq.ToggleSwitch(id='vlines', value=False, label='Show forced error points', color='#ABE2FB', labelPosition='left'),
                dbc.Row([
                    dbc.Col(html.Div('N groups'), width=3),
                    dbc.Col(dcc.Slider(id='n_groups', min=0, max=10, step=1, value=5))]),
        ])
    ]),
    dbc.Col(dcc.Graph(id='wcst_plot')),
    dbc.Row([
        dbc.Col([
            html.Div(id='anova_table'), 
            html.Div([html.Div(id='summary_table')])],
            width=8),
        dbc.Col(dcc.Graph(id='group_pie')),
    ]),
    dbc.Col(dcc.Graph(id='summary_plot_2'))
])

@app.callback(
    Output(component_id='summary_table',                component_property='children'),
    Output(component_id='anova_table',                  component_property='children'),
    Output(component_id='wcst_plot',                    component_property='figure'),
    Output(component_id='group_pie',                    component_property='figure'),
    Output(component_id='summary_plot_1',               component_property='figure'),
    Output(component_id='summary_plot_2',               component_property='figure'),

    Input(component_id='colours',                       component_property='value'),
    Input(component_id='wcst_groups',                   component_property='value'),
    Input(component_id='n_groups',                      component_property='value'),
    Input(component_id='varx',                          component_property='value'),
    Input(component_id='vary',                          component_property='value'),
    Input(component_id='group',                         component_property='value'),
    Input(component_id='vlines',                        component_property='value')
    )


def page_1_dropdown(colours, wcst_groups, n_groups, varx, vary, group, vlines):
    spf.create_performance_groupings(n_groups)
    spf.random_participants_sample(n_groups)
    spf.compute_wcst_performance_trial_bins(n_groups, use_seq=True, g=wcst_groups)
    x    = spf.compute_summary_stats(data=spf.ed.summary_table, value_var=varx, group_var=group, resetIndex=True)
    dt1  = html.Div([
        html.Div('Average Scores per Group', style={'color': '#555555', 'fontSize': 20, 'margin':'10px'}),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in x.columns],
            data=x.to_dict('records'),
            style_cell={'overflow': 'scroll', 'textOverflow': 'ellipsis'},
            style_table={'overflow': 'scroll'},
            style_header=dict(backgroundColor="#ddf3fc"),
            style_data=dict(backgroundColor="#fffafa")
        )])
    x    = spf.ANOVA(data=spf.ed.summary_table, group_var=group, value_var=varx)
    dt2  = html.Div([
            html.Div('ANOVA: ' + varx, style={'color': '#555555', 'fontSize': 20, 'margin':'10px'}),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in x.columns],
                data=x.to_dict('records'),
                style_cell={'overflow': 'scroll', 'textOverflow': 'ellipsis'},
                style_table={'overflow': 'scroll'},
                style_header=dict(backgroundColor="#ddf3fc"),
                style_data=dict(backgroundColor="#fffafa")
            )
        ])
    wcst = spf.wcst_performance_plot(group=group, colours=clrs[colours], mean_plot=True, show_vlines=vlines)['figure'] 
    grp  = spf.basic_pie_chart(df=spf.ed.summary_table, dummy_var=group, colors=clrs[colours], labels=None, title='Group Distribution: '+group) 
    sm1  = spf.scatter_plot(data=spf.ed.summary_table, group_var=group, xvar=varx, yvar=vary, xlab=varx, cols=clrs[colours], ylab=vary, title='Raw Data')
    sm2  = spf.distribution_plot(data=spf.ed.summary_table, nbinsx=50, xvar=varx, group_var=group, title='Distributions over: ' + varx, cols=clrs[colours]) 
    return dt1, dt2, wcst, grp, sm1, sm2
# ================================================== Model-Free Analysis ==================================================




# =================================================== Theoretical Models ===================================================
page_2_layout = dbc.Container([
    html.Div('Theoretical Neuroscience', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'left'}),
    html.Div('Theoretical Neuroscience is increasingly taken as mathematical approximations to neurological behaviour. In our study we adopt this strategy to simulate executive functions with computational models.', style={'color': '#555555', 'fontSize': 14, 'margin-right':'20px', 'text-align': 'left'}),
    navigation_menu,
    html.H1('Page 2'),
    dcc.RadioItems(
        id='page-2-radios',
        options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
        value='Orange'
    ),
])

@app.callback(dash.dependencies.Output('page-2-content', 'children'),
              [dash.dependencies.Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')













# # ------------------------- Anylsis: Model Free -------------------------x
# model_free = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             html.Div('', style={'margin-top':'30px'}),
#             dbc.Row([
#                 dbc.Col(html.Div('Variable X'), width=4),
#                 dbc.Col([dcc.Dropdown(id='varx', options=spf.continuous_vars, value='wcst_accuracy')])
#                 ]),
#             dbc.Row([
#                 dbc.Col(html.Div('Variable Y'), width=4),
#                 dbc.Col([dcc.Dropdown(id='vary', options=spf.continuous_vars,value='navon_perc_correct')])
#                 ]),
#             dbc.Row([
#                 dbc.Col(html.Div('Group'), width=4),
#                 dbc.Col([dcc.Dropdown(id='group', options=spf.categorical_vars, value='')])
#                 ])      
#         ], width=4),
#         dbc.Col(dcc.Graph(id='summary_plot_1'))
#     ]),
#     dbc.Row([
#     dbc.Col([
#         dbc.Row([
#             dbc.Col(html.Div('Colour scheme'), width=4),
#             dbc.Col([dcc.Dropdown(id='colours', options=colours_choice, value='Plasma')])
#                         ]),
#         dbc.Row([
#             dbc.Col(html.Div('Trial batch size'), width=4),
#             dbc.Col(dcc.Slider(id='wcst_groups', min=1, max=10, step=1, value=1)) # options: [1,2,3,4,5]
#                     ]),
#     ], width=6),
#     dbc.Col([
#             daq.ToggleSwitch(id='vlines', value=False, label='Show forced error points', color='#ABE2FB', labelPosition='left'),
#             dbc.Row([
#                 dbc.Col(html.Div('N groups'), width=3),
#                 dbc.Col(dcc.Slider(id='n_groups', min=0, max=10, step=1, value=5))]),
#     ])
#     ]),
#     dbc.Col(dcc.Graph(id='wcst_plot')),
#     dbc.Row([
#         dbc.Col([
#             html.Div(id='anova_table'), 
#             html.Div([html.Div(id='summary_table')])],
#             width=8),
#         dbc.Col(dcc.Graph(id='group_pie')),
#     ]),
#     dbc.Col(dcc.Graph(id='summary_plot_2'))
# ]),
# # ------------------------- Anylsis: Model Free -------------------------x

# app.layout = dbc.Container([
#     html.Div('Neuropsychological Data Analysis', style={'color': '#555555', 'fontSize': 25, 'margin':'15px', 'text-align': 'center'}),
#     dbc.Row([html.Div('We\'re a cognitive neuroscience research unit comparing human & Artificial Intelligence learning in dynamic decision making tasks under uncertainty. We\'re utilising Statistical Learning to reverse engineer neurological executive functions. This interactive tools aims to aid in one\'s intuitive understanding about relationships in the data. In affliantion with University of Helsinki & University of Cape Town.', style={'color': '#555555', 'fontSize': 12, 'margin-right':'20px', 'text-align': 'center'})], justify="center", align="center"),
#     html.Br(),
#     dbc.Row([
#         dbc.Col([
#             dbc.Col(dcc.Location(id='url', refresh=False)),
#             dbc.Col(dcc.Link('Model Free Analysis', href='/model-free')),
#             dbc.Col(dcc.Link('Theoretical Statistics & Neuroscience', href='/theory')),
#             dbc.Col(html.Label([html.A('Connect with us!',
#                 href='https://raw.githubusercontent.com/ZachWolpe/Dynocog/main/images/description.png')],
#                 style={'color': '#555555', 'fontSize': 12, 'margin-right':'20px', 'text-align': 'center'}))
#         ], width=8),
#         dbc.Col([
#         html.Img(src=app.get_asset_url('uct-uhs.png'),style={'width':'60%'}
#         )], width=4),
#     ]),

   

    



#     # content will be rendered in this element
#     html.Div(id='page-content')
# ])




# # @app.callback(dash.dependencies.Output('page-content', 'children'),
# #               [dash.dependencies.Input('url', 'pathname')])

# @app.callback(
#     Output(component_id='page-content', component_property='children'),
    
#     # Output(component_id='summary_table',                component_property='children'),
#     # Output(component_id='anova_table',                  component_property='children'),
#     # Output(component_id='wcst_plot',                    component_property='figure'),
#     # Output(component_id='group_pie',                    component_property='figure'),
#     # Output(component_id='summary_plot_1',               component_property='figure'),
#     # Output(component_id='summary_plot_2',               component_property='figure'),

#     Input(component_id='url', component_property='pathname'),
#     # Input(component_id='colours',                       component_property='value'),
#     # Input(component_id='wcst_groups',                   component_property='value'),
#     # Input(component_id='n_groups',                      component_property='value'),
#     # Input(component_id='varx',                          component_property='value'),
#     # Input(component_id='vary',                          component_property='value'),
#     # Input(component_id='group',                         component_property='value'),
#     # Input(component_id='vlines',                        component_property='value')
#     )

    
# def display_page(pathname#, colours, wcst_groups, n_groups, varx, vary, group, vlines
# ):
#     print(pathname)
#     if pathname=='/model-free':
#         return model_free
#     return dbc.Container([
#         html.H3('You are on page {}'.format(pathname)), 
#     ])



# if __name__=='__main__':
#     app.run_server(debug=True, host = '127.0.0.1') 