import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np

app = dash.Dash()
    
np.random.seed(42)
random_x = np.random.randint(1,101,1000)
random_y = np.random.randint(1,101,1000)


app.layout = html.Div([
    dcc.Graph(id='my scatter',
        figure={'data': [go.Scatter(x=random_x, y=random_y, mode='markers', marker={'size':5, 'color':'steelblue'})],
                'layout':go.Layout(title='My Scatter Plot')}),
    dcc.Graph(id='my scatter 2',
        figure={'data': [go.Scatter(x=random_y, y=random_x, mode='markers', marker={'size':3, 'color':'darkred'})],
                'layout':go.Layout(title='My Scatter Plot')})
                        ])



if __name__=='__main__':
    app.run_server(host = '127.0.0.1') 