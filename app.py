import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import numpy as np

from clt_mc import ExploreCLT

app = dash.Dash()
server = app.server


note_text = '''
*[Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
'''

source_text = '''
[Github repo](https://github.com/drlarue/clt-monte-carlo)
'''

app.layout = html.Div(children=[
    html.H1(children='Exploring CLT via Monte Carlo Simulation'),

    html.Div(children='''
        For a single sample (consisting of n independent observations), 
        the Central Limit Theorem (CLT) is typically invoked to argue that 
        the sample mean approximately follows a Normal distribution. 
        But exactly how good is this approximation?
    '''),

    html.Div(children=[
        html.Label('Parent population to draw samples from: '),
        dcc.Dropdown(id='input-distribution',
            options=[{'label': 'Exponential', 'value': 'exponential'},
                     {'label': 'Logistic', 'value': 'logistic'},
                     {'label': 'Normal', 'value': 'normal'},
                     {'label': 'Uniform', 'value': 'uniform'}],
            value='exponential'),

        html.Label(' Sample size: '),
        html.Br(),
        dcc.Input(id='input-sample_size', min=5, value=50, type='number'),
        html.Br(),

        html.Label(' Desired α (two-tailed test): '),
        html.Br(),
        dcc.Input(id='input-desired_alpha', value=0.05, type='number'),

        html.Button(id='submit-button', n_clicks=0, children='Submit'),
        ], style={'marginTop': 25, 'fontFamily': 'arial', 'color': '#6f6f6f'}),

    html.Div([dcc.Graph(id='plots')]),

    html.Div([dcc.Markdown(children=note_text)]),

    html.Div([dcc.Markdown(children=source_text)])
])

exponential = ExploreCLT(lambda n: np.random.exponential(size=n), 1)
logistic = ExploreCLT(lambda n: np.random.logistic(size=n), 0)
normal = ExploreCLT(lambda n: np.random.normal(size=n), 0)
uniform = ExploreCLT(lambda n: np.random.uniform(size=n), 0.5)

@app.callback(
    Output('plots', 'figure'),
    [Input('submit-button', 'n_clicks')],
    state=[State('input-distribution', 'value'),
           State('input-sample_size', 'value'),
           State('input-desired_alpha', 'value')])
def update_fig(n_click, dist, n, a):
    sample_size = int(n)
    desired_alpha = float(a)

    if dist == 'exponential':
        parent = exponential
    elif dist == 'logistic':
        parent = logistic
    elif dist == 'normal':
        parent = normal
    elif dist == 'uniform':
        parent = uniform

    sampling_distribution = parent.sampling_mean(sample_size)
    fig = parent.report(sampling_distribution, desired_alpha)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
