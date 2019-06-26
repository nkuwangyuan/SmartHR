# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:48:05 2019

@author: xiong
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd



df = pd.read_csv('C:/Users/xiong/Desktop/insight_app/results1.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )
   
app = dash.Dash()
server=app.server

app.layout = html.Div([
    html.H1(
        children='Welcome to Meter Identification System',
        style={
            'textAlign': 'center', 'backgroundColor':'green'}
    ),

    html.Div([
        html.Label('Select Your Meter Size'),
        dcc.Dropdown(
            id='dropdown1',
            options=[{'label': i, 'value': i} for i in [ 1.   ,  3.   ,  0.625,  2.   ,  4.   ,  0.75 ,  1.5  ,  6.   ,
       10.   ,  8. ]]
        ),
        html.Label('Select Your Customer Type'),
        dcc.Dropdown(
            id='dropdown2',
            options=[{'label': i, 'value': i} for i in [ 1.,  3.,  2.,  5.,  4., 13., -1.,  8.,  9., 12.,  7.]]
        ),
        html.Label('Input Your Meter ID'),
        dcc.Input(id='box1',type='number')
    ],
    style={'width': '14%', 'display': 'inline-block'}),

    html.Div(id='tablecontainer', style={'backgroundColor':'white','fontWeight':'bold'})
])

@app.callback(
    dash.dependencies.Output('tablecontainer', 'children'),
    [dash.dependencies.Input('dropdown1', 'value'),
    dash.dependencies.Input('dropdown2', 'value'),
    dash.dependencies.Input('box1', 'value')])   
    
def update_table(dropdown1,dropdown2,box1):
    if box1 is not None:
        dff=df[df.Meter_id==box1]
        return generate_table(dff)
    elif box1 is None:
        if dropdown1 is None and dropdown2 is None:
            return generate_table(df)
        else:
            dff = df[(df.Meter_size==dropdown1)]
            dff = dff[(dff.Cust_type_code==dropdown2)] 
            return generate_table(dff)

  
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

   

if __name__ == '__main__':
    app.run_server(debug=True)
