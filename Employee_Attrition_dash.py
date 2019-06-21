# -*- coding: utf-8 -*-
import dash
import dash_table as table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import colorlover as cl
import plotly_express as px
import plotly.graph_objs as go
import base64


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config['suppress_callback_exceptions']=True

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

all_col = list(data)
remove = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
target = ['Attrition']
feature = list(set(all_col)-set(remove)-set(target))
cat_col = ['Attrition','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
num_col = list(set(all_col)-set(remove)-set(cat_col))


feature_importances_list = [0.03489322, 0.04384273, 0.04995378, 0.02728444, 0.11044229,
       0.02294369, 0.09278575, 0.04318212, 0.01870959, 0.01987845,
       0.02085904, 0.03519061, 0.05598235, 0.00685271, 0.08265259,
       0.00833128, 0.06414993, 0.01767811, 0.00548018, 0.02769908,
       0.01030461, 0.04067725, 0.03109849, 0.07080172, 0.00470841,
       0.01627989, 0.00653033, 0.0158892 , 0.        , 0.01491816]
Feature_Importances = pd.DataFrame({'Feature': feature, 'Feature Importance': feature_importances_list}) \
                        .sort_values(by='Feature Importance',ascending=True).head(30)



markdown_text = '''
### Dash Title Here

Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''


app.layout = html.Div([

    html.H1(children='Smart HR'),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Tab 1 Name', value='tab-1'),
        dcc.Tab(label='Tab 2 Name', value='tab-2'),
        dcc.Tab(label='Tab 3 Name', value='tab-3'),
        dcc.Tab(label='Tab 4 Name', value='tab-4'),
        dcc.Tab(label='Tab 5 Name', value='tab-5'),
        dcc.Tab(label='Tab 6 Name', value='tab-6'),
    ]),

    html.Div(id='tabs-content'),

    html.H2(children='Footnote Here All pages')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])

def render_content(tab):
    if   tab == 'tab-1':
        return Tab1_Design
    elif tab == 'tab-2':
        return Tab2_Design
    elif tab == 'tab-3':
        return Tab3_Design
    elif tab == 'tab-4':
        return Tab4_Design
    elif tab == 'tab-5':
        return Tab5_Design
    elif tab == 'tab-6':
        return Tab6_Design

def Header():
    return html.Div([
        get_header(),
        html.Br([]),
        get_menu()
    ])

def get_header():
    header = html.Div([

        html.Div([
            html.H5(
                'Calibre Financial Index Fund Investor Shares')
        ], className="twelve columns padded")

    ], className="row gs-header gs-text-header")
    return header

def get_menu():
    menu = html.Div([
        dcc.Link('[   Age   ]', className="tab first"),
        dcc.Link('[   Price Performance   ]', className="tab"),
    ], className="row ")
    return menu

def generate_table(dataframe, max_rows=15):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def Turnover_by_Feature(col):
    
    df = data.sort_values(by=[col, 'Attrition'])    

    tmp = pd.crosstab(df[col], df['Attrition']).reset_index()
    tmp['sum']  = tmp.sum(axis=1)
    tmp['Turnover_Rate%'] = tmp['Yes']/(tmp['Yes']+tmp['No'])*100
    #locals()['Turnover_by_%s'%col] = tmp
    return tmp

Turnover_by_Age  = Turnover_by_Feature('Age')
Turnover_by_Year = Turnover_by_Feature('YearsAtCompany')

Age_plot =  dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=Turnover_by_Age['Age'],
                        y=Turnover_by_Age['Yes'],
                        name='Yes',
                        marker=dict(color='rgb(49,130,189)')
                    ),

                    go.Bar(
                        x=Turnover_by_Age['Age'],
                        y=Turnover_by_Age['No'],
                        name='No',
                        marker=dict(color='rgb(204,204,204)')
                    ),

                    go.Scatter(
                        x=Turnover_by_Age['Age'],
                        y=Turnover_by_Age['Turnover_Rate%'],
                        name='Turnover_Rate%',
                        yaxis = 'y2',
                        marker=dict(color='red')
                    )
                ],

                layout = go.Layout(
                    title = 'Turnover Rate cross Age',
                    xaxis=dict(range= [19, 60]),
                    yaxis=dict(range= [0, 70], title= 'Count'), 
                    yaxis2=dict(range= [0, 70],
                                overlaying= 'y',
                                anchor= 'x',
                                side= 'right',
                                zeroline=False,
                                showgrid= False,
                                title= 'Turnover Rate (%)'),
                    legend=dict(x=0.8,y=0.97,)
                )
            )
    )

Year_plot =  dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=Turnover_by_Year['YearsAtCompany'],
                        y=Turnover_by_Year['Yes'],
                        name='Yes',
                        marker=dict(color='rgb(49,130,189)')
                    ),

                    go.Bar(
                        x=Turnover_by_Year['YearsAtCompany'],
                        y=Turnover_by_Year['No'],
                        name='No',
                        marker=dict(color='rgb(204,204,204)')
                    ),

                    go.Scatter(
                        x=Turnover_by_Year['YearsAtCompany'],
                        y=Turnover_by_Year['Turnover_Rate%'],
                        name='Turnover_Rate%',
                        yaxis = 'y2',
                        marker=dict(color='red')
                    )
                ],

                layout = go.Layout(
                    title = 'Turnover Rate cross Year At Company',
                    xaxis=dict(range= [0, 20]),
                    yaxis=dict(range= [0, 200], title= 'Count'), 
                    yaxis2=dict(range= [0, 40],
                                overlaying= 'y',
                                anchor= 'x',
                                side= 'right',
                                zeroline=False,
                                showgrid= False,
                                title= 'Turnover Rate (%)'),
                    legend=dict(x=0.8,y=0.97,)
                )
            )
    )


Tab1_Design = html.Div(children=[
   
    html.H1(children='Big Title Here'),

    dcc.Markdown(children=markdown_text),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(id='graph1',
    figure=go.Figure(
        data=[
                go.Scatter(
                x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                y= [15.5,15.7,13.0,11.2,11.3,11.5,12.5,13.0,14.2,15.3,16.0],
                text='%',
                name='Technology',
                marker=dict(color='rgb(49,130,189)')
                ),

                go.Scatter(
                x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                y= [13.7,12.6,10.8,9.8,9.6,10.4,11.0,11.25,11.6,12.2,12.9],
                text='%',
                name='All_Industries',
                )
            ],

        layout=go.Layout(
                title='Turnover Rate Over Last 10 Years',
                xaxis=dict(
                    title='Year',
                    #zeroline=True
                    ),
                yaxis=dict(
                    range=[0, 20],
                    zeroline=True,
                    showgrid= True,
                    title= 'Turnover Rate (%)'
                    ),
                showlegend=True,
                legend=go.layout.Legend(x=0.8,y=0.1),
                #margin=go.layout.Margin(l=40, r=0, t=40, b=30)
           ),
    )
    ),

    #dcc.Graph(px.violin(data, x='Attrition', y='MonthlyRate', color='Gender', box=True, points='all',)),

    html.H4(children='New Table Here'),

    generate_table(data)
])

Tab2_Design = html.Div(children=[
    
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF']
    ),

    html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=9,
                     marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
                    value=5,
                 ),
])


Tab3_Design = html.Div(
    children=[
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        html.Div(
            className='container', children=[
                
                html.Div(id='left-side-column', className='seven columns padded', 
                    style={
                        #'display': 'flex',
                        #'flexDirection': 'column',
                        #'flex': 1,
                        #'height': 'calc(100vh - 5px)',
                        #'backgroundColor': '#F2F2F2',
                        #'overflow-y': 'scroll',
                        #'marginLeft': '0px',
                        'justifyContent': 'flex-start',
                        'alignItems': 'left'
                    },
                    children=[
                    html.Label('Text Input'),
                    dcc.Input(value='MTL', type='text'),
                    dcc.Graph(id='graph1',
                        figure=go.Figure(
                            data=[
                                    go.Scatter(
                                    x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                                    y= [15.5,15.7,13.0,11.2,11.3,11.5,12.5,13.0,14.2,15.3,16.0],
                                    text='%',
                                    name='Technology',
                                    marker=dict(color='rgb(49,130,189)')
                                    ),

                                    go.Scatter(
                                    x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                                    y= [13.7,12.6,10.8,9.8,9.6,10.4,11.0,11.25,11.6,12.2,12.9],
                                    text='%',
                                    name='All_Industries',
                                    )
                                ],

                            layout=go.Layout(
                                    title='Turnover Rate Over Last 10 Years',
                                    xaxis=dict(
                                        title='Year',
                                        #zeroline=True
                                        ),
                                    yaxis=dict(
                                        range=[0, 20],
                                        zeroline=True,
                                        showgrid= True,
                                        title= 'Turnover Rate (%)'
                                        ),
                                    showlegend=True,
                                    legend=go.layout.Legend(x=0.6,y=0.1),
                                    #margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                            ),
                        )
                    ),
                    ]
                    ),

    html.Div(id='right-side-column',
            className='five columns',
            # style={
            #     'height': 'calc(100vh - 5px)',
            #     'overflow-y': 'scroll',
            #     'marginLeft': '1%',
            #     'display': 'flex',
            #     'backgroundColor': '#F9F9F9',
            #     'flexDirection': 'column'
            # },
            children=[
                 dcc.Graph(id='graph2',
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                x=Feature_Importances['Feature Importance'],
                                y=Feature_Importances['Feature'],
                                orientation='h',
                            )
                        ],

                        layout = go.Layout(
                            title='Features Importance Ranking',
                    
                            yaxis=go.layout.YAxis(
                                #title='Feature',
                                tickmode='array',
                                automargin=True,
                                showgrid=False,
                                showline=True,
                                showticklabels=True,
                                titlefont=dict(size=15),
                            ),

                            xaxis=dict(
                                zeroline=False,
                                showline=False,
                                showticklabels=False,
                                showgrid=True,
                                #domain=[0, 0.42],
                            ),
                
                            autosize=False,
                            width=900,
                            height=900,
                            #plot_bgcolor='#c7c7c7'
                        )
                    )
                 )
            ]
    )
],
#    style={'columnCount': 2}
)])

Tab4_Design = html.Div(
    children=[
        
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        
        html.Div(
            children=[

                html.Div(children=[
                            html.Label('Text Input'),
                            dcc.Input(value='MTL', type='text'),
                            
                            dcc.Graph(id='graph1',
                                figure=go.Figure(
                                    data=[
                                            go.Scatter(
                                            x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                                            y= [15.5,15.7,13.0,11.2,11.3,11.5,12.5,13.0,14.2,15.3,16.0],
                                            text='%',
                                            name='Technology',
                                            marker=dict(color='rgb(49,130,189)')
                                            ),

                                            go.Scatter(
                                            x= [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],
                                            y= [13.7,12.6,10.8,9.8,9.6,10.4,11.0,11.25,11.6,12.2,12.9],
                                            text='%',
                                            name='All_Industries',
                                            )
                                        ],

                                    layout=go.Layout(
                                            title='Turnover Rate Over Last 10 Years',
                                            xaxis=dict(
                                                title='Year',
                                                #zeroline=True
                                                ),
                                            yaxis=dict(
                                                range=[0, 20],
                                                zeroline=True,
                                                showgrid= True,
                                                title= 'Turnover Rate (%)'
                                                ),
                                            showlegend=True,
                                            legend=go.layout.Legend(x=0.6,y=0.1),
                                            #margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                                    ),
                                )
                            ),
                        ],
                ),      

                html.Div(children=[
                            dcc.Graph(id='graph2',
                                figure=go.Figure(
                                    data=[
                                        go.Bar(
                                            x=Feature_Importances['Feature Importance'],
                                            y=Feature_Importances['Feature'],
                                            orientation='h',
                                        )
                                    ],

                                    layout = go.Layout(
                                        title='Features Importance Ranking',
                                
                                        yaxis=go.layout.YAxis(
                                            #title='Feature',
                                            tickmode='array',
                                            automargin=True,
                                            showgrid=False,
                                            showline=True,
                                            showticklabels=True,
                                            titlefont=dict(size=15),
                                        ),

                                        xaxis=dict(
                                            zeroline=False,
                                            showline=False,
                                            showticklabels=False,
                                            showgrid=True,
                                            #domain=[0, 0.42],
                                        ),
                            
                                        autosize=False,
                                        width=540,
                                        height=540,
                                        #plot_bgcolor='#c7c7c7'
                                    )
                                )
                            )
                        ]
                ),
                ],

                style={'columnCount': 2}
        )
    ]
)    


Tab5_Design = html.Div(
    children=[
        Header(),
        html.Div(
            [
            dcc.RadioItems(
                id='feature_name',
                options=[{'label': i, 'value': i} for i in ['[ Age ]', '[ Year ]', '[ Wage ]', '[ Overtime ]', '[ Gender ]','[ Business Travel ]','[ Job Role ]','clear']],
                value='clear',
                labelStyle={'display': 'inline-block'}
            )
            ],
            style={'width': '90%', 'display': 'inline-block'},
        ),
        html.Div(id='feature_plot')
    ])

@app.callback(
    Output(component_id='feature_plot', component_property='children'),
    [Input(component_id='feature_name', component_property='value')]
)
def update_output_div(input_value):
    if input_value == '[ Age ]':
        return Age_plot
    elif input_value == '[ Year ]':
        return Year_plot
    elif input_value == '[ Wage ]':
        return dcc.Graph(figure = px.violin(data, x='Attrition', y='MonthlyRate', color='Gender', box=True, points='all',))
    elif input_value == '[ Overtime ]':
        return turnover_rate_plot('OverTime')
    elif input_value == '[ Gender ]':
        return turnover_rate_plot('Gender')
    elif input_value == '[ Business Travel ]':
        return turnover_rate_plot('BusinessTravel')
    elif input_value == '[ Job Role ]':
        return turnover_rate_plot('JobRole')
    else:
        return 'Please select a feature to explore'

def turnover_rate_plot(feature):
    df = data.sort_values(by=[feature, 'Attrition'])    
    tmp = pd.crosstab(df[feature], df['Attrition']).reset_index()
    tmp['sum']  = tmp.sum(axis=1)
    tmp['Turnover Rate (%)'] = tmp['Yes']/(tmp['Yes']+tmp['No'])*100
    #locals()['Turnover_by_%s'%feature] = tmp
    #tmp = tmp.sort_values(by='Turnover Rate (%)')
    fig = px.bar(tmp, x=feature, y='Turnover Rate (%)', color=feature,
                #category_orders='Turnover Rate (%)', text='Turnover Rate (%)',
                #title='Turnover Rate cross %s'%feature,
                template='plotly_dark+presentation', width=1300)
    return dcc.Graph(figure=fig)


Year = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
All_Industries = [13.7,12.6,10.8,9.8,9.6,10.4,11.0,11.25,11.6,12.2,12.9]
Technology =[15.5,15.7,13.0,11.2,11.3,11.5,12.5,13.0,14.2,15.3,16.0]

People_Ranking = pd.DataFrame(list(zip(Year, Year, All_Industries, Technology)), columns =['ID', 'Name','Risk', 'Cost'])
People_Risk    = People_Ranking.sort_values(by=['Risk'], ascending=False)
People_Cost    = People_Ranking.sort_values(by=['Cost'], ascending=False)
People_All     = People_Ranking.sort_values(by=['ID'], ascending=False)

colors = cl.scales['9']['seq']['YlOrRd']

All_Employee =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        #columnwidth = [1,2,3,1],
                        header = dict(
                            values = ['<b>ID</b>', '<b>Name</b>', '<b>Risk</b>','<b>Cost<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [People_All['ID'], People_All['Name'],People_All['Risk'], People_All['Cost']],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow','lemonchiffon','lightgoldenrodyellow']),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    autosize=False,
                    width=500,
                    height=600
                )
            )
        )

People_Risk_table =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        #columnwidth = [1,2,3,1],
                        header = dict(
                            values = ['<b>ID</b>', '<b>Name</b>', '<b>Risk</b>','<b>Cost<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [People_Risk['ID'], People_Risk['Name'],People_Risk['Risk'], People_Risk['Cost']],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow',np.array(colors)[np.arange(5,0,-1)],'lemonchiffon']),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    autosize=False,
                    width=500,
                    height=600
                )
            )
        )

People_Cost_table =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        #columnwidth = [1,2,3,1],
                        header = dict(
                            values = ['<b>ID</b>', '<b>Name</b>', '<b>Risk</b>','<b>Cost<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [People_Cost['ID'], People_Cost['Name'],People_Cost['Risk'], People_Cost['Cost']],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow','lemonchiffon', np.array(colors)[np.arange(5,0,-1)]]),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    autosize=False,
                    width=500,
                    height=600
                )
            )
        )

image_filename = 'Gru.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
add_image = html.Div([

        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), height='200px')
            ], className="ten columns padded"
        )
    ], className="row gs-header")



Tab6_Design = html.Div(
    children=[
        html.H1(children='Big Title Here'),
        
        html.Div(
            [
            dcc.RadioItems(
                id='All_Employee',
                options=[{'label': i, 'value': i} for i in ['[ All Employee ]','[ Exit Risk ]', '[ Exit Cost ]']],
                value='[ All Employee ]',
                labelStyle={'display': 'inline-block'}
            )
            ],
            style={'width': '50%', 'display': 'inline-block'},
        ),
        
        html.Div(id='Employee_Plot'),

        html.Div(children=[
            html.H1(children='Big Title Here'),
            dcc.Input(id='Employee_ID', value=0, type='number'),

            html.Div([
                html.H1(children='Title Here'), 
                add_image, 
                html.Div(id='Employee_Analysis')])
        ])
    ]
)    

@app.callback(
    Output(component_id='Employee_Plot', component_property='children'),
    [Input(component_id='All_Employee', component_property='value')]
)
def update_output_div(input_value):
    if input_value == '[ Exit Risk ]':
        return People_Risk_table
    elif input_value == '[ Exit Cost ]':
        return People_Cost_table
    else:
        return All_Employee
   

@app.callback(
    Output(component_id='Employee_Analysis', component_property='children'),
    [Input(component_id='Employee_ID', component_property='value')]
)
def update_output_div(Employee_ID):
    html.H1(children='Title Here')
    Employee = data.iloc[Employee_ID,[0,11,31,4,15,14,18]]
    trace=[go.Table(
        header=dict(line = dict(color='white')),
        cells = dict(
            values = [Employee.index, Employee.values],
            line = dict(color = ['white']),
            fill = dict(color = ['whitesmoke','lightgoldenrodyellow']),
            align = 'left',
            font = dict(color = 'black', size = 12),
            height = 30
            )
        )]
    layout = go.Layout(
            autosize=False,
            width=500,
            height=600
        )
    return dcc.Graph(figure=go.Figure(trace, layout))



if __name__ == '__main__':
    app.run_server(debug=True)
    