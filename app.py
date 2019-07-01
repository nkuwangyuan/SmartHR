# -*- coding: utf-8 -*-
import dash
import dash_table as table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
import colorlover as cl
import plotly_express as px
import plotly.graph_objs as go
import base64

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import lime
import lime.lime_tabular


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config['suppress_callback_exceptions']=True

app.title = 'Smart HR'

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
Feature_Ranking = pd.read_csv('Feature_Ranking.csv').sort_values(by='Feature_Importance',ascending=True)


all_col = list(data)
remove = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
target = ['Attrition']
feature = list(set(all_col)-set(remove)-set(target))
cat_col = ['Attrition','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
num_col = list(set(all_col)-set(remove)-set(cat_col))






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
        dcc.Tab(label='Company Strategy', value='tab-1'),
        dcc.Tab(label='Features', value='tab-2'),
        dcc.Tab(label='Employee Analysis', value='tab-3'),
    ]),

    html.Div(id='tabs-content'),

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

color_RdYlGn = cl.scales['11']['div']['RdYlGn']
Feature_Ranking = dcc.Graph(figure = go.Figure(
    data=[go.Bar(
            x=Feature_Ranking['Feature_Importance'],
            y=Feature_Ranking['Feature'],
            orientation='h',
            #marker=dict(color=cl.scales['9']['seq']['Reds'],reversescale=True),
            marker=dict(color=np.array(color_RdYlGn)[np.arange(10,0,-1)]),
            )
        ],
    layout = go.Layout(
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
        width=700,
        height=500,
        #plot_bgcolor='#c7c7c7'
        )
    )
)

Tab1_Design = html.Div(children=[
        
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        
        html.Div(children=[
            html.Div([
                html.Label('Load HR Data Here'),
                dcc.Dropdown(id='load_data',
                    options=[{'label': 'Tech_Company_A', 'value': 'Tech_Company_A'}],
                    )
                ], className="four columns"
            ),
            
            html.H4(' --- Features attributes to Attrition --- ',
                className='eight columns')

            ], className="row"
        ),

        html.Div(children=[
            html.Div(id='Turnover_Rate',
                className="four columns"),           

            html.Div(id='Feature_Ranking',
                className="eight columns"),            
            
            ], className="row"
        )
    ]
) 

@app.callback(
    Output(component_id='Turnover_Rate', component_property='children'),
    [Input(component_id='load_data', component_property='value')]
)
def update_output_div(load_data):
    if load_data == 'Tech_Company_A':
        return html.Label('Turnover Rate is 17%.')
    else:
        return

@app.callback(
    Output(component_id='Feature_Ranking', component_property='children'),
    [Input(component_id='load_data', component_property='value')]
)
def update_output_div(load_data):
    if load_data == 'Tech_Company_A':
        return Feature_Ranking
    else:
        return 


Tab2_Design = html.Div(
    children=[
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
def update_output_div(feature_name):
    if feature_name == '[ Age ]':
        return Age_plot
    elif feature_name == '[ Year ]':
        return Year_plot
    elif feature_name == '[ Wage ]':
        return dcc.Graph(figure = px.violin(data, x='Attrition', y='MonthlyRate', color='Gender', box=True, points='all',))
    elif feature_name == '[ Overtime ]':
        return turnover_rate_plot('OverTime')
    elif feature_name == '[ Gender ]':
        return turnover_rate_plot('Gender')
    elif feature_name == '[ Business Travel ]':
        return turnover_rate_plot('BusinessTravel')
    elif feature_name == '[ Job Role ]':
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


df = data.copy()
People_Ranking = df.loc[[44,1255,131,610,295],['Department','JobRole','JobLevel','YearsAtCompany','MonthlyIncome']].head(5)
People_Ranking['Risk'] = ['84.6%','82.3%','73.2%','66.1%','62.0%']
People_Ranking['Cost'] = df['MonthlyIncome']*4

All_Employee   = People_Ranking.sort_values(by=['MonthlyIncome'], ascending=False)
People_Risk    = People_Ranking.sort_values(by=['Risk'], ascending=False)
People_Cost    = People_Ranking.sort_values(by=['Cost'], ascending=False)

colors_YlOrRd = cl.scales['9']['seq']['YlOrRd']

All_Employee =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        columnwidth = [1,2,2,1,1,1,1,1],
                        header = dict(
                            values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                                        '<b>Years at Company<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost ($)<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [All_Employee.index, All_Employee.iloc[:,0], All_Employee.iloc[:,1], All_Employee.iloc[:,2],
                                    All_Employee.iloc[:,3], All_Employee.iloc[:,4], All_Employee.iloc[:,5], All_Employee.iloc[:,6]],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow',
                                                'lightyellow','lightyellow','lightyellow','lightyellow']),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    #automargin=True,
                    autosize=False,
                    width=1200,
                    height=300,
                    margin=go.layout.Margin(t=30,b=30)
                )
            )
        )

People_Risk_table =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        columnwidth = [1,2,2,1,1,1,1,1],
                        header = dict(
                            values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                                        '<b>Years at Company<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost ($)<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [People_Risk.index, People_Risk.iloc[:,0],People_Risk.iloc[:,1],People_Risk.iloc[:,2],
                                    People_Risk.iloc[:,3],People_Risk.iloc[:,4],People_Risk.iloc[:,5],People_Risk.iloc[:,6]],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow','lightyellow',
                                                'lightyellow',np.array(colors_YlOrRd)[np.arange(5,0,-1)],'lightyellow']),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    autosize=False,
                    width=1200,
                    height=300,
                    margin=go.layout.Margin(t=30,b=30)
                )
            )
        )

People_Cost_table =  dcc.Graph(
            figure=go.Figure(
                data=[go.Table(
                        columnwidth = [1,2,2,1,1,1,1,1],
                        header = dict(
                            values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                                        '<b>Years at Company<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost ($)<b>'],
                            line = dict(color = 'silver'),
                            fill = dict(color = 'lightskyblue'),
                            align = 'center',
                            font = dict(color = 'white', size = 15),
                            height = 40
                        ),
                        cells = dict(
                            values = [People_Cost.index, People_Cost.iloc[:,0], People_Cost.iloc[:,1], People_Cost.iloc[:,2],
                                    People_Cost.iloc[:,3], People_Cost.iloc[:,4], People_Cost.iloc[:,5], People_Cost.iloc[:,6]],
                            line = dict(color = ['silver']),
                            fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow','lightyellow',
                                                'lightyellow','lightyellow',np.array(colors_YlOrRd)[np.arange(5,0,-1)]]),
                            align = 'center',
                            font = dict(color = 'black', size = 12),
                            height = 30
                        )
                    )
                ],
                layout = go.Layout(
                    autosize=False,
                    width=1200,
                    height=300,
                    margin=go.layout.Margin(t=30,b=30)
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


df = data.copy()
    
all_col = list(df)
remove = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
target = ['Attrition']
feature = list(set(all_col)-set(remove)-set(target))

for col in all_col:
    df[col] = LabelEncoder().fit(df[col]).transform(df[col])

X = df[feature]
y = df['Attrition'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.75, random_state=42)

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=500,learning_rate=1,algorithm='SAMME')
ada_fit = ada.fit(X_train, y_train)

def Exit_Analysis(Employee_ID):
       
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature,
                                                    class_names=['No','Yes'], discretize_continuous=False)

    predict_fn_ada = lambda x: ada.predict_proba(x).astype(float)
    exp = explainer.explain_instance(X.iloc[Employee_ID], predict_fn_ada, num_features=5)

    tmp_lime = pd.DataFrame(exp.as_list(),columns=['feature', 'weight'])
    tmp_lime['color']=(tmp_lime['weight']>0).astype('int')
    tmp_lime['weight%'] = (round(tmp_lime['weight']*10000,2)).astype(str)+'%'

    fig = px.bar(tmp_lime[::-1], y='feature', x='weight', color = 'color', orientation = 'h',
        color_discrete_sequence = ['green','red'], opacity =0.8,
        #color_discrete_sequence=px.colors.diverging.Tealrose,opacity =0.7,
        labels={'color':'Attrition'},
        #text='weight%',
        template='plotly_white+presentation+xgridoff',
        )

    fig = fig.update(layout=dict(
        title=dict(text='Top Exit Risk Attributes                          \'No\'       \'Yes\'       ',
                    font=dict(family='Arial', size=28, color='black')), 
        yaxis=dict(title=None,ticks='outside',showline=True,showgrid=False,mirror=True,linecolor='black'), 
        xaxis=dict(title=None,showticklabels=False,showline=True,mirror=True,linecolor='black'),
        autosize=False, width=1200, height=400, margin=go.layout.Margin(l=350,r=100,t=50) 
        ))

    return fig


Tab3_Design = html.Div(children=[

    html.Div([
        dcc.RadioItems(
                    id='All_Employee',
                    options=[{'label': i, 'value': i} for i in ['[ All Employee ]','[ Exit Risk ]', '[ Replacement Cost ]']],
                    value='[ All Employee ]',
                    labelStyle={'display': 'inline-block'}
        )],
        style={'width': '100%', 'display': 'inline-block'},),
            
    html.Div(id='Employee_Plot'),
            
    html.Label('Input Employee ID'),

    html.Div([
        dcc.Dropdown(id='Employee_ID', placeholder='Enter...',
        options=[{'label': i, 'value': i} for i in df.index]),
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': "middle"}),
    html.Div([
        html.Button(id='button', children='Submit', type='submit'),
        ], style={'display': 'inline-block', 'verticalAlign': "middle"}),


    
    html.Div(id='Exit_Analysis'), 
])


@app.callback(
    Output(component_id='Employee_Plot', component_property='children'),
    [Input(component_id='All_Employee', component_property='value')]
)
def update_output_div(input_value):
    if input_value == '[ Exit Risk ]':
        return People_Risk_table
    elif input_value == '[ Replacement Cost ]':
        return People_Cost_table
    else:
        return All_Employee


@app.callback(
    Output(component_id='Exit_Analysis', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
    [State(component_id='Employee_ID', component_property='value')] 
)
def update_output_div(n_clicks, Employee_ID):
    
    if Employee_ID in df.index:
        fig = Exit_Analysis(Employee_ID)
        return dcc.Graph(figure=fig)
    else:
        return


if __name__ == '__main__':
    app.run_server(debug=True)
    
