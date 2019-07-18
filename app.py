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


# dash page style

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True
server = app.server
app.title = 'SmartHR'


# load dataset
# datasets: 'Feature_Ranking.csv', 'Exit_Risk.csv', 'Employee_Attrition.csv'

data = pd.read_csv('Employee_Attrition.csv')


# add image

def add_image(image_name):
    image_filename = image_name
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                        style={'width': '100%'})
    return image


# dash layout

app.layout = html.Div([

    html.Div([add_image('SmartHR.png')], className="row"),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Workforce Analytics', value='tab-1'),
        dcc.Tab(label='Attrition Factors', value='tab-2'),
        dcc.Tab(label='Employee Analytics', value='tab-3'),
    ]),
    
    html.Div(id='top-bar', className='row', style={'backgroundColor': '#1f77b4', 'height': '3px'}),

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


# tab-1 page

Tab1_Design = html.Div(children=[

    html.Div(children=[
        html.Div([
            html.Label('Load HR Data Here'),
            dcc.Dropdown(id='load_data',
                options=[{'label': 'Tech_Company_X', 'value': 'Tech_Company_X'}],)
            ], className='four columns'),
        
        html.H6(' ---  Top Factors Attribute to Employee Attrition  --- ',
            style={'horizontalAlign': "middle"},
            className='eight columns'),

        ], className="row"
    ),

    html.Div(children=[
        html.Div(id='Turnover_Rate',
            className="four columns"),           

        html.Div(id='Feature_Ranking',
            className="eight columns"),            
            
        ], className="row"
    )
]) 

@app.callback(
    Output(component_id='Turnover_Rate', component_property='children'),
    [Input(component_id='load_data', component_property='value')]
)
def update_output_div(load_data):
    if load_data == 'Tech_Company_X':
        return html.H6('Turnover Rate is 17%.')
    else:
        return

@app.callback(
    Output(component_id='Feature_Ranking', component_property='children'),
    [Input(component_id='load_data', component_property='value')]
)
def update_output_div(load_data):
    if load_data == 'Tech_Company_X':
        return Feature_Ranking
    else:
        return 

# feature ranking 

Feature_Ranking = pd.read_csv('Feature_Ranking.csv').sort_values(by='Feature_Importance',ascending=True)
color_PuBu = cl.scales['9']['seq']['PuBu']

Feature_Ranking = dcc.Graph(figure = go.Figure(
    data=[go.Bar(
            x=Feature_Ranking['Feature_Importance'],
            y=Feature_Ranking['Feature'],
            orientation='h',
            #marker=dict(color=cl.scales['9']['seq']['Reds'],reversescale=True),
            marker=dict(color=np.array(color_PuBu)[[4,4,5,5,6,6,7,7,8,8]]),
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
            tickfont=dict(size=18),
            ),
        xaxis=dict(
            zeroline=False,
            showline=False, 
            showticklabels=False,
            showgrid=True,
            #domain=[0, 0.42],
            ),
        autosize=False, width=800, height=600, margin=go.layout.Margin(t=30)
        )
    ))


# tab-2 page

Tab2_Design = html.Div(children=[
    
    html.Div(children=[  
        html.H6('Turnover Rate by Factor'),
        dcc.RadioItems(id='feature_name',
            options=[{'label': i, 'value': i} for i in ['Working Over Time', 'Working Years At Company', 'Employee Current Age', 'Monthly Income', 'Business Travel', 'Job Role', 'clear']],
            value='clear',)
        ], className="three columns"),

    html.Div(id='feature_plot',
        className="nine columns"),            
            
    ], className="row")

@app.callback(
    Output(component_id='feature_plot', component_property='children'),
    [Input(component_id='feature_name', component_property='value')]
)
def update_output_div(feature_name):
    if feature_name == 'Employee Current Age':
        return Age_plot
    elif feature_name == 'Working Years At Company':
        return Year_plot
    elif feature_name == 'Monthly Income':
        return Salary_plot
    elif feature_name == 'Working Over Time':
        return turnover_rate_plot('OverTime')
    elif feature_name == 'Business Travel':
        return turnover_rate_plot('BusinessTravel')
    elif feature_name == 'Job Role':
        return turnover_rate_plot('JobRole')
    else:
        return

def Turnover_by_Feature_table(feature):
    
    df = data.sort_values(by=[feature, 'Attrition'])    
    tmp = pd.crosstab(df[feature], df['Attrition']).reset_index()
    tmp['sum']  = tmp.sum(axis=1)
    tmp['Turnover_Rate%'] = tmp['Yes']/(tmp['Yes']+tmp['No'])*100
    #locals()['Turnover_by_%s'%col] = tmp
    return tmp

Turnover_by_Age  = Turnover_by_Feature_table('Age')

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
            xaxis=dict(range= [20, 55]),
            yaxis=dict(range= [0, 80], title= 'Count'), 
            yaxis2=dict(range= [0, 70],
                        overlaying= 'y',
                        anchor= 'x',
                        side= 'right',
                        zeroline=False,
                        showgrid= False,
                        title= 'Turnover Rate (%)'),
            legend=dict(x=0.8,y=0.97,),
            width=1100, height=600,
            )
        )
    )

Turnover_by_Year = Turnover_by_Feature_table('YearsAtCompany')

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
            yaxis2=dict(range= [0, 50],
                        overlaying= 'y',
                        anchor= 'x',
                        side= 'right',
                        zeroline=False,
                        showgrid= False,
                        title= 'Turnover Rate (%)'),
            legend=dict(x=0.8,y=0.97,),
            width=1100, height=600,

            )
        )   
    )

Salary_plot = dcc.Graph(
    figure = px.violin(data, x='Attrition', y='MonthlyRate', color='Gender',
                        box=True, points='all',
                        template='plotly_white+presentation+xgridoff', width=1100)
    )

def turnover_rate_plot(feature):
    
    df = data.sort_values(by=[feature, 'Attrition'])    
    tmp = pd.crosstab(df[feature], df['Attrition']).reset_index()
    tmp['sum']  = tmp.sum(axis=1)
    tmp['Turnover Rate (%)'] = tmp['Yes']/(tmp['Yes']+tmp['No'])*100
 
    fig = px.bar(tmp, x=feature, y='Turnover Rate (%)', color=feature,
                #text='Turnover Rate (%)',
                title='Turnover Rate cross %s'%feature,
                template='plotly_white+presentation')
    fig = fig.update(layout=dict(
                showlegend = False,
                autosize=False, width=1100, height=600,
                margin=go.layout.Margin(l=200,r=200,t=100)
        )) 
    return dcc.Graph(figure=fig)


# tab-3 page

Tab3_Design = html.Div(children=[
    
    html.H6('Which Employees are at the Highest Exit Risk?'),
    
    html.Div([
        dcc.RadioItems(id='Employee_Table_Option',
            options=[{'label': i, 'value': i} for i in ['[ All Employee ]','[ Exit Risk ]', '[ Replacement Cost ]']],
            value='[ All Employee ]',
            labelStyle={'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'inline-block'},),
            
    html.Div(id='Employee_Table'),

    html.H6('Employee Exit Analysis'),

    html.Label('Input Employee ID'),

    html.Div([
        dcc.Dropdown(id='Employee_ID', placeholder='Enter...',
        options=[{'label': i, 'value': i} for i in data.index]),
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': "middle"}),
    
    html.Div([
        html.Button(id='button', children='Submit', type='submit'),
        ], style={'display': 'inline-block', 'verticalAlign': "middle"}),
  
    html.Div(id='Exit_Analysis'), 
])

@app.callback(
    Output(component_id='Employee_Table', component_property='children'),
    [Input(component_id='Employee_Table_Option', component_property='value')]
)
def update_output_div(input_value):
    if input_value == '[ Exit Risk ]':
        return Employee_Risk_table
    elif input_value == '[ Replacement Cost ]':
        return Employee_Cost_table
    else:
        return All_Employee_table

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

# employee table

Exit_Risk = pd.read_csv('Exit_Risk.csv')
All_Employee  = Exit_Risk.sort_values(by=['Employee ID'], ascending=True)
Employee_Risk = Exit_Risk.sort_values(by=['Risk'], ascending=False)
Employee_Cost = Exit_Risk[Exit_Risk['Exit Risk']=='HIGH'].sort_values(by=['Cost'], ascending=False)

colors_RdYlGn = cl.scales['5']['div']['RdYlGn']
colors_Reds = cl.scales['9']['seq']['Reds']
colors_order_1  =  [0]*len(Employee_Risk[Employee_Risk['Exit Risk']=='HIGH'])\
                 + [1]*len(Employee_Risk[Employee_Risk['Exit Risk']=='MEDIUM'])\
                 + [2]*len(Employee_Risk[Employee_Risk['Exit Risk']=='LOW'])\
                 + [3]*len(Employee_Risk[Employee_Risk['Exit Risk']=='NO'])
colors_order_2  = [n for n in range(7) for i in range(3)][::-1]

All_Employee_table  = dcc.Graph(
    figure=go.Figure(
        data=[go.Table(
                columnwidth = [2,3,3,2,2,2,2,2],
                header = dict(
                    values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                                '<b>Working Years<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost<b>'],
                    line = dict(color = 'silver'),
                    fill = dict(color = 'lightskyblue'),
                    align = 'center',
                    font = dict(color = 'white', size = 13),
                    height = 40),    
                cells = dict(
                    values = [All_Employee.iloc[:,0],All_Employee.iloc[:,1],All_Employee.iloc[:,2],All_Employee.iloc[:,3],
                            All_Employee.iloc[:,4],All_Employee.iloc[:,5],All_Employee.iloc[:,6],All_Employee.iloc[:,7]],
                    line = dict(color = ['silver']),
                    fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow',
                                        'lightyellow','lightyellow','lightyellow','lightyellow']),
                    align = 'center',
                    font = dict(color = 'black', size = 12),
                    height = 30)
            )],
        layout = go.Layout(
            autosize=False,
            width=1300,
            height=300,
            margin=go.layout.Margin(t=30,b=30,l=60)
            )
        )
    )   

Employee_Risk_table = dcc.Graph(
    figure=go.Figure(
        data=[go.Table(
                columnwidth = [2,3,3,2,2,2,2,2],
                header = dict(
                    values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                            '<b>Years at Company<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost<b>'],
                    line = dict(color = 'silver'),
                    fill = dict(color = 'lightskyblue'),
                    align = 'center',
                    font = dict(color = 'white', size = 13),
                    height = 40),
                cells = dict(
                    values = [Employee_Risk.iloc[:,0],Employee_Risk.iloc[:,1],Employee_Risk.iloc[:,2],Employee_Risk.iloc[:,3],
                              Employee_Risk.iloc[:,4],Employee_Risk.iloc[:,5],Employee_Risk.iloc[:,6],Employee_Risk.iloc[:,7]],
                    line = dict(color = ['silver']),
                    fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow','lightyellow',
                                        'lightyellow',np.array(colors_RdYlGn)[colors_order_1],'lightyellow']),
                    align = 'center',
                    font = dict(color = 'black', size = 12),
                    height = 30)
            )],
        layout = go.Layout(
            autosize=False,
            width=1300,
            height=300,
            margin=go.layout.Margin(t=30,b=30,l=60)
            )
        )
    )

Employee_Cost_table = dcc.Graph(
    figure=go.Figure(
        data=[go.Table(
                columnwidth = [2,3,3,2,2,2,2,2],
                header = dict(
                    values = ['<b>Employee ID</b>', '<b>Department</b>', '<b>Job Role</b>','<b>Job Level<b>',
                                '<b>Years at Company<b>','<b>Monthly Income<b>','<b>Exit Risk<b>','<b>Replacement Cost<b>'],
                    line = dict(color = 'silver'),
                    fill = dict(color = 'lightskyblue'),
                    align = 'center',
                    font = dict(color = 'white', size = 13),
                    height = 40),
                cells = dict(
                    values = [Employee_Cost.iloc[:,0],Employee_Cost.iloc[:,1],Employee_Cost.iloc[:,2],Employee_Cost.iloc[:,3],
                              Employee_Cost.iloc[:,4],Employee_Cost.iloc[:,5],Employee_Cost.iloc[:,6],Employee_Cost.iloc[:,7]],
                    line = dict(color = ['silver']),
                    fill = dict(color = ['whitesmoke','lightyellow','lightyellow','lightyellow','lightyellow',
                                        'lightyellow','lightyellow',np.array(colors_Reds)[colors_order_2]]),
                    align = 'center',
                    font = dict(color = 'black', size = 12),
                    height = 30)
            )],
        layout = go.Layout(
            autosize=False,
            width=1300,
            height=300,
            margin=go.layout.Margin(t=30,b=30,l=60)
            )
        )
    )

# exit analysis

df = data.copy()
    
all_col = list(df)
remove = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
target = ['Attrition']
feature = list(set(all_col)-set(remove)-set(target))
feature_update = feature.copy()
feature_update.remove('MonthlyIncome')
feature_update.remove('DailyRate')
feature_update.remove('HourlyRate')

for col in all_col:
    df[col] = LabelEncoder().fit(df[col]).transform(df[col])

X = df[feature_update]
y = df['Attrition'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.75, random_state=42)

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=250,learning_rate=1,algorithm='SAMME')
ada_fit = ada.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_update,
                                                    class_names=['No','Yes'], discretize_continuous=False)
predict_fn_ada = lambda x: ada.predict_proba(x).astype(float)

def Exit_Analysis(Employee_ID):

    exp = explainer.explain_instance(X.loc[Employee_ID], predict_fn_ada, num_features=5)

    tmp_lime = pd.DataFrame(exp.as_list(),columns=['feature', 'weight'])
    tmp_lime['weight%'] = (round(tmp_lime['weight']*10000,2)).astype(str)+'%'
    tmp_lime['Exit']=(tmp_lime['weight']>0).astype('str')
    tmp_lime = tmp_lime.replace(regex={'True': 'Yes', 'False': 'No'})

    fig = px.bar(tmp_lime[::-1], y='feature', x='weight', color = 'Exit', orientation = 'h',
        color_discrete_map = {'No':'green', 'Yes':'red'}, opacity =0.8,
        category_orders={'Exit':['No','Yes']},
        text='weight%',
        template='plotly_white+presentation+xgridoff',
        )

    fig = fig.update(layout=dict(
        title=dict(text='Top Exit Risk Attributes',font=dict(family='Arial', size=28, color='black')),
        legend=dict(orientation='h', x=0.2, y=1.15),
        yaxis=dict(title=None,ticks='outside',showline=True,showgrid=False,mirror=True,linecolor='black'), 
        xaxis=dict(title=None,showticklabels=False,showline=True,mirror=True,linecolor='black'),
        autosize=False, width=1300, height=500, margin=go.layout.Margin(l=350,r=200,t=100) 
        ))

    return fig

# dash end

if __name__ == '__main__':
    app.run_server(debug=True)
