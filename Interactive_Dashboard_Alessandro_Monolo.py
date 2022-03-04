''' -                           Project: Tweets analysis of the principal Italian populist leaders:
                  Giuseppe Grillo (Movimento 5 stelle), Giorgia Meloni (Fratelli d'Itlia) and Matteo Salvini (Lega).
                  
    -                                      Student: Alessandro Monolo | 1790210;
    -                                              Lecturer: Erik Hekman;
    -                     Fundamentals of Data Science - Master Data-Driven Design - Hogeschool Utrecht;
    -                                              August 2021 - Block E.                                                   '''

### Interactive dashboard of the most mentioned words and hashtags inside the last tweets of the
### Italian populist leaders: @beppe_grillo, @GiorgiaMeloni and @matteosalvinimi;
### Sentiment Analysis with emotion and sentiment classification over the last tweets using FEEL-IT classifier (NLP);
### Text-Length and emotion classes distribution among the timeline.

### Import libraries:
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
import csv
import plotly
import plotly.express as px
import plotly.io as pio
import flask
import glob
import os
import random

### Set plotly with dark theme style:
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"

### Import csv files using pandas:
# beppe_grillo
df_hashtags_grillo = pd.read_csv('beppe_grillo_most_used_hashtags.csv')
df_words_grillo = pd.read_csv('beppe_grillo_most_used_words.csv')
# GiorgiaMeloni
df_hashtags_meloni = pd.read_csv('GiorgiaMeloni_most_used_hashtags.csv')
df_words_meloni = pd.read_csv('GiorgiaMeloni_most_used_words.csv')
# matteosalvinimi
df_hashtags_salvini = pd.read_csv('matteosalvinimi_most_used_hashtags.csv')
df_words_salvini = pd.read_csv('matteosalvinimi_most_used_words.csv')
# Tweets - beppe_grillo
df_grillo = pd.read_csv('beppe_grillo_tweets_2018_2021.csv')
# Tweets - GiorgiaMeloni
df_meloni = pd.read_csv('GiorgiaMeloni_tweets_2019_2021.csv')
# Tweets - matteosalvinimi
df_salvini = pd.read_csv('matteosalvinimi_tweets_2020_2021.csv')

### Slicing dataframes to get only the top 25 elements of each df:
# beppe_grillo
top_hashtags_grillo = df_hashtags_grillo[:25]
top_words_grillo = df_words_grillo[:25]
# GiorgiaMeloni
top_hashtags_meloni = df_hashtags_meloni[:25]
top_words_meloni = df_words_meloni[:25]
# matteosalvinimi
top_hashtags_salvini = df_hashtags_salvini[:25]
top_words_salvini = df_words_salvini[:25]

### Favorite and retweet mean count per each tweet emotion classes in df:
# @beppe_grillo
df_emotion_grillo = df_grillo.groupby(['tweet_emotion'], as_index = False)[['retweet', 'favorite']].mean()
df_emotion_grillo.sort_values(by=['retweet'], inplace = True, ascending = False)
# @GiorgiaMeloni
df_emotion_meloni = df_meloni.groupby(['tweet_emotion'], as_index = False)[['retweet', 'favorite']].mean()
df_emotion_meloni.sort_values(by=['retweet'], inplace = True, ascending = False)
# @matteosalvinimi
df_emotion_salvini = df_salvini.groupby(['tweet_emotion'], as_index = False)[['retweet', 'favorite']].mean()
df_emotion_salvini.sort_values(by=['retweet'], inplace = True, ascending = False)

### Set RangeSlider years dict:
# mark values - beppe_grillo
mark_values_grillo = {2018:'2018', 2019:'2019',2020:'2020',2021:'2021'}
# mark values - GiorgiaMeloni
mark_values_meloni = {2019:'2019',2020:'2020',2021:'2021'}
# mark values - matteosalvinimi
mark_values_salvini = {2020 : '2020' , 2021 : '2021'}

### Set the figures:
# 1.1 - Top hashtags - beppe_grillo
fig_top_hashtags_grillo = go.Figure([go.Bar(x = top_hashtags_grillo['hashtag'],
                                            y = top_hashtags_grillo['count'],
                                            name= 'beppe_grillo',
                                            marker= {'color':"rgb(255,196,0)"})])
# 1.2 - Top hashtags - GiorgiaMeloni
fig_top_hashtags_meloni = go.Figure([go.Bar(x = top_hashtags_meloni['hashtag'],
                                            y = top_hashtags_meloni['count'],
                                            name= 'GiorgiaMeloni',
                                            marker= {'color':"rgb(255,196,0)"})])
# 1.3 - Top hashtags - matteosalvinimi
fig_top_hashtags_salvini = go.Figure([go.Bar(x = top_hashtags_salvini['hashtag'],
                                             y = top_hashtags_salvini['count'],
                                             name= 'matteosalvinimi',
                                             marker= {'color':"rgb(255,196,0)"})])
# 2.1 - Top words - beppe_grillo
fig_top_words_grillo = go.Figure([go.Bar(x = top_words_grillo['word'],
                                         y = top_words_grillo['count'],
                                         name= 'beppe_grillo',
                                         marker= {'color':"rgb(255,196,0)"})])
# 2.2 - Top words - GiorgiaMeloni
fig_top_words_meloni = go.Figure([go.Bar(x = top_words_meloni['word'],
                                         y = top_words_meloni['count'],
                                         name= 'GiorgiaMeloni',
                                         marker= {'color':"rgb(255,196,0)"})])
# 2.3 - Top words - matteosalvinimi
fig_top_words_salvini = go.Figure([go.Bar(x = top_words_salvini['word'],
                                          y = top_words_salvini['count'],
                                          name= 'matteosalvinimi',
                                          marker= {'color':"rgb(255,196,0)"})])
# 3.1 - Sentiment analysis - beppe_grillo
fig_SA_grillo = px.sunburst(df_grillo,
                            path=['twitter_user', 'tweet_sentiment', 'tweet_emotion'],
                            color='tweet_lenght',
                            color_continuous_scale='rdbu')
# 3.2 - Sentiment analysis - GiorgiaMeloni
fig_SA_meloni = px.sunburst(df_meloni,
                            path=['twitter_user', 'tweet_sentiment', 'tweet_emotion'],
                            color='tweet_lenght',
                            color_continuous_scale='rdbu')
# 3.3 - Sentiment analysis - matteosalvinimi
fig_SA_salvini = px.sunburst(df_salvini,
                            path=['twitter_user', 'tweet_sentiment', 'tweet_emotion'],
                            color='tweet_lenght',
                            color_continuous_scale='rdbu')
# 4.1 - bar chart - @beppe_grillo
bar_emotion_grillo = go.Figure()
bar_emotion_grillo.add_trace(go.Bar(x = df_emotion_grillo['tweet_emotion'], 
                     y = df_emotion_grillo['retweet'],
                     width = 0.35, name = 'retweet'))
bar_emotion_grillo.add_trace(go.Bar(x = df_emotion_grillo['tweet_emotion'],
                     y = df_emotion_grillo['favorite'],
                     width = 0.35, name = 'favorite'))
# 4.2 - bar chart - @GiorgiaMeloni
bar_emotion_meloni = go.Figure()
bar_emotion_meloni.add_trace(go.Bar(x = df_emotion_meloni['tweet_emotion'], 
                     y = df_emotion_meloni['retweet'],
                     width = 0.35, name = 'retweet'))
bar_emotion_meloni.add_trace(go.Bar(x = df_emotion_meloni['tweet_emotion'],
                     y = df_emotion_meloni['favorite'],
                     width = 0.35, name = 'favorite'))
# 4.3 - bar chart - @matteosalvinimi
bar_emotion_salvini = go.Figure()
bar_emotion_salvini.add_trace(go.Bar(x = df_emotion_salvini['tweet_emotion'], 
                     y = df_emotion_salvini['retweet'],
                     width = 0.35, name = 'retweet'))
bar_emotion_salvini.add_trace(go.Bar(x = df_emotion_salvini['tweet_emotion'],
                     y = df_emotion_salvini['favorite'],
                     width = 0.35, name = 'favorite'))


### Set details of each figures:
# Hashtags beppe_grillo
def HashtagsGrillo():
    fig = go.Figure(data = fig_top_hashtags_grillo)
    fig.update_layout(title_text="@beppe_grillo, 3296 different words and 799 hashtags in total", 
    title_x=0.5, title_font_color="rgb(171,199,255)", 
    title_font_size=13, 
    height = 550, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Hashtags GiorgiaMeloni
def HashtagsMeloni():
    fig = go.Figure(data = fig_top_hashtags_meloni)
    fig.update_layout(title_text="@GiorgiaMeloni, 2781 different words and 651 hashtags in total",
    title_x=0.5, title_font_color="rgb(171,199,255)",
    title_font_size=13, 
    height = 550, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Hashtags matteosalvini
def HashtagsSalvini():
    fig = go.Figure(data = fig_top_hashtags_salvini)
    fig.update_layout(title_text="@matteosalvinimi, 2936 different words and 558 hashtags in total",
    title_font_color="rgb(171,199,255)",
    title_x=0.5, title_font_size=13,
    height = 550, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Words beppe_grillo
def WordsGrillo():
    fig = go.Figure(data = fig_top_words_grillo)
    fig.update_layout(title_text="@beppe_grillo, 3215 tweets from May 18 to August 21",
    title_font_color="rgb(171,199,255)",
    title_x=0.5, title_font_size=13,
    height = 450, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Words GiorgiaMeloni
def WordsMeloni():
    fig = go.Figure(data = fig_top_words_meloni)
    fig.update_layout(title_text="@GiorgiaMeloni, 2781 tweets from December 19 August 21",
    title_x=0.5, title_font_color="rgb(171,199,255)",
    title_font_size=13, height = 450, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Words matteosalvini
def WordsSalvini():
    fig = go.Figure(data = fig_top_words_salvini)
    fig.update_layout(title_text="@matteosalvinimi, 3178 tweets from November 20 to August 21",
    title_font_color="rgb(171,199,255)",
    title_x=0.5, title_font_size=13,
    height = 450, width = 630,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)')
    return fig
# Sentiment Analysis - beppe_grillo
def SA_Grillo():
    fig = go.Figure(data = fig_SA_grillo)
    fig.update_layout(title_text="@beppe_grillo, Sentiment Analysis over his last 3214 tweets", 
    title_x=0.5, title_font_color="rgb(171,199,255)", 
    title_font_size=13,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)', 
    height = 550, width = 550)
    return fig
# Sentiment Analysis - GiorgiaMeloni
def SA_Meloni():
    fig = go.Figure(data = fig_SA_meloni)
    fig.update_layout(title_text="@GiorgiaMeloni, Sentiment Analysis over her last 2781 tweets", 
    title_x=0.5, title_font_color="rgb(171,199,255)", 
    title_font_size=13,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)', 
    height = 550, width = 550)
    return fig
# Sentiment Analysis - matteosalvinimi
def SA_Salvini():
    fig = go.Figure(data = fig_SA_salvini)
    fig.update_layout(title_text="@matteosalvinimi, Sentiment Analysis over his last 3178 tweets", 
    title_x=0.5, title_font_color="rgb(171,199,255)", 
    title_font_size=13, 
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(234,283,183,0.1)',
    font_color ='rgba(171,199,255,1)',
    height = 550, width = 550)
    return fig
# bar chart retweet and favorite - @beppe_grillo
def BarChartGrillo():
    bar_emotion_grillo.update_layout(title = "@beppe_grillo", barmode = 'group',title_font_size = 13,width = 600,height = 500, title_x=0.5,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(234,283,183,0.1)',
                                     font_color ='rgba(171,199,255,1)',
                                     legend = go.layout.Legend(x=1, y=1, traceorder = "normal", font=dict(family = "Verdana", size = 10, color = "rgb(171,199,255)")))
    bar_emotion_grillo.update_xaxes(title_text = 'Tweet emotion classes')
    bar_emotion_grillo.update_yaxes(title_text = "Retweet & favorite mean count")
    return bar_emotion_grillo
# bar chart retweet and favorite - @GiorgiaMeloni
def BarChartMeloni():
    bar_emotion_meloni.update_layout(title = "@GiorgiaMeloni", barmode = 'group',title_font_size = 13,width = 600,height = 500, title_x=0.5,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(234,283,183,0.1)',
                                     font_color ='rgba(171,199,255,1)',
                                     legend = go.layout.Legend(x=1, y=1, traceorder = "normal", font=dict(family = "Verdana", size = 10, color = "rgb(171,199,255)")))
    bar_emotion_meloni.update_xaxes(title_text = 'Tweet emotion classes')
    bar_emotion_meloni.update_yaxes(title_text = "Retweet & favorite mean count")
    return bar_emotion_meloni
# bar chart retweet and favorite - @matteosalvinimi
def BarChartSalvini():
    bar_emotion_salvini.update_layout(title = "@matteosalvinimi", barmode = 'group',title_font_size = 13,width = 600,height = 500, title_x=0.5,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(234,283,183,0.1)',
                                     font_color ='rgba(171,199,255,1)',
                                     legend = go.layout.Legend(x=1, y=1, traceorder = "normal", font=dict(family = "Verdana", size = 10, color = "rgb(171,199,255)")))
    bar_emotion_salvini.update_xaxes(title_text = 'Tweet emotion classes')
    bar_emotion_salvini.update_yaxes(title_text = "Retweet & favorite mean count")
    return bar_emotion_salvini


### Set the Headers of the Dash App:
Headers = dbc.Jumbotron([dbc.Container([html.H1("Tweets analysis of the principal Italian populist leaders", style={'textAlign': 'center', 'color' : 'rgb(0,172,238)'}),
                                      html.Div(style={'padding': 25}),
                                      html.H4("Interactive Dashboard:", style={'textAlign': 'center', 'color' : 'rgb(171,199,255)'}),
                                      html.Div(style={'padding': 10}),
                                      html.H5("1 - The 25 most mentioned hashtags and words from @beppe_grillo, @GiorgiaMeloni and @matteosalvinimi tweets;", style={'textAlign': 'center', 'color' : 'rgb(171,199,255)'}),
                                      html.H5("2 - Emotion and Sentiment Classification over the tweets with FEEL-IT (NLP) and mean count of retweet and favorite per each emotion class.", style={'textAlign': 'center', 'color' : 'rgb(171,199,255)'}),
                                      html.H5("3 - Emotional classes and text length distribution of tweets along timeline.", style={'textAlign': 'center', 'color' : 'rgb(171,199,255)'})
                       ],fluid=True,)],fluid=True,)

### Set the first row of th Dash App:
First_Row = html.Div([

    dbc.Row([
        dbc.Col(),
        dbc.Row([dbc.Col(html.Div([
            html.H5("25 Most mentioned hashtags:", style={'textAlign': 'center','color': 'rgb(0,172,238)'}),
            dcc.Graph(id="bar_chart_hashtags",
                      figure = HashtagsGrillo())]))]),
        dcc.Dropdown(
            id="selection_hashtags",
            style={'height': '20px', 'width': '100px', 'verticalAlign':'middle'},
            options=[{"label": "@beppe_grillo", "value": "beppe_grillo"},
                     {"label": "@GiorgiaMeloni", "value": "GiorgiaMeloni"},
                     {"label": "@matteosalvinimi", "value": "matteosalvinimi"}
    ]),

        dbc.Col(),
        dbc.Row([dbc.Col(html.Div([
            html.H5("25 Most mentioned words:", style={'textAlign': 'center','color': 'rgb(0,172,238)'}),
            dcc.Graph(id="bar_chart_words",
                      figure = WordsGrillo())]))]),
        dcc.Dropdown(
            id="selection_words",
            style={'height': '20px', 'width': '100px', 'verticalAlign':'middle'},
            options=[{"label": "@beppe_grillo", "value": "beppe_grillo"},
                     {"label": "@GiorgiaMeloni", "value": "GiorgiaMeloni"},
                     {"label": "@matteosalvinimi", "value": "matteosalvinimi"}])
])])

### Set the second row of th Dash App:
Second_Row = html.Div([

    dbc.Row([
        dbc.Col(),
        dbc.Row([dbc.Col(html.Div([
            html.H5("Sentiment Analysis:", style={'textAlign': 'center','color': 'rgb(0,172,238)'}),
            dcc.Graph(id="sunburst_chart_SA",
                      figure = SA_Grillo(), style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'})]))]),
        dcc.Dropdown(
            id="selection_SA",
            style={'height': '20px', 'width': '100px', 'verticalAlign':'middle'},
            options=[{"label": "@beppe_grillo", "value": "beppe_grillo"}, {"label": "@GiorgiaMeloni", "value": "GiorgiaMeloni"}, {"label": "@matteosalvinimi", "value": "matteosalvinimi"}] 
                     ),

        dbc.Col(),
        dbc.Row([
                dbc.Col(html.Div([
                                  html.H5("Favorite & Retweet mean count per tweet emotions:", style={'textAlign': 'center','color': 'rgb(0,172,238)'}),
                                  dcc.Graph(id="bar_chart_emotion",
                                  figure = BarChartGrillo(), style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'})]))]),

                dcc.Dropdown(id="selection_barchart",
                            style={'height': '20px', 'width': '100px', 'verticalAlign':'middle'},
                            options=[{"label": "@beppe_grillo", "value": "beppe_grillo"}, {"label": "@GiorgiaMeloni", "value": "GiorgiaMeloni"}, {"label": "@matteosalvinimi", "value": "matteosalvinimi"}])
                            ])
                            ])


### Set the bubble_grillo row
bubble_grillo = html.Div([
        html.Div([
            html.H2(children= "Emotion classification and text length distribution of tweets along timeline:",
            style={"text-align": "center", "font-size":"150%", "color":"rgb(0,172,238)"})
        ]),

        html.Div([
            dcc.Graph(id='bubble_grillo',style={'display': 'inline-block', 'vertical-align': 'center'})
        ],style=dict(display='flex')),

        html.Div([
            dcc.RangeSlider(id='the_year_grillo',
                min=2018,
                max=2021,
                value=[2018,2021],
                marks=mark_values_grillo,
                step=None)
        ],style={"width": "100%", "position":"middle",
                 "left":"5%"})
])

### Set the bubble_meloni row
bubble_meloni = html.Div([

        html.Div([
            dcc.Graph(id='bubble_meloni',style={'display': 'inline-block', 'vertical-align': 'center'})
        ],style=dict(display='flex')),

        html.Div([
            dcc.RangeSlider(id='the_year_meloni',
                min=2019,
                max=2021,
                value=[2019,2021],
                marks=mark_values_meloni,
                step=None)
        ],style={"width": "100%", "position":"center",
                 "left":"5%"})
])

### Set the bubble_salvini row
bubble_salvini = html.Div([

        html.Div([
            dcc.Graph(id='bubble_salvini',style={'display': 'inline-block', 'vertical-align': 'center'})
        ],style=dict(display='flex')),

        html.Div([
            dcc.RangeSlider(id='the_year_salvini',
                min=2020,
                max=2021,
                value=[2020,2021],
                marks=mark_values_salvini,
                step=None)
        ],style={"width": "100%", "position":"center",
                 "left":"5%"})
])

### Set the end of the Dash App:
End = dbc.Jumbotron([dbc.Container([html.H6("Student : Alessandro Monolo  |  1790210  -  Lecturer : Erik Hekman", style={'textAlign': 'center', 'color' : 'rgb(255,255,255)'}),
                                    html.H6('Foundamental of Data Science  -  Data-Driven Design Master  -  Hogeschool Utrecht.  AUGUST 2021', style={'textAlign': 'center', 'color' : 'rgb(255,255,255)'})],
                                            fluid=True)],fluid=True)

### Create the Dash App using SLATE Dash theme:
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

### Assemble the Dash App with the above html/core components:
app.layout = html.Div([
                       Headers,
                       First_Row,
                       html.Div(style={'padding': 10}),
                       Second_Row,
                       html.Div(style={'padding': 15}),
                       bubble_grillo,
                       html.Div(style={'padding': 15}),
                       bubble_meloni,
                       html.Div(style={'padding': 15}),
                       bubble_salvini,
                       html.Div(style={'padding': 15}),
                       End
    ])

### Set the 'hashtag' dropdown and its function to return the related plot
@app.callback(
    Output("bar_chart_hashtags", "figure"),
    [Input("selection_hashtags", "value")])
def display_graph(item_selected):
    if item_selected == "beppe_grillo":
        fig = HashtagsGrillo()
    if item_selected == "GiorgiaMeloni":
        fig = HashtagsMeloni()
    if item_selected == "matteosalvinimi":
        fig = HashtagsSalvini()

    return fig

### Set the 'word' dropdown and its function to return the related plot
@app.callback(
    Output("bar_chart_words", "figure"),
    [Input("selection_words", "value")])
def display_graph(item_selected):
    if item_selected == "beppe_grillo":
        fig = WordsGrillo()
    if item_selected == "GiorgiaMeloni":
        fig = WordsMeloni()
    if item_selected == "matteosalvinimi":
        fig = WordsSalvini()

    return fig

### Set the RangeSlider and bubble plot for @beppe_grillo
@app.callback(Output('bubble_grillo','figure'),
              [Input('the_year_grillo','value')])
def update_graph_1(years_chosen):
    df_range_grillo = df_grillo[(df_grillo['tweet_year'] >= years_chosen[0]) & (df_grillo['tweet_year'] <= years_chosen[1])]
    scatterplot_grillo = px.scatter(
        data_frame=df_range_grillo,
        x="datetime",
        y="tweet_lenght",
        size= 'retweet',
        hover_name='tweet_sentiment',
        opacity=0.5,
        color='tweet_emotion',
        height=500, width=1520)
    scatterplot_grillo.update_layout(
        title_text="189 characters is the mean text length per each tweet of @beppe_grillo", 
        title_x=0.5,
        title_font_color="rgb(171,199,255)", 
        title_font_size=15,
        font_color ='rgba(171,199,255,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,8,0,0.9)')
    return (scatterplot_grillo)

### Set the RangeSlider and bubble plot for @GiorgiaMeloni  
@app.callback(Output('bubble_meloni','figure'),
              [Input('the_year_meloni','value')])
def update_graph_2(years_chosen):
    df_range_meloni = df_meloni[(df_meloni['tweet_year'] >= years_chosen[0]) & (df_meloni['tweet_year'] <= years_chosen[1])]
    scatterplot_meloni = px.scatter(
        data_frame=df_range_meloni,
        x="datetime",
        y="tweet_lenght",
        size= 'retweet',
        hover_name='tweet_sentiment',
        opacity=0.5,
        color='tweet_emotion',
        height=500, width=1520)
    scatterplot_meloni.update_layout(
        title_text="247 characters is the mean text length per each tweet of @GiorgiaMeloni", 
        title_x=0.5,
        title_font_color="rgb(171,199,255)", 
        title_font_size=15,
        font_color ='rgba(171,199,255,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,8,0,0.9)')
    return (scatterplot_meloni)

### Set the RangeSlider and bubble plot for @matteosalvinimi
@app.callback(Output('bubble_salvini','figure'),
              [Input('the_year_salvini','value')])
def update_graph_3(years_chosen):
    df_range_salvini = df_salvini[(df_salvini['tweet_year'] >= years_chosen[0]) & (df_salvini['tweet_year'] <= years_chosen[1])]
    scatterplot_salvini = px.scatter(
        data_frame=df_range_salvini,
        x="datetime",
        y="tweet_lenght",
        size= 'retweet',
        hover_name='tweet_sentiment',
        opacity=0.5,
        color='tweet_emotion',
        height=500, width=1520)
    scatterplot_salvini.update_layout(
        title_text="205 characters is the mean text length per each tweet of @matteosalvinimi", 
        title_x=0.5,
        title_font_color="rgb(171,199,255)", 
        title_font_size=15,
        font_color ='rgba(171,199,255,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,8,0,0.9)')
    return (scatterplot_salvini)

### Set the 'SA' dropdown and its function to return the related plot
@app.callback(
    Output("sunburst_chart_SA", "figure"),
    [Input("selection_SA", "value")])
def display_graph(item_selected):
    if item_selected == "beppe_grillo":
        fig = SA_Grillo()
    if item_selected == "GiorgiaMeloni":
        fig = SA_Meloni()
    if item_selected == "matteosalvinimi":
        fig = SA_Salvini()

    return fig

### Set the bar chart retweet/favorite mean and its function to return the related plot
@app.callback(
    Output("bar_chart_emotion", "figure"),
    [Input("selection_barchart", "value")])
def display_graph(item_selected):
    if item_selected == "beppe_grillo":
        fig = BarChartGrillo()
    if item_selected == "GiorgiaMeloni":
        fig = BarChartMeloni()
    if item_selected == "matteosalvinimi":
        fig = BarChartSalvini()

    return fig

### Run the Dash App:
if __name__ == "__main__":
  app.run_server(debug = True)