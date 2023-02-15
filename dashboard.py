import spacy
import dash
import requests 
import math
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import html, dcc
from bs4 import BeautifulSoup
from langdetect import detect
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.matcher import DependencyMatcher
from textblob import TextBlob
from lexicalrichness import LexicalRichness
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

title = html.H1(children='Metacritic Dashboard')

input_box = dcc.Input(id='input-box', type='url', size = '75',
    value = 'https://www.metacritic.com/game/playstation-5/cyberpunk-2077/user-reviews')

button = html.Button('Submit', id='button')

datePicker = dcc.DatePickerRange(id='my-date-picker-range', start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period", display_format='MMM Do, YYYY')

startDate = None
endDate = None
platform = None
game = None
reviewer = None
nlp = None

app.layout = html.Div([
    title,
    input_box,
    button,
    datePicker,
    html.Div(id='updade_data'),
    html.Div(id='scores'),
    dcc.Graph(id='graph1'),
    dcc.Graph(id='graph2'),
    dcc.Graph(id='graph3'),
    dcc.Graph(id='graph4'),
    html.Div(id='text1'),
    dcc.Graph(id='graph5'),
    dcc.Graph(id='graph6'),
    dcc.Graph(id='graph7'),
    dcc.Graph(id='graph8'),
    dcc.Graph(id='graph9'),
    html.Div(id='text2'),
    dcc.Graph(id='graph10'),
    dcc.Graph(id='graph11'),
    dcc.Graph(id='graph12'),
    dcc.Graph(id='graph13')
])

@app.callback(
    dash.dependencies.Output('updade_data', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
    [dash.dependencies.Input('my-date-picker-range', 'start_date')],
    [dash.dependencies.Input('my-date-picker-range', 'end_date')])

def update_data(n_clicks, value, start_date, end_date):
    print('updating data')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    global nlp

    startDate = start_date
    endDate = end_date

    # verifica se a próxima página de reviews existe
    def nextPage(soup):
        # print(soup.find(rel="next"))
        if(soup.find(rel="next") == None):
            return False
        
        return True

    url = value
    url = url[32:]
    div = url.find('/')
    platform = url[:div]
    url = url[div+1:]
    div = url.find('/')
    game = url[:div]
    url = url[div+1:]
    div = url.find('/')
    reviewer = url[:]

    try:
        reviews = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
        reviews = reviews.drop(columns=['Unnamed: 0'])
        #reviews.to_csv('dashboard.csv')
        #print('dashboard.csv updated ('+game+'_'+reviewer+'_'+platform+'.csv)')
        print(game+'_'+reviewer+'_'+platform+'.csv')

        aspects = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
        aspects = aspects.drop(columns=['Unnamed: 0'])
        #aspects.to_csv('aspects-dashboard.csv')
        #print('aspects-dashboard.csv updated (aspects-'+game+'_'+reviewer+'_'+platform+'.csv)')
        print('aspects-'+game+'_'+reviewer+'_'+platform+'.csv')

    except:
        print('web scraping')
        review_dict = {'name':[], 'date':[], 'rating':[], 'review':[]}
        page = 0

        url = value + '?page='+str(page)
        user_agent = {'User-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"}
        response  = requests.get(url, headers = user_agent)
        soup = BeautifulSoup(response.text, 'html.parser')

        while((reviewer == 'user-reviews' and (nextPage(soup) or page == 0)) or (reviewer == 'critic-reviews' and page == 0)):
            print("page " + str(page))
            for review in soup.find_all('div', class_='review_content'):
                #if review.find('div', class_='name') == None:
                if review == None:
                    break


                if(reviewer == 'user-reviews'):
                    if review.find('span', class_='blurb blurb_expanded'):
                        rv = review.find('span', class_='blurb blurb_expanded').text
                    else:
                        if(review.find('div', class_='review_body').find('span') is None):
                            rv = ''
                        else:
                            rv = review.find('div', class_='review_body').find('span').text
                else:
                    if(review.find('div', class_='review_body') is None):
                        rv = ''
                    else:
                        rv = review.find('div', class_='review_body').text

                try:
                    language = detect(rv)
                except:
                    continue

                if(language == 'en'):
                    try:
                        if(reviewer == 'user-reviews'):
                            name = review.find('div', class_='name').find('a').text
                            rating = int(review.find('div', class_='review_grade').find_all('div')[0].text)
                        else:
                            name = review.find('div', class_='source').find('a').text
                            rating = int(review.find('div', class_='review_grade').find_all('div')[0].text)
                            rating = round(rating/10)

                        date = review.find('div', class_='date').text
                    except:
                        continue

                    lex = LexicalRichness(rv)
                    words = lex.words
                    mattr = lex.mattr(window_size=max(1,int(lex.words/2)))
                    
                    if(words >= 50 and mattr >= 0.75):
                        review_dict['name'].append(name)
                        review_dict['date'].append(date)
                        review_dict['rating'].append(rating)
                        review_dict['review'].append(rv)

            page += 1
            url = value+'?page='+str(page)
            user_agent = {'User-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"}
            response  = requests.get(url, headers = user_agent)
            soup = BeautifulSoup(response.text, 'html.parser')

        print('saving ' + game+'_'+reviewer+'_'+platform+'.csv')
        reviews = pd.DataFrame(review_dict)
        reviews.to_csv(game+'_'+reviewer+'_'+platform+'.csv')
        #reviews.to_csv('dashboard.csv')

        print('aspect-based sentiment analysis')

        df = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
        df = df.drop(columns=['Unnamed: 0'])

        aspects_dict = []

        pattern = [{"RIGHT_ID": "target", "RIGHT_ATTRS": {"POS": "NOUN"}},
          {"LEFT_ID": "target", "REL_OP": ">", "RIGHT_ID": "modifier", "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "nummod"]}}},]

        matcher = DependencyMatcher(nlp.vocab)
        matcher.add("FOUNDED", [pattern])

        n = len(df)

        for i in range(n):
            date = df['date'][i]
            sentence = df['review'][i]
            rating = df['rating'][i]
            doc = nlp(sentence)
            for match_id, (target, modifier) in matcher(doc):
                if(doc[modifier].pos_ == 'ADJ'):
                    prepend = ''
                    for child in doc[modifier].children:
                        if child.pos_ != 'ADV' and child.text != 'not':
                            continue
                        prepend += child.text + ' '

                    descriptive_term = prepend + doc[modifier].text

                    if((doc[modifier].text, True) in [(e.text,e._.negex) for e in doc.ents]):
                        descriptive_term = 'not ' + descriptive_term
                    
                    descriptive_term = descriptive_term.lower()
                    sentiment = TextBlob(descriptive_term).sentiment
                    aspects_dict.append({'aspect': doc[target].text,'description': descriptive_term, 'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity, 'date': date, 'rating': rating,'review': sentence})
        
        print('saving aspect-' + game+'_'+reviewer+'_'+platform+'.csv')
        aspects = pd.DataFrame(aspects_dict)
        aspects.to_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv')
        #aspects.to_csv('aspects-dashboard.csv')

    return

@app.callback(
    dash.dependencies.Output('scores', 'children'),
    dash.dependencies.Input('updade_data', 'children'))

def update_scores(data):
    print('updating scores')

    global startDate
    global endDate
    global platform
    global game
    global reviewer

    df = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['name'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['name','date','rating','review'])
    df = df_filteredByDate

    n = len(df)

    #mean = 0
    #rvs = 0
    weighted = 0
    weight = 0

    for i in range(n):
        rv = df['review'][i]
        r = df['rating'][i]

        lex = LexicalRichness(rv)
        mattr = lex.mattr(window_size=max(1,int(lex.words/2)))
        
        #mean += r
        #rvs += 1
        
        weighted += mattr * r
        weight += mattr
            
    #mean = mean / rvs
    weighted = weighted / weight

    #colorMean = None
    colorWeighted = None

    #if(mean <= 4.0):
    #    colorMean = 'red'
    #elif(mean <= 7.0):
    #    colorMean = 'yellow'
    #else:
    #    colorMean = 'green'

    if(weighted < 5.0):
        colorWeighted = 'red'
    elif(weighted < 8.0):
        colorWeighted = 'yellow'
    else:
        colorWeighted = 'green'

    return html.Div([
        html.Br(),
        #html.Span('%.1f'%mean,style={'border-radius':'50%','width':'60px','height':'48px',
        #        'padding':'8px','background':colorMean,'color':'white','text-align':'center',
        #        'font':'42px Arial, sans-serif'}),
        #html.Span('\tScore',style={'font':'32px Arial, sans-serif'}),
        #html.Br(), html.Br(),
        html.Span('%.1f'%weighted,style={'border-radius':'50%','width':'60px','height':'48px',
                'padding':'8px','background':colorWeighted,'color':'white','text-align':'center',
                'font':'42px Arial, sans-serif'}),
        html.Span('\tScore Weighted by Lexical Richness',style={'font':'32px Arial, sans-serif'}),
        html.Br(), html.Br(),
    ])

@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    dash.dependencies.Input('scores', 'children'))

def update_graph1(data):
    print('updating graph1')

    global startDate
    global endDate
    global platform
    global game
    global reviewer

    df = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['name'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['name','date','rating','review'])
    df = df_filteredByDate

    n = len(df['rating'])

    ratings = []
    for i in range(n):
        rating = df['rating'][i]
        ratings.append(rating)

    ratings_number = [0]*(11)
    for i in ratings:
        ratings_number[i] += 1

    return {
            'data': [
                {'x': [i for i in range(11)], 'y': ratings_number, 'type': 'bar', 'name': 'SF'}
            ],
            'layout': {
                'title': 'Histogram of Review Score'
            }
        }

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    dash.dependencies.Input('graph1', 'figure'))

def update_graph2(data):
    print('updating graph2')
    
    global startDate
    global endDate
    global platform
    global game
    global reviewer

    df = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['name'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['name','date','rating','review'])
    df = df_filteredByDate

    dates = []

    for d in df['date']:
        date_obj = datetime.strptime(d, '%b %d, %Y')

        monday = 0
        monday_this_week = date_obj + timedelta(monday - date_obj.weekday())
        
        date_string = monday_this_week.strftime('%Y-%m-%d')
        
        dates.append(date_string)
        
    dates_sorted = dates.copy()
    dates_sorted.sort()

    ratings_sum = {}
    var_sum = {}
    ratings_num = {}

    for d in dates_sorted:
        ratings_sum[d] = 0
        var_sum[d] = 0
        ratings_num[d] = 0

    n = len(df)
    x = []

    average_rating = sum(df['rating']) / n

    for i in range(n):
        d = dates[i]
        r = df['rating'][i]
        ratings_sum[d] += r
        var_sum[d] += (r - average_rating)**2
        ratings_num[d] += 1

    dayBefore_sum = 0
    dayBefore_num = 0

    for d in ratings_sum.keys():
        ratings_sum[d] += dayBefore_sum
        ratings_num[d] += dayBefore_num
        dayBefore_sum = ratings_sum[d]
        dayBefore_num = ratings_num[d]

    x = list(ratings_sum.keys())
    y = [ratings_sum[d]/ratings_num[d] for d in ratings_sum.keys()]
    error_y = [math.sqrt(var_sum[d]/ratings_num[d]) for d in ratings_sum.keys()]
    
    return {
            'data': [
                {'x': x, 'y': y, 'name': 'Average', 'error_y':dict(type='data', array=error_y, visible=True)}
                ],
            'layout': {
                'title': 'Average Rating with Standard Deviation Over Time'
            }
        }

@app.callback(
    dash.dependencies.Output('graph3', 'figure'),
    dash.dependencies.Input('graph2', 'figure'))

def update_graph3(data):
    print('updating graph3')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv(game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['name'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['name','date','rating','review'])
    df = df_filteredByDate

    dates = []

    for d in df['date']:
        date_obj = datetime.strptime(d, '%b %d, %Y')
        
        monday = 0
        monday_this_week = date_obj + timedelta(monday - date_obj.weekday())
        
        date_string = monday_this_week.strftime('%Y-%m-%d')
        
        dates.append(date_string)
        
    dates_sorted = dates.copy()
    dates_sorted.sort()
    dates_sorted

    ratings = {'Positive':{}, 'Mixed':{}, 'Negative':{}}

    for s in ['Positive', 'Mixed', 'Negative']:
        for d in dates_sorted:
            ratings[s][d] = 0
        
    n = len(df)
        
    for i in range(n):
        d = dates[i]
        r = df['rating'][i]
        s = ''

        if(r >= 0 and r <= 4):
            s = 'Negative'

        elif(r >= 5 and r <= 7):
            s = 'Mixed'
        
        elif(r >= 8 and r <= 10):
            s = 'Positive'

        ratings[s][d] += 1        
    
    x = list(ratings['Positive'].keys())
    y = []

    for s in ['Positive', 'Mixed', 'Negative']:
        y.append(list(ratings[s].values()))

    return {
        'data': [
            {'x': x, 'y': y[1], 'name': 'Mixed'},
            {'x': x, 'y': y[2], 'name': 'Negative'},
            {'x': x, 'y': y[0], 'name': 'Positive'}
        ],
        'layout': {
            'title': 'Ratings per week'
        }
    }

@app.callback(
    dash.dependencies.Output('graph4', 'figure'),
    dash.dependencies.Input('graph3', 'figure'))

def update_graph4(data):
    print('updating graph4')
    
    global startDate
    global endDate
    global platform
    global game
    global reviewer

    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = []
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity < 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    break
            else:
                descriptions[aspect].append(description)
                topn.append((df['polarity'][i], descriptive_term, df['review'][i]))

    topn.sort()

    n = len(topn)
    topn_unique = []

    for i in range(n):
        p,d,r = topn[i]
        pa,da,ra = topn[i-1]

        if(d != da):
            topn_unique.append((p,d,r))

    x = [d for p,d,r in topn_unique[:20]]
    y = [p for p,d,r in topn_unique[:20]]

    reviews = ''
    for p,d,r in topn_unique[:10]:
        reviews += r + '\n\n'

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Negative terms'
        }
    }

@app.callback(
    dash.dependencies.Output('text1', 'children'),
    dash.dependencies.Input('graph4', 'figure'))

def update_text1(data):
    print('updating text1')
    
    global startDate
    global endDate
    global platform
    global game
    global reviewer
    global nlp

    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = []
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity < 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    break
            else:
                descriptions[aspect].append(description)
                topn.append((df['polarity'][i], description, aspect, df['review'][i]))

    topn.sort()

    n = len(topn)
    topn_unique = []

    for i in range(n):
        p,d,a,r = topn[i]
        pa,da,aa,ra = topn[i-1]
        
        dt = d+' '+a
        dta = da+' '+aa
        
        if(dt != dta):
            topn_unique.append((p,d,a,r))

    pattern = [{"RIGHT_ID": "target", "RIGHT_ATTRS": {"POS": "NOUN"}},
      {"LEFT_ID": "target", "REL_OP": ">", "RIGHT_ID": "modifier", "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "nummod"]}}},]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("FOUNDED", [pattern])

    reviews = []
    cont = 0

    for p,d,a,r in topn_unique:
        if(cont == 20):
            break

        sentence = r
        doc = nlp(sentence)
        for match_id, (target, modifier) in matcher(doc):
            if(doc[modifier].pos_ == 'ADJ'):
                prepend = ''
                for child in doc[modifier].children:
                    if child.pos_ != 'ADV' and child.text != 'not':
                        continue
                    prepend += child.text + ' '

                descriptive_term = prepend + doc[modifier].text

                if((doc[modifier].text, True) in [(e.text,e._.negex) for e in doc.ents]):
                    descriptive_term = 'not ' + descriptive_term
                
                descriptive_term = descriptive_term.lower()
                aspect = doc[target].text
                
                if(d == descriptive_term and a == lemmatizer.lemmatize(aspect)):
                    for i in range(0,modifier):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)
                        
                    reviews.append(html.B(' '+doc[modifier].text))

                    for i in range(modifier+1,target):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)

                    reviews.append(html.B(' '+doc[target].text))

                    for i in range(target+1,len(doc)):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)

                    reviews.append(html.Hr())
                    cont += 1
                    break

    return reviews

@app.callback(
    dash.dependencies.Output('graph5', 'figure'),
    dash.dependencies.Input('text1', 'children'))

def update_graph5(data):
    print('updating graph5')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = {}
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            topn[descriptive_term] = 0
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    descriptive_term = d+' '+aspect
                    topn[descriptive_term] += 1
                    break
            else:
                descriptions[aspect].append(description)
                topn[descriptive_term] += 1

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse = True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Most Frequent Negative terms'
        }
    }

@app.callback(
    dash.dependencies.Output('graph6', 'figure'),
    dash.dependencies.Input('graph5', 'figure'))

def update_graph6(data):
    print('updating graph6')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    total = 0
    topn = {}
    frequency = {}
    polarities = {}
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity < 0.0):
            topn[descriptive_term] = 0
            frequency[descriptive_term] = 0
            polarities[descriptive_term] = polarity
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    descriptive_term = d+' '+aspect
                    frequency[descriptive_term] += 1
                    total += 1
                    break
            else:
                descriptions[aspect].append(description)
                frequency[descriptive_term] += 1
                total += 1
                
    for descriptive_term in list(topn.keys()):
        f = frequency[descriptive_term]
        p = polarities[descriptive_term]
        
        if(f > 0):   
            topn[descriptive_term] = f / total * p

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort()

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Negative Percentage x Sentiment Polarity terms'
        }
    }

@app.callback(
    dash.dependencies.Output('graph7', 'figure'),
    dash.dependencies.Input('graph6', 'figure'))

def update_graph7(data):
    print('updating graph7')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = {}
    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        descriptive_term = df['description'][i]+' '+lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            topn[lemmatizer.lemmatize(df['aspect'][i])] = 0

    for i in range(n):
        descriptive_term = df['description'][i]+' '+lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity < 0.0):
            topn[lemmatizer.lemmatize(df['aspect'][i])] += 1

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse = True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Most Frequent Negative aspects'
        }
    }

@app.callback(
    dash.dependencies.Output('graph8', 'figure'),
    dash.dependencies.Input('graph7', 'figure'))

def update_graph8(data):
    print('updating graph8')
    
    global startDate
    global endDate
    global platform
    global game
    global reviewer

    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    total = 0
    topn = {}
    frequency = {}
    polarity = {}
    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        topn[aspect] = 0
        frequency[aspect] = 0
        polarity[aspect] = df['polarity'][i]

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        frequency[aspect] += 1
        total += 1

    for aspect in list(topn.keys()):
        topn[aspect] = frequency[aspect] / total * polarity[aspect]
        
    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort()

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Negative Percentage x Sentiment Polarity aspects'
        }
    }

@app.callback(
    dash.dependencies.Output('graph9', 'figure'),
    dash.dependencies.Input('graph8', 'figure'))

def update_graph9(data):
    print('updating graph9')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = []
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity > 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    break
            else:
                descriptions[aspect].append(description)
                topn.append((df['polarity'][i], descriptive_term, df['review'][i]))

    topn.sort(reverse=True)

    n = len(topn)
    topn_unique = []

    for i in range(n):
        p,d,r = topn[i]
        pa,da,ra = topn[i-1]

        if(d != da):
            topn_unique.append((p,d,r))

    x = [d for p,d,r in topn_unique[:20]]
    y = [p for p,d,r in topn_unique[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Positive terms'
        }
    }

@app.callback(
    dash.dependencies.Output('text2', 'children'),
    dash.dependencies.Input('graph9', 'figure'))

def update_text2(data):
    print('updating text2')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    global nlp
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = []
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity > 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    break
            else:
                descriptions[aspect].append(description)
                topn.append((df['polarity'][i], description, aspect, df['review'][i]))

    topn.sort(reverse=True)

    n = len(topn)
    topn_unique = []

    for i in range(n):
        p,d,a,r = topn[i]
        pa,da,aa,ra = topn[i-1]
        
        dt = d+' '+a
        dta = da+' '+aa
        
        if(dt != dta):
            topn_unique.append((p,d,a,r))

    pattern = [{"RIGHT_ID": "target", "RIGHT_ATTRS": {"POS": "NOUN"}},
      {"LEFT_ID": "target", "REL_OP": ">", "RIGHT_ID": "modifier", "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "nummod"]}}},]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("FOUNDED", [pattern])

    reviews = []
    cont = 0

    for p,d,a,r in topn_unique:
        if(cont == 20):
            break

        sentence = r
        doc = nlp(sentence)
        for match_id, (target, modifier) in matcher(doc):
            if(doc[modifier].pos_ == 'ADJ'):
                prepend = ''
                for child in doc[modifier].children:
                    if child.pos_ != 'ADV' and child.text != 'not':
                        continue
                    prepend += child.text + ' '

                descriptive_term = prepend + doc[modifier].text

                if((doc[modifier].text, True) in [(e.text,e._.negex) for e in doc.ents]):
                    descriptive_term = 'not ' + descriptive_term
                
                descriptive_term = descriptive_term.lower()
                aspect = doc[target].text
                
                if(d == descriptive_term and a == lemmatizer.lemmatize(aspect)):
                    for i in range(0,modifier):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)
                        
                    reviews.append(html.B(' '+doc[modifier].text))

                    for i in range(modifier+1,target):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)

                    reviews.append(html.B(' '+doc[target].text))

                    for i in range(target+1,len(doc)):
                        w = doc[i].text
                        
                        if(w[0] != ',' and w[0] != '.' and w[0] != '!' and w[0] != '?'):
                            reviews.append(' ')

                        reviews.append(w)

                    reviews.append(html.Hr())
                    cont += 1
                    break

    return reviews

@app.callback(
    dash.dependencies.Output('graph10', 'figure'),
    dash.dependencies.Input('text2', 'children'))

def update_graph10(data):
    print('updating graph10')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = {}
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            topn[descriptive_term] = 0
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    descriptive_term = d+' '+aspect
                    topn[descriptive_term] += 1
                    break
            else:
                descriptions[aspect].append(description)
                topn[descriptive_term] += 1

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse = True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Most Frequent Positive terms'
        }
    }

@app.callback(
    dash.dependencies.Output('graph11', 'figure'),
    dash.dependencies.Input('graph10', 'figure'))

def update_graph11(data):
    print('updating graph11')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    total = 0
    topn = {}
    frequency = {}
    polarities = {}
    descriptions = {}

    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        
        if(polarity > 0.0):
            topn[descriptive_term] = 0
            frequency[descriptive_term] = 0
            polarities[descriptive_term] = polarity
            descriptions[aspect] = []

    for i in range(n):
        description = df['description'][i]
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        descriptive_term = description+' '+aspect
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            for d in descriptions[aspect]:
                token1 = nlp(description)[0]
                token2 = nlp(d)[0]
                if(token1.similarity(token2) >= 0.7):
                    descriptive_term = d+' '+aspect
                    frequency[descriptive_term] += 1
                    total += 1
                    break
            else:
                descriptions[aspect].append(description)
                frequency[descriptive_term] += 1
                total += 1
                
    for descriptive_term in list(topn.keys()):
        f = frequency[descriptive_term]
        p = polarities[descriptive_term]
        
        if(f > 0):   
            topn[descriptive_term] = f / total * p

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse=True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Positive Percentage x Sentiment Polarity terms'
        }
    }

@app.callback(
    dash.dependencies.Output('graph12', 'figure'),
    dash.dependencies.Input('graph11', 'figure'))

def update_graph12(data):
    print('updating graph12')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    topn = {}
    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        descriptive_term = df['description'][i]+' '+lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            topn[lemmatizer.lemmatize(df['aspect'][i])] = 0

    for i in range(n):
        descriptive_term = df['description'][i]+' '+lemmatizer.lemmatize(df['aspect'][i])
        polarity = df['polarity'][i]
        if(polarity > 0.0):
            topn[lemmatizer.lemmatize(df['aspect'][i])] += 1

    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse = True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Most Frequent Positive aspects'
        }
    }

@app.callback(
    dash.dependencies.Output('graph13', 'figure'),
    dash.dependencies.Input('graph12', 'figure'))

def update_graph13(data):
    print('updating graph13')

    global startDate
    global endDate
    global platform
    global game
    global reviewer
    
    df = pd.read_csv('aspects-'+game+'_'+reviewer+'_'+platform+'.csv', lineterminator='\n')
    df = df.drop(columns=['Unnamed: 0'])
    df.description.replace(np.nan, "null", inplace=True)

    filteredByDate = []
    n = len(df)

    for i in range(n):
        date = df['date'][i]
        date_obj = datetime.strptime(date, '%b %d, %Y')
        date_string = date_obj.strftime('%Y-%m-%d')
        
        flagDate = False
        if(startDate is None and endDate is None): 
            flagDate = True
        elif(startDate != None and endDate != None and date_string >= startDate and date_string <= endDate): 
            flagDate = True
        elif(startDate != None and endDate is None and date_string >= startDate): 
            flagDate = True
        elif(startDate is None and endDate != None and date_string <= endDate): 
            flagDate = True

        if(flagDate):
            filteredByDate.append([df['aspect'][i],df['description'][i],df['polarity'][i],df['subjectivity'][i],df['date'][i],df['rating'][i],df['review'][i]])

    df_filteredByDate = pd.DataFrame(data=filteredByDate, columns = ['aspect','description','polarity','subjectivity','date','rating','review'])
    df = df_filteredByDate

    n = len(df)
    total = 0
    topn = {}
    frequency = {}
    polarity = {}
    lemmatizer = WordNetLemmatizer()

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        topn[aspect] = 0
        frequency[aspect] = 0
        polarity[aspect] = df['polarity'][i]

    for i in range(n):
        aspect = lemmatizer.lemmatize(df['aspect'][i])
        frequency[aspect] += 1
        total += 1

    for aspect in list(topn.keys()):
        topn[aspect] = frequency[aspect] / total * polarity[aspect]
        
    topn_sorted = [(topn[k],k) for k in list(topn.keys())] 
    topn_sorted.sort(reverse = True)

    x = [k for v,k in topn_sorted[:20]]
    y = [v for v,k in topn_sorted[:20]]

    return {
        'data': [
            {'x': x, 'y': y, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Top Positive Percentage x Sentiment Polarity aspects'
        }
    }

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")
    ts = termset("en")
    nlp.add_pipe("negex",last=True, config={"neg_termset":ts.get_patterns()})
    app.run_server(debug=True)