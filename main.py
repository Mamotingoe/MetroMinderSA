# South African Metro Service Delivery Dashboard

## 1. Web Scraping
from scrapy import Spider, Request
from newspaper import Article

class NewsSpider(Spider):
    name = 'news_spider'
    start_urls = [
        # Add relevant South African news sites here
        'https://www.news24.com/',
        'https://www.iol.co.za/',
        # Add more as needed
    ]

    def parse(self, response):
        # Extract news article links
        article_links = response.css('a.article-link::attr(href)').getall()
        for link in article_links:
            yield Request(link, callback=self.parse_article)

    def parse_article(self, response):
        article = Article(response.url)
        article.download()
        article.parse()

        yield {
            'title': article.title,
            'text': article.text,
            'date': article.publish_date,
            'url': response.url
        }

## 2. Data Processing and Analysis
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def process_data(df):
    # Preprocess text data
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        return ' '.join([w for w in tokens if w not in stop_words])

    df['processed_text'] = df['text'].apply(preprocess_text)

    # Add more data processing steps as needed
    return df

## 3. Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(df):
    X = df['processed_text']
    y = df['service_delivery_issue']  # You'll need to create this label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)
    print(classification_report(y_test, predictions))

    return model, vectorizer

## 4. Dashboard Creation
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('South African Metro Service Delivery Dashboard'),
    dcc.Dropdown(
        id='municipality-dropdown',
        options=[
            {'label': 'City of Johannesburg', 'value': 'JHB'},
            {'label': 'City of Cape Town', 'value': 'CPT'},
            {'label': 'eThekwini Metropolitan Municipality', 'value': 'ETH'},
            {'label': 'City of Tshwane', 'value': 'TSH'},
            {'label': 'Nelson Mandela Bay Municipality', 'value': 'NMB'},
            {'label': 'City of Ekurhuleni', 'value': 'EKU'},
            {'label': 'Buffalo City Metropolitan Municipality', 'value': 'BCM'},
            {'label': 'Mangaung Metropolitan Municipality', 'value': 'MAN'}
        ],
        value='JHB'
    ),
    dcc.Graph(id='service-delivery-graph')
])

@app.callback(
    Output('service-delivery-graph', 'figure'),
    Input('municipality-dropdown', 'value')
)
def update_graph(selected_municipality):
    # Update this function to display relevant data for the selected municipality
    pass

if __name__ == '__main__':
    app.run_server(debug=True)

## 5. Main Execution
if __name__ == "__main__":
    # Run web scraping
    # Process and analyze data
    # Train predictive model
    # Launch dashboard
    pass
