from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import datetime
from io import StringIO
import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_tickers', methods=['POST'])
def submit_tickers():
    data = request.json
    tickers = data['tickers'].split(',')
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    combined_data = process_tickers(tickers, start_date, end_date)
    return jsonify(combined_data)

def process_tickers(tickers, start_date, end_date):
    window = 14
    short_window = 50
    long_window = 200
    combined_data = {'Ticker': [], 'Latest Close': [], 'RSI': [], 'P/E Ratio': [], 'P/S Ratio': [],
                     f'SMA_{short_window}': [], f'SMA_{long_window}': [],
                     'Percent Change from 50 to 200': [], 'Volume Surge 100 Days': [],
                     'Gross Margin': [], 'Operating Margin': [], 'LTM Revenue': [],
                     'LTM FCF': [],'Sentiment': []}

    for ticker in tickers:
        try:
            data_with_rsi = fetch_and_calculate_rsi(ticker, start_date, end_date, window)
            if data_with_rsi is None:
                print(f"No data available for ticker {ticker}. Skipping.")
                continue

            prma_data = fetch_and_calculate_prma(ticker, start_date, end_date, short_window, long_window)
            if prma_data is None:
                print(f"No data available for ticker {ticker}. Skipping.")
                continue

            pe_ratio = get_pe_ratio([ticker])[ticker]
            ps_ratios = get_ps_ratio([ticker])

            if ticker not in ps_ratios or isinstance(ps_ratios[ticker], str) and ps_ratios[ticker].startswith("Error"):
                print(f"Skipping ticker {ticker} due to error in P/S ratio calculation.")
                continue

            ps_ratio = ps_ratios[ticker]
            vs = get_vs(ticker, (datetime.datetime.today() - datetime.timedelta(100)).strftime('%Y-%m-%d'), end_date)
            sent = sentiment([ticker])

            if not data_with_rsi.empty:
                rsi = data_with_rsi['RSI'].iloc[-1]

                combined_data['Ticker'].append(ticker)
                combined_data['Latest Close'].append(prma_data['Latest Close'])
                combined_data['RSI'].append(rsi)
                combined_data['P/E Ratio'].append(pe_ratio)
                combined_data['P/S Ratio'].append(ps_ratio)
                combined_data[f'SMA_{short_window}'].append(prma_data[f'SMA_{short_window}'])
                combined_data[f'SMA_{long_window}'].append(prma_data[f'SMA_{long_window}'])
                combined_data['Percent Change from 50 to 200'].append(prma_data['Percent Change from 50 to 200'])
                combined_data['Volume Surge 100 Days'].append(vs)
                combined_data['Gross Margin'].append(get_gross_margin([ticker]))
                combined_data['Operating Margin'].append(get_operating_margin([ticker]))
                combined_data['LTM Revenue'].append(get_ltm_revenue([ticker]))
                combined_data['LTM FCF'].append(get_ltm_fcf([ticker]))
                combined_data['Sentiment'].append(sent)

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            continue

    return combined_data

def fetch_and_calculate_rsi(ticker, start_date, end_date, window=14):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    data['RSI'] = calculate_rsi(data, window)
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_calculate_prma(ticker, start_date, end_date, short_window=50, long_window=200):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    data = calculate_moving_averages(data, short_window, long_window)
    latest_close = data['Close'].iloc[-1]
    latest_short_ma = data[f'SMA_{short_window}'].iloc[-1]
    latest_long_ma = data[f'SMA_{long_window}'].iloc[-1]
    percent_change = ((latest_long_ma - latest_short_ma) / latest_short_ma) * 100

    return {
        'Ticker': ticker,
        'Latest Close': latest_close,
        f'SMA_{short_window}': latest_short_ma,
        f'SMA_{long_window}': latest_long_ma,
        'Percent Change from 50 to 200': f"{percent_change:.2f}%",
        'Close > SMA_50': latest_close > latest_short_ma,
        'Close > SMA_200': latest_close > latest_long_ma
    }

def calculate_moving_averages(data, short_window=50, long_window=200):
    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()
    return data

def get_pe_ratio(tickers):
    pe_ratios = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_price = stock.history(period="1d")['Close'].iloc[0]
            eps = stock.info['trailingEps']
            pe_ratio = stock_price / eps if eps != 0 else 'N/A'
            pe_ratios[ticker] = pe_ratio
        except Exception as e:
            pe_ratios[ticker] = f"Error: {e}"
    return pe_ratios

def get_ps_ratio(tickers):
    ps_ratios = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_price = stock.history(period="1d")['Close'].iloc[0]

            if 'sharesOutstanding' not in stock.info or 'totalRevenue' not in stock.info:
                raise ValueError(f"Required data not found for ticker {ticker}")

            shares_outstanding = stock.info['sharesOutstanding']
            market_cap = stock_price * shares_outstanding
            total_revenue = stock.info['totalRevenue']

            ps_ratio = market_cap / total_revenue
            ps_ratios[ticker] = ps_ratio
        except Exception as e:
            ps_ratios[ticker] = f"Error: {e}"
            print(f"Error fetching data for {ticker}: {e}")

    return ps_ratios

def calculate_vs(data):
    delta = data['Volume']
    avg_volume = delta.mean()
    curr_volume = delta.iloc[-1]
    return 100 * (curr_volume - avg_volume) / avg_volume

def get_vs(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    return calculate_vs(data)

def get_gross_margin(tickers):
    gross_margins = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            profit_margin = stock.financials.loc['Gross Profit'].iloc[0]
            total_revenue = stock.financials.loc['Total Revenue'].iloc[0]
            if total_revenue != 0:
                gross_margin = 100 * profit_margin / total_revenue
            else:
                gross_margin = 'N/A'
            gross_margins[ticker] = gross_margin
        except Exception as e:
            gross_margins[ticker] = f"Error: {e}"
    return gross_margins

def get_operating_margin(tickers):
    operating_margins = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.quarterly_financials
            if len(income_statement.columns) < 1:
                raise ValueError("Not enough data to calculate operating margin")
            operating_income = income_statement.loc['Operating Income'].iloc[0]
            revenue = income_statement.loc['Total Revenue'].iloc[0]
            if revenue == 0:
                raise ValueError("Revenue cannot be zero")
            operating_margin = (operating_income / revenue) * 100
            operating_margins[ticker] = operating_margin
        except Exception as e:
            operating_margins[ticker] = f"Error: {e}"

    return operating_margins

def get_ltm_revenue(tickers):
    ltm_revenues = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            if len(income_statement.columns) < 4:
                raise ValueError("Not enough data to calculate LTM revenue")
            ltm_revenue = income_statement.loc['Total Revenue'].iloc[:4].sum()
            ltm_revenues[ticker] = ltm_revenue
        except Exception as e:
            ltm_revenues[ticker] = f"Error: {e}"
    return ltm_revenues

def get_ltm_fcf(tickers):
    ltm_fcfs = {}
    for ticker in tickers:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)

            # Get the financial data (cash flow statement)
            cash_flow_statement = stock.cashflow


            fcf_ltm = (
                cash_flow_statement.loc['Free Cash Flow'].iloc[:1].sum()
            )

            ltm_fcfs[ticker] = fcf_ltm
        except Exception as e:
            ltm_fcfs[ticker] = f"Error: {e}"

    return ltm_fcfs

    

def sentiment(tickers):
    news = pd.DataFrame()

    for ticker in tickers:
        url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
        ret = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'},
        )

        html = BeautifulSoup(ret.content, "html.parser")

        try:
            df = pd.read_html(
                str(html),
                attrs={'class': 'fullview-news-outer'}
            )[0]
            df.columns = ['Date', 'Headline']  # Correctly indented
            df['Ticker'] = ticker  # Optionally, add the ticker as a column
            news = pd.concat([news, df], ignore_index=True)
        except ValueError:
            return 0

    dateNTime = df.Date.apply(lambda x: ','+x if len(x)<8 else x).str.split(r' |,', expand = True).replace("", None).ffill()
    df = pd.merge(df, dateNTime, right_index=True, left_index=True).drop('Date', axis=1).rename(columns={0:'Date', 1:'Time'})
    df = df[df["Headline"].str.contains("Loading.") == False].loc[:, ['Date', 'Time', 'Headline']]
    df["Ticker"] = ticker
    news = pd.concat([news, df], ignore_index = True)
    news.head()

    nltk.download('vader_lexicon')

    # New words and values
    new_words = {
        'crushes': 10,
        'Buy': 100,
        'Strong Buy': 150,
        'gains': 100,
        'beats': 50,
        'misses': -50,
        'trouble': -100,
        'falls': -100,
        'sell': -100,
        'downgrade': -100,
        'upgraded': 100,
        'outperforms': 50,
        'underperforms': -50,
        'surges': 100,
        'plummets': -100,
        'soars': 100,
        'tumbles': -100,
        'rises': 50,
        'declines': -50,
        'jumps': 100,
        'dips': -50,
        'steady': 0,
        'volatile': -50,
        'bullish': 100,
        'bearish': -100,
        'optimistic': 50,
        'pessimistic': -50,
        'profit': 100,
        'loss': -100,
        'increase': 50,
        'decrease': -50,
        'positive': 50,
        'negative': -50,
        'strong': 50,
        'weak': -50,
        'growth': 50,
        'decline': -50,
        'record': 100,
        'high': 50,
        'low': -50,
        'gain': 50,
        'drop': -50,
        'rise': 50,
        'fall': -50,
        'up': 50,
        'down': -50,
        'advance': 50,
        'retreat': -50,
        'plunge': -100,
        'skyrocket': 100,
        'rebound': 50,
        'slump': -100,
        'jump': 100,
        'crash': -100,
        'collapse': -100,
        'spike': 50,
        'dip': -50,
    }
    # Instantiate the sentiment intensity analyzer with the existing lexicon
    vader = SentimentIntensityAnalyzer()

    # Update the lexicon
    vader.lexicon.update(new_words)
    # Use these column names
    columns = ['Ticker', 'Date', 'Time', 'Headline']
    # Convert the list of lists into a DataFrame
    scored_news = df
    # Iterate through the headlines and get the polarity scores
    scores = [vader.polarity_scores(Headline) for Headline in scored_news.Headline.values]
    # Convert the list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames
    scored_news = pd.concat([scored_news, scores_df], axis=1)


    scored_news.head()
    # prompt: Only store all value that have date todat in the scored_news
    scored_news = scored_news[scored_news['Date'] == 'Today']
    scored_news.head()

    print(scored_news.head())

    return np.mean(scored_news['compound'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
