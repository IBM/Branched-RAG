import yfinance as yf
import os
import json
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
import entity_extraction



def recommendation_and_financials_data(data,directory):

    recommendations_file_path = os.path.join(directory, 'recommendations_data.json')
    financials_file_path = os.path.join(directory, 'financials_data.json')

    if os.path.exists(recommendations_file_path):
        with open(recommendations_file_path, 'r') as file:
            recommendations_data = json.load(file)
    else:
        recommendations_data = {}

    if os.path.exists(financials_file_path):
        with open(financials_file_path, 'r') as file:
            financials_data = json.load(file)
    else:
        financials_data = {}

    for stock in data.get("stocks", []):
        symbol = stock.get("symbol")

        if symbol:
            try:
                ticker = yf.Ticker(symbol)

                recommendation_data = ticker.recommendations.to_json()
                recommendations_data[symbol] = json.loads(recommendation_data)

                financial_data = ticker.financials.to_json()
                financials_data[symbol] = json.loads(financial_data)

                print(f"Recommendation and financials JSON data for {symbol} have been added to the combined data.")

            except Exception as e:
                print(f"An error occurred while processing symbol {symbol}: {e}")

    with open(recommendations_file_path, 'w') as file:
        json.dump(recommendations_data, file, indent=4)

    print(f"Recommendations JSON data has been saved to {recommendations_file_path}")

    with open(financials_file_path, 'w') as file:
        json.dump(financials_data, file, indent=4)

    print(f"Financials JSON data has been saved to {financials_file_path}")


########################################################################################################################################################

#### newsapi data



def news_data(data,directory):
    # Initialize the News API client
    api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

    combined_news_file_path = os.path.join(directory, 'all_stocks_news_data.json')

    if os.path.exists(combined_news_file_path):
        with open(combined_news_file_path, 'r') as file:
            all_news_data = json.load(file)
    else:
        all_news_data = {}

    for stock in data.get("stocks", []):
        symbol = stock.get("symbol")
        if symbol:
            try:
                response = api.get_everything(q=symbol)
                articles = response.get('articles', [])

                news_list = [{"title": article['title']} for article in articles]

                if symbol in all_news_data:
                    all_news_data[symbol]["top_news_headlines"].extend(news_list)
                else:
                    all_news_data[symbol] = {"top_news_headlines": news_list}

                print(f"News data for {symbol} has been added to the combined data.")

            except Exception as e:
                print(f"An error occurred while processing news for symbol {symbol}: {e}")

    with open(combined_news_file_path, 'w') as file:
        json.dump(all_news_data, file, indent=4)

    print(f"Combined news JSON data has been saved to {combined_news_file_path}")



#########################################################################################################################################################
#### alphavantage data


def stock_performance_data(data,directory):
    # Initialize the Alpha Vantage TimeSeries client
    ts = TimeSeries(key=os.getenv('alpha_vantage_key'), output_format='pandas')


    combined_performance_file_path = os.path.join(directory, 'stock_performance_data.json')

    # Load existing data if the file exists
    if os.path.exists(combined_performance_file_path):
        with open(combined_performance_file_path, 'r') as file:
            all_stock_performance_data = json.load(file)
    else:
        all_stock_performance_data = {}

    for stock in data.get("stocks", []):
        symbol = stock.get("symbol")

        if symbol:
            try:
                stock_data, meta_data = ts.get_monthly(symbol=symbol)
                stock_data_json = stock_data.to_json()

                if symbol in all_stock_performance_data:
                    all_stock_performance_data[symbol].update(json.loads(stock_data_json))
                else:
                    all_stock_performance_data[symbol] = json.loads(stock_data_json)

                print(f"Performance data for {symbol} has been added to the combined data.")

            except Exception as e:
                print(f"An error occurred while processing data for symbol {symbol}: {e}")

    with open(combined_performance_file_path, 'w') as file:
        json.dump(all_stock_performance_data, file, indent=4)

    print(f"Combined stock performance JSON data has been saved to {combined_performance_file_path}")

def generate_data(question,directory):

    data = entity_extraction.entity_extraction(question)
    recommendation_and_financials_data(data,directory)
    news_data(data,directory)
    stock_performance_data(data,directory)
    return data
