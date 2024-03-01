import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from stocknews import StockNews  # Import StockNews here
import datetime

# Fetch historical stock data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to fetch fundamental data
def get_fundamental_data(symbol, period, statement):
    if period == 'annual':
        if statement == 'balance sheet':
            return yf.Ticker(symbol).balance_sheet.T
        elif statement == 'income statement':
            return yf.Ticker(symbol).financials.T
        elif statement == 'cash flow':
            return yf.Ticker(symbol).cashflow.T
    elif period == 'quarterly':
        if statement == 'balance sheet':
            return yf.Ticker(symbol).quarterly_balance_sheet.T
        elif statement == 'income statement':
            return yf.Ticker(symbol).quarterly_financials.T
        elif statement == 'cash flow':
            return yf.Ticker(symbol).quarterly_cashflow.T
    else:
        st.error('Wrong entry')

# Calculate moving averages
def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Generate buy/sell signals based on moving average crossovers
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Buy'] = np.where(data['Short_MA'] > data['Long_MA'], 1.0, 0.0)
    signals['Sell'] = np.where(data['Short_MA'] < data['Long_MA'], -1.0, 0.0)
    signals['Signal'] = signals['Buy'] + signals['Sell']
    return signals

def main():
    # Streamlit UI
    st.markdown(
        """
        <h1 style='text-align: center; color: black;'>📈Equity Research Screener📈</h1>
        """,
        unsafe_allow_html=True
    )

    # Sidebar inputs
    symbol = st.sidebar.text_input("Enter stock symbol", "WIPRO.NS")
    start_date = st.sidebar.text_input("Enter start date (YYYY-MM-DD)", "2023-01-01")
    end_date = st.sidebar.text_input("Enter end date (YYYY-MM-DD)", datetime.date.today().strftime('%Y-%m-%d'))  # Use today's date as the default
    short_window = st.sidebar.slider("Short window", 1, 100, 22)
    long_window = st.sidebar.slider("Long window", 1, 200, 44)  # Adjusted the range and default value
    period = st.sidebar.selectbox('Period', ['annual', 'quarterly'], key='period_selectbox')
    statement = st.sidebar.selectbox('Statement', ['balance sheet', 'income statement', 'cash flow'], key='statement_selectbox')
    selected_tab = st.sidebar.radio("Navigation", ["Company Profile", "Summary and Statistical Data", "Candlestick Chart", "Moving Averages and Signals", "Volume Data", "Fundamental Data", "Additional Information", "News"])

    # Fetch data
    data = fetch_data(symbol, start_date, end_date)

    # Calculate moving averages
    data = calculate_moving_averages(data, short_window, long_window)

    # Generate signals
    signals = generate_signals(data)

    # Filter data based on start and end dates
    filtered_data = data.loc[start_date:end_date]

    # Fetch fundamental data
    fd_data = get_fundamental_data(symbol, period, statement)

    # Fetch additional company information
    ticker = yf.Ticker(symbol)

    if selected_tab == "Company Profile":
        # Display company profile
        st.subheader("Company Profile")
        info = ticker.info
        st.write("**Sector:**", info.get('sector', 'N/A'))
        st.write("**Industry:**", info.get('industry', 'N/A'))
        st.write("**Country:**", info.get('country', 'N/A'))
        st.write("**Website:**", info.get('website', 'N/A'))
        # Display key executives
        st.subheader("Key Executives")
        executives = info.get('companyOfficers', [])
        for executive in executives:
            st.write(executive['name'], "-", executive['title'])

    elif selected_tab == "Summary and Statistical Data":
        # Display summary of the stock and statistical data
        st.subheader("Summary and Statistical Data")
        try:
            info = ticker.info  # Moved the assignment here
            st.write("**Market Cap:**", info.get('marketCap', 'N/A'))
            st.write("**Forward PE Ratio:**", info.get('forwardPE', 'N/A'))
            st.write("**Trailing PE Ratio:**", info.get('trailingPE', 'N/A'))
            st.write("**Dividend Yield:**", info.get('dividendYield', 'N/A'))
            st.write("**Beta:**", info.get('beta', 'N/A'))
            st.write("**Mean Close Price:**", filtered_data['Close'].mean())
            st.write("**Standard Deviation Close Price:**", filtered_data['Close'].std())
            st.write("**Minimum Close Price:**", filtered_data['Close'].min())
            st.write("**Maximum Close Price:**", filtered_data['Close'].max())
            if 'fiftyTwoWeekHigh' in info:
                st.write("**52-week High:**", info['fiftyTwoWeekHigh'])
            if 'fiftyTwoWeekLow' in info:
                st.write("**52-week Low:**", info['fiftyTwoWeekLow'])
        except Exception as e:
            st.error("Error: " + str(e))

    elif selected_tab == "Candlestick Chart":
        # Display candlestick chart
        st.subheader("Candlestick Chart")
        fig_candlestick = go.Figure()
        # Candlestick
        fig_candlestick.add_trace(go.Candlestick(x=filtered_data.index,
                                                  open=filtered_data['Open'],
                                                  high=filtered_data['High'],
                                                  low=filtered_data['Low'],
                                                  close=filtered_data['Close'], name='market data'))
        # Add 20-day SMA
        fig_candlestick.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Short_MA'], name='20-day SMA', line=dict(color='blue')))
        # Add 200-day SMA
        fig_candlestick.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Long_MA'], name='200-day SMA', line=dict(color='red')))
        # Update layout for candlestick chart
        fig_candlestick.update_layout(
            title='Candlestick chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=4, label="4h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="todate"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),  # Weekly time frame
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        st.plotly_chart(fig_candlestick)

    elif selected_tab == "Moving Averages and Signals":
        # Display moving averages and signals
        st.subheader("Moving Averages and Signals")
        fig = go.Figure()
        # Close price
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

        # Short moving average
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Short_MA'], mode='lines', name='Short Moving Average', line=dict(color='orange')))

        # Long moving average
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Long_MA'], mode='lines', name='Long Moving Average', line=dict(color='green')))

        # Buy signals
        buy_signals = filtered_data.loc[signals['Signal'] == 1.0]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Short_MA'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=8, color='green')))

        # Sell signals
        sell_signals = filtered_data.loc[signals['Signal'] == -1.0]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Short_MA'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=8, color='red')))

        # Update layout for the main figure
        fig.update_layout(
            title='Moving Averages and Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis=dict(tickformat='%Y-%m-%d'),
            yaxis=dict(type='linear'),
            showlegend=True
        )

        # X-Axes configuration for Plotly figure
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="todate"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),  # Weekly time frame
                    dict(step="all")
                ])
            )
        )

        # Show Plotly figure
        st.plotly_chart(fig)

    elif selected_tab == "Volume Data":
        # Display volume data
        st.subheader("Volume Data")
        volume_data = data['Volume']
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=volume_data.index, y=volume_data, name='Volume', marker=dict(color='blue')))
        fig_volume.update_layout(title="Volume", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_volume)

    elif selected_tab == "Fundamental Data":
        # Display fundamental data
        st.subheader("Fundamental Data")
        if fd_data is not None:
            st.write(fd_data.head())

            # Visualize fundamental data in a pie chart
            st.subheader("Fundamental Data Pie Chart:")
            labels = fd_data.columns
            values = fd_data.iloc[0].values
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
            st.plotly_chart(fig_pie)

        else:
            st.error("Failed to retrieve fundamental data.")

    elif selected_tab == "Additional Information":
        # Display additional information
        st.subheader("Additional Information")
        st.write("**Major Holders:**")
        st.write(ticker.major_holders)

        st.write("**Insider Transactions:**")
        st.write(ticker.get_insider_transactions())

    elif selected_tab == "News":
        # News section
        st.subheader('Stock News')
        with st.expander("Latest News"):
            st.header(f'News of {symbol}')
            sn = StockNews(symbol, save_news=False)
            df_news = sn.read_rss()
            for i in range(10):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                title_sentiment = df_news['sentiment_title'][i]
                st.write(f'Title Sentiment: {title_sentiment}')
                news_sentiment = df_news['sentiment_summary'][i]
                st.write(f'News Sentiment: {news_sentiment}')


if __name__ == "__main__":
    main()
