import yfinance as yf

def load(ticker: str, timeframe: str = "1d", start: str = "2020-01-01", end: str = "2026-01-01"):
    dat = yf.download(tickers=ticker, start=start, end=end, interval=timeframe)
    return dat