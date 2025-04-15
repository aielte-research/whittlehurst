import yfinance as yf
from plot_volatility import plot_volatility

data = yf.download("^VIX", start="2007-04-02", end="2025-03-12")
print(data)
data['calc'] = data['Close']
#data['calc'] = data['High']
data['calc'] = 0.01*data['calc'].dropna()
#data['calc'] = np.log(data['calc']).dropna()

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

plot_volatility(
    vol, dates, "Daily CBOE Volatility Index", f"volatility_VIX", window=252, stride=63,         
)