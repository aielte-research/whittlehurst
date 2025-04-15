import numpy as np
import pandas as pd
from plot_volatility import plot_volatility

try:
    df = pd.read_csv('SnP500-5m.csv', sep=';', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
except:
    df = pd.read_csv('volatility/SnP500-5m.csv', sep=';', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

print(df)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],format="%d/%m/%Y %H:%M:%S")
df['Date'] = pd.to_datetime(df['Date'],format="%d/%m/%Y")
df = df.set_index('Datetime')
df = df.drop(columns=['Time'])
# df = df.resample('1H').agg({
#     'Date': 'first',
#     'Open': 'first',
#     'High': 'max',
#     'Low': 'min',
#     'Close': 'last',
#     'Volume': 'sum',
# }).dropna()

print(df)
df['calc'] = df['Close']
df['calc'] = np.log(df['calc'] / df['calc'].shift(1))
df=df.dropna()
print(df)
df = df.resample('1D').agg({
    'Date': 'first',
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'calc': 'std'
}).dropna()
print(df)
# df = df.groupby('Date').std().reset_index()
# print(df)
df = df.set_index('Date')
df = df.sort_values(by=['Date'])
df['calc'] = df['calc'] *np.sqrt(24*12)
print(df)
data = df

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

plot_volatility(vol, dates, "S&P 500 daily Log-Volatility", "volatility_SnP500_daily_5m", window=252, stride=63)