# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('stock_prices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for missing values
data.fillna(method='ffill', inplace=True)

# Visualize the stock price
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price')
plt.title('Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Decompose the time series
result = seasonal_decompose(data['Close'], model='additive', period=30)
result.plot()
plt.show()

# Fit ARIMA model
model = ARIMA(data['Close'], order=(1, 1, 1))
arima_result = model.fit()

# Forecasting future prices with ARIMA
forecast = arima_result.forecast(steps=10)
print(f'ARIMA Forecast:\n{forecast}')

# Prepare data for Prophet
prophet_data = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Create and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Make future dataframe for forecasting
future = prophet_model.make_future_dataframe(periods=10)
forecast_prophet = prophet_model.predict(future)

# Plot the results of Prophet
prophet_model.plot(forecast_prophet)
plt.title('Prophet Forecast')
plt.show()

# Calculate RMSE for ARIMA
arima_rmse = np.sqrt(mean_squared_error(data['Close'][-10:], forecast))
print(f'ARIMA RMSE: {arima_rmse:.2f}')

# Calculate RMSE for Prophet
prophet_rmse = np.sqrt(mean_squared_error(data['Close'][-10:], forecast_prophet['yhat'][-10:]))
print(f'Prophet RMSE: {prophet_rmse:.2f}')
