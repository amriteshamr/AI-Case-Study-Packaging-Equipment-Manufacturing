import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load your data
# Assuming you have three separate datasets: availability_data, performance_data, and quality_data
# Combine them to calculate OEE
# For simplicity, let's assume your data is stored in CSV files with date and OEE columns
oee_data = pd.read_csv("oee_results.csv", parse_dates=["Date"], index_col="Date")


# Check for stationarity
def check_stationarity(data):
    result = adfuller(data)
    print('ADF Statistic: {:.3f}'.format(result[0]))
    print('p-value: {:.3f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.3f}'.format(key, value))

check_stationarity(oee_data)

# If the data is not stationary, apply differencing
differenced_oee = oee_data.diff().dropna()

# Identify parameters for ARIMA model
plot_acf(differenced_oee)
plt.show()

plot_pacf(differenced_oee)
plt.show()

# Train-test split
train_size = int(len(differenced_oee) * 0.8)
train, test = differenced_oee.iloc[:train_size], differenced_oee.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast
forecast, stderr, conf_int = model_fit.forecast(steps=len(test))

# Invert differencing
def invert_diff(history, forecast):
    forecast = np.cumsum(forecast)
    return forecast + history.iloc[-1]

predicted_oee = invert_diff(oee_data.iloc[train_size-1], forecast)

# Evaluate the model
mse = mean_squared_error(test, predicted_oee)
print('Mean Squared Error:', mse)

# Plot results
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predicted_oee, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('OEE')
plt.title('ARIMA Forecast vs Actual OEE')
plt.legend()
plt.show()
