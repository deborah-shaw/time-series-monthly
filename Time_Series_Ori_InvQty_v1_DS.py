#!/usr/bin/env python
# coding: utf-8

# ## Time_Series_Ori_InvQty_v1_DS
# 
# null

# In[ ]:


# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# In[ ]:


# Check whether running in Fabric or locally, and set the data location accordingly
if "AZURE_SERVICE" in os.environ:
    is_fabric = True
    data_location = "abfss://7e373771-c704-4855-bb94-026ffb6be497@onelake.dfs.fabric.microsoft.com/740e989a-d750-4fd9-a4d9-def5fe22a5db/Files/forecasting/"
    print("Running in Fabric, setting data location to /lakehouse/default/Files/")
else:
    is_fabric = False
    data_location = ""
    print("Running locally, setting data location to current directory")


# In[ ]:


# Load the combined sales economic data
data = pd.read_csv(data_location + 'modelGeneratedData/overall_monthly_with_economic_and_future.csv')
# data = pd.read_csv(data_location + 'modelGeneratedData/overall_monthly_with_trend_seasonality_economic_and_future.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {data.shape}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
print(f"\nData types:")
print(data.dtypes)
print(f"\nFirst few rows:")
data.head()


# ## Data Preparation for Overall Forecasting
# 
# We'll prepare the data for overall quantity forecasting by aggregating all sales data across segments and subcategories to create a single time series for the total quantity.

# In[ ]:


# Create overall aggregation
print("=== CREATING OVERALL AGGREGATION ===")

# Print column names
print(f"Columns in dataset: {data.columns.tolist()}")

# Rename columns in the data
data = data.rename(columns={
    'data_PP_Spot': 'PP_Spot',
    'data_Resin': 'Resin',
    'data_WTI_Crude_Oil': 'WTI_Crude_Oil',
    'data_Natural_Gas': 'Natural_Gas',
    'data_Energy_Average': 'Energy_Average',
    'data_PPI_Freight': 'PPI_Freight',
    'data_PMI_Data': 'PMI_Data',
    'data_Factory_Utilization': 'Factory_Utilization',
    'data_Capacity_Utilization': 'Capacity_Utilization',
    'data_Beverage': 'Beverage',
    'data_Household_consumption': 'Household_consumption',
    'data_packaging': 'packaging',
    'data_Diesel': 'Diesel',
    'data_PPI_Delivery': 'PPI_Delivery',
    'data_Oil-to-resin': 'Oil-to-resin',
    'Electricity Price': 'Electricity Price',
    'Electricity Price (Lag6)': 'Electricity Price (Lag6)',
    'Gas Price': 'Gas Price',
    'Gas Price (Lag6)': 'Gas Price (Lag6)',
    'Global Supply Chain Pressure Index': 'Global Supply Chain Pressure Index',
    'GSCPI (Lag1)': 'GSCPI (Lag1)',
    'Manufacturing Orders Volume Index': 'Manufacturing Orders Volume Index',
    'MOVI (Lag6)': 'MOVI (Lag6)',
    'packaging (Lag2)': 'packaging (Lag2)',
    'PPI_Freight (Lag2)': 'PPI_Freight (Lag2)',
    'trend': 'decomp_trend',
    'seasonality': 'decomp_seasonality',
})

# Define exogenous variables for modeling
exog_vars = [
    'PP_Spot',
    'Resin',
    'PMI_Data',
    'Natural_Gas',
    # 'WTI_Crude_Oil',
    #'Factory_Utilization',
    'packaging',
    'Energy_Average',
    'Electricity Price (Lag6)',
    'Gas Price (Lag6)',
    'Global Supply Chain Pressure Index', # neutral
    # 'future_orders_qty_total',
    'future_orders_qty_next_1m',
    'future_orders_qty_next_3m',
    #'future_orders_qty_next_6m',
    # 'future_orders_qty_next_12m',
    # 'future_orders_avg_lead_time',
    # 'future_orders_min_lead_time',
    # 'future_orders_max_lead_time',
    # 'decomp_trend', 
    #'decomp_seasonality'
]

# Overall aggregation (sum across all segments and subcategories)
print(f"Overall time series shape: {data.shape}")

# Display summary statistics
print("\n=== OVERALL SUMMARY ===")
print(f"Overall total quantity range: {data['Quantity Invoiced'].min():,.0f} - {data['Quantity Invoiced'].max():,.0f}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
print(f"Number of data points: {len(data)}")

# Display columns and their data types
print("\n=== COLUMNS AND DATA TYPES ===")
print(data.dtypes)

data.head()


# In[ ]:


# Visualize the overall time series
fig = go.Figure()

# Plot Overall Total Quantity
fig.add_trace(
    go.Scatter(x=data['Date'], y=data['Quantity Invoiced'],
              mode='lines+markers', name='Overall Total Quantity',
              line=dict(color='darkblue', width=3),
              marker=dict(size=6))
)

fig.update_layout(
    height=600,
    title_text="Overall Sales Quantity Time Series Analysis",
    xaxis_title="Date",
    yaxis_title="Quantity Invoiced",
    showlegend=True,
    template="plotly_white"
)

fig.show()

# Statistical summary
print("\n=== STATISTICAL SUMMARY ===")
print("Overall Quantity Statistics:")
print(data['Quantity Invoiced'].describe())

# Additional time series analysis
print(f"\nTime Series Characteristics:")
print(f"Mean: {data['Quantity Invoiced'].mean():,.0f}")
print(f"Standard Deviation: {data['Quantity Invoiced'].std():,.0f}")
print(f"Coefficient of Variation: {(data['Quantity Invoiced'].std() / data['Quantity Invoiced'].mean() * 100):.2f}%")

# Check for trends and seasonality
data_monthly = data.set_index('Date')['Quantity Invoiced']
monthly_avg = data_monthly.groupby(data_monthly.index.month).mean()
print(f"\nMonthly Averages:")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, avg in monthly_avg.items():
    print(f"  {month_names[month-1]}: {avg:,.0f}")

# Year-over-year growth
yearly_avg = data_monthly.groupby(data_monthly.index.year).mean()
print(f"\nYearly Averages:")
for year, avg in yearly_avg.items():
    print(f"  {year}: {avg:,.0f}")


# # Overall Quantity Forecasting Implementation
# 
# Functions defining the forecasting methods for overall quantity prediction.

# ## Forecasting Models
# 
# We'll implement multiple forecasting approaches:
# 1. **ARIMA** - Auto-regressive Integrated Moving Average
# 2. **SARIMA** - Seasonal ARIMA with economic indicators
# 3. **Exponential Smoothing** - Holt-Winters method
# 4. **Prophet** - Meta's time series forecasting tool with trend and seasonality
# 5. **Ensemble** - Weighted combination of methods

# In[ ]:


def forecast_arima(series, steps=12, order=(1,1,1)):
    """
    ARIMA forecasting with automatic order selection if needed
    """
    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=steps)
        conf_int = fitted_model.get_forecast(steps=steps).conf_int()
        return forecast, conf_int, fitted_model.aic
    except:
        # Try simpler model if original fails
        try:
            model = ARIMA(series, order=(1,0,1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=steps)
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()
            return forecast, conf_int, fitted_model.aic
        except:
            # Last resort - simple naive forecast
            last_value = series.iloc[-1]
            forecast = pd.Series([last_value] * steps)
            conf_int = pd.DataFrame({
                'lower Quantity Invoiced': forecast * 0.9,
                'upper Quantity Invoiced': forecast * 1.1
            })
            return forecast, conf_int, float('inf')

def forecast_sarima(series, steps=12, exog=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    SARIMA forecasting with external regressors
    """
    try:
        model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # For forecast, we need future exogenous variables
        # Use last known values as a simple assumption
        if exog is not None:
            future_exog = pd.DataFrame([exog.iloc[-1]] * steps)
            future_exog.index = pd.date_range(start=exog.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
        else:
            future_exog = None
            
        forecast = fitted_model.forecast(steps=steps, exog=future_exog)
        conf_int = fitted_model.get_forecast(steps=steps, exog=future_exog).conf_int()
        return forecast, conf_int, fitted_model.aic
    except:
        # Fallback to simple ARIMA
        return forecast_arima(series, steps, order)

def forecast_exponential_smoothing(series, steps=12, seasonal_periods=12):
    """
    Exponential Smoothing (Holt-Winters) forecasting
    """
    try:
        if len(series) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        else:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
        
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=steps)
        
        # Simple confidence intervals based on residuals
        residuals = fitted_model.resid
        std_resid = residuals.std()
        conf_int = pd.DataFrame({
            'lower Quantity Invoiced': forecast - 1.96 * std_resid,
            'upper Quantity Invoiced': forecast + 1.96 * std_resid
        })
        
        return forecast, conf_int, fitted_model.aic
    except:
        # Fallback to ARIMA
        return forecast_arima(series, steps)

def forecast_prophet(series, steps=12, exog=None):
    """
    Prophet forecasting with trend, seasonality, and external regressors
    """
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_data = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # Monthly data doesn't need weekly seasonality
        daily_seasonality=False,   # Monthly data doesn't need daily seasonality
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,  # Controls flexibility of trend
        seasonality_prior_scale=10.0,  # Controls flexibility of seasonality
        interval_width=0.95
    )
    
    # Add external regressors if provided
    if exog is not None and len(exog.columns) > 0:
        # Add each regressor to the model
        for col in exog.columns:
            if col in ['PP_Spot', 'Resin', 'WTI_Crude_Oil', 'Natural_Gas', 'Energy_Average']:
                # Energy-related indicators tend to have strong impact
                model.add_regressor(col, prior_scale=0.5, mode='multiplicative')
            elif col in ['PMI_Data', 'Factory_Utilization', 'Capacity_Utilization']:
                # Manufacturing indicators
                model.add_regressor(col, prior_scale=0.3, mode='additive')
            else:
                # Other economic indicators
                model.add_regressor(col, prior_scale=0.1, mode='additive')
        
        # Add regressors to prophet_data
        for col in exog.columns:
            prophet_data[col] = exog[col].values
    
    # Fit the model
    model.fit(prophet_data)
    
    # Create future dataframe
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), 
        periods=steps, 
        freq='MS'
    )
    
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add future regressor values if available
    if exog is not None and len(exog.columns) > 0:
        # Use last known values for future regressors (simple assumption)
        last_regressor_values = exog.iloc[-1]
        for col in exog.columns:
            future_df[col] = last_regressor_values[col]
    
    # Make forecast
    forecast_df = model.predict(future_df)
    
    # Extract forecast values and confidence intervals
    forecast = pd.Series(
        forecast_df['yhat'].values, 
        index=future_dates,
        name='Quantity Invoiced'
    )
    
    conf_int = pd.DataFrame({
        'lower Quantity Invoiced': forecast_df['yhat_lower'].values,
        'upper Quantity Invoiced': forecast_df['yhat_upper'].values
    }, index=future_dates)
    
    # Calculate approximate AIC using cross-validation or residual-based metric
    # Prophet doesn't have built-in AIC, so we'll use MAE on in-sample predictions
    in_sample_forecast = model.predict(prophet_data)
    mae = mean_absolute_error(prophet_data['y'], in_sample_forecast['yhat'])
    pseudo_aic = 2 * len(prophet_data) + 2 * np.log(mae)  # Approximation
    
    return forecast, conf_int, pseudo_aic

def ensemble_forecast(forecasts, aics=None):
    """
    Create ensemble forecast from multiple methods (weighted by inverse AIC)
    """
    weights = []

    if aics is None:
        weights = [1/len(forecasts)] * len(forecasts)
    else:
        weights = [1/aic if aic != float('inf') else 0 for aic in aics]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        else:
            # Equal weights if all AICs are infinite
            weights = [1/len(forecasts)] * len(forecasts)

    # If forecast index is not aligned, reindex to the first forecast's index
    first_index = forecasts[0].index
    for i in range(len(forecasts)):
        if not forecasts[i].index.equals(first_index):
            forecasts[i] = forecasts[i].reindex(first_index)

    # Print weights
    print(f"Model weights - ARIMA: {weights[0]:.3f}, SARIMA: {weights[1]:.3f}, EXP: {weights[2]:.3f}, Prophet: {weights[3]:.3f}")
    
    ensemble = sum(f * w for f, w in zip(forecasts, weights))

    return ensemble

print("Forecasting functions defined successfully!")


# ## Calculate Accuracy Metrics

# In[ ]:


def calculate_accuracy_metrics(actual, predicted, method_name):
    """Calculate comprehensive accuracy metrics"""
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return None
    
    # Calculate metrics
    mae = mean_absolute_error(actual_clean, predicted_clean)
    mse = mean_squared_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero for MAPE
    mape = np.mean(np.abs((actual_clean - predicted_clean) / np.where(actual_clean != 0, actual_clean, 1))) * 100
    
    # Additional metrics
    mean_actual = np.mean(actual_clean)
    mean_predicted = np.mean(predicted_clean)
    bias = np.mean(predicted_clean - actual_clean)
    bias_pct = (bias / mean_actual) * 100 if mean_actual != 0 else 0
    
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - mean_actual) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'Method': method_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Bias': bias,
        'Bias_Percent': bias_pct,
        'Mean_Actual': mean_actual,
        'Mean_Predicted': mean_predicted,
        'Data_Points': len(actual_clean)
    }


# In[ ]:


def forecast_overall(data, exog_vars, forecast_steps, forecast_dates):
    """
    Generate overall forecasts using multiple methods and ensemble approach
    
    Parameters:
    - data: DataFrame with overall time series data
    - exog_vars: List of exogenous variables to use in forecasting
    - forecast_steps: Number of steps to forecast
    - forecast_dates: Date range for forecasts
    
    Returns:
    - DataFrame with all forecast methods and ensemble result
    """
    print("=== LEVEL 0: OVERALL FORECASTING ===")
    
    # Prepare overall data
    overall_series = data.set_index('Date')['Quantity Invoiced']
    display(data.head())
    overall_exog = data.set_index('Date')[exog_vars]
    
    # Generate forecasts using different methods
    print("Generating ARIMA forecast...")
    overall_arima_forecast, overall_arima_conf, overall_arima_aic = forecast_arima(overall_series, forecast_steps)
    
    print("Generating SARIMA forecast...")
    overall_sarima_forecast, overall_sarima_conf, overall_sarima_aic = forecast_sarima(overall_series, forecast_steps, overall_exog)
    
    print("Generating Exponential Smoothing forecast...")
    overall_exp_forecast, overall_exp_conf, overall_exp_aic = forecast_exponential_smoothing(overall_series, forecast_steps)
    
    print("Generating Prophet forecast...")
    overall_prophet_forecast, overall_prophet_conf, overall_prophet_aic = forecast_prophet(overall_series, forecast_steps, overall_exog)

    
    # Print exponential smoothing forecast
    print("Exponential Smoothing Forecast:")
    print(overall_exp_forecast.head())

    # Create ensemble forecast
    aics = [overall_arima_aic, overall_sarima_aic, overall_exp_aic, overall_prophet_aic]    
    overall_ensemble_forecast = ensemble_forecast(
        [overall_arima_forecast, overall_sarima_forecast, overall_exp_forecast, overall_prophet_forecast], 
        aics
    )

    print("Forecast Method Value Lengths:")
    print(f"  ARIMA: {len(overall_arima_forecast)}, SARIMA: {len(overall_sarima_forecast)}, "
        f"ExpSmoothing: {len(overall_exp_forecast)}, Prophet: {len(overall_prophet_forecast)}, "
        f"Ensemble: {len(overall_ensemble_forecast)}")

    
    # Store overall forecasts
    overall_forecasts = pd.DataFrame({
        'Date': forecast_dates,
        'ARIMA': overall_arima_forecast.values,
        'SARIMA': overall_sarima_forecast.values,
        'ExpSmoothing': overall_exp_forecast.values,
        'Prophet': overall_prophet_forecast.values,
        'Ensemble': overall_ensemble_forecast.values,
        'Level': 'Overall',
        'Segment': 'Total'
    })
    
    print(f"Overall forecast range: {overall_ensemble_forecast.min():,.0f} - {overall_ensemble_forecast.max():,.0f}")
    
    return overall_forecasts


# # Running Overall Quantity Forecasting
# 
# Execute the overall quantity forecasting using multiple methods and ensemble approach.

# In[ ]:


# Set forecasting parameters
FORECAST_STEPS = 12  # 12 months ahead
START_DATE = data['Date'].max()
FORECAST_DATES = pd.date_range(start=START_DATE, periods=FORECAST_STEPS, freq='MS')

# Generate overall forecasts
overall_forecasts = forecast_overall(
    data=data,
    exog_vars=exog_vars,
    forecast_steps=FORECAST_STEPS,
    forecast_dates=FORECAST_DATES
)

overall_forecasts


# ## Overall Forecast Visualization & Results
# 
# Let's visualize the overall forecasts and compare them with historical data.

# In[ ]:


# Create comprehensive forecast visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Overall Forecast: Historical vs Predicted', 'Forecast Methods Comparison', 
                    'Forecast Trend Analysis', 'Forecast Method Performance'),
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
    specs=[[{"colspan": 2}, None],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Overall forecast plot (spans both columns in first row)
fig.add_trace(
    go.Scatter(x=data['Date'], y=data['Quantity Invoiced'],
              mode='lines+markers', name='Historical Total',
              line=dict(color='darkblue', width=2),
              marker=dict(size=4)),
    row=1, col=1
)

# Add forecast methods
forecast_colors = {'ARIMA': 'red', 'SARIMA': 'green', 'ExpSmoothing': 'orange', 'Prophet': 'purple', 'Ensemble': 'black'}
for method, color in forecast_colors.items():
    if method in overall_forecasts.columns:
        fig.add_trace(
            go.Scatter(x=overall_forecasts['Date'], y=overall_forecasts[method],
                      mode='lines+markers', name=f'{method} Forecast',
                      line=dict(color=color, width=3 if method == 'Ensemble' else 2, 
                               dash='solid' if method == 'Ensemble' else 'dash'),
                      marker=dict(size=6 if method == 'Ensemble' else 4)),
            row=1, col=1
        )

# Method comparison - just the forecast values
forecast_methods = ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']
method_colors = ['red', 'green', 'orange', 'purple', 'black']

for method, color in zip(forecast_methods, method_colors):
    if method in overall_forecasts.columns:
        fig.add_trace(
            go.Scatter(x=overall_forecasts['Date'], y=overall_forecasts[method],
                      mode='lines+markers', name=f'{method}',
                      line=dict(color=color, width=2),
                      marker=dict(size=5),
                      showlegend=False),
            row=2, col=1
        )

# Forecast statistics by method
if all(method in overall_forecasts.columns for method in forecast_methods):
    method_stats = []
    for method in forecast_methods:
        mean_val = overall_forecasts[method].mean()
        std_val = overall_forecasts[method].std()
        method_stats.append({'Method': method, 'Mean': mean_val, 'Std': std_val})
    
    stats_df = pd.DataFrame(method_stats)
    
    fig.add_trace(
        go.Bar(x=stats_df['Method'], y=stats_df['Mean'],
               name='Mean Forecast',
               marker_color=['red', 'green', 'orange', 'purple', 'black'],
               showlegend=False),
        row=2, col=2
    )

fig.update_layout(
    height=800,
    title_text="Overall Sales Quantity Forecasting: Historical vs Predicted (12-Month Forecast)",
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    template="plotly_white"
)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_xaxes(title_text="Method", row=2, col=2)

fig.update_yaxes(title_text="Total Quantity", row=1, col=1)
fig.update_yaxes(title_text="Total Quantity", row=2, col=1)
fig.update_yaxes(title_text="Mean Forecast", row=2, col=2)

fig.show()

# Print forecast summary
print("\n=== OVERALL FORECAST SUMMARY ===")
print(f"Forecast Period: {FORECAST_DATES[0].strftime('%Y-%m')} to {FORECAST_DATES[-1].strftime('%Y-%m')}")
print(f"\nOverall Forecast Summary:")
print(f"  Mean Monthly Forecast: {overall_forecasts['Ensemble'].mean():,.0f}")
print(f"  Total 12-Month Forecast: {overall_forecasts['Ensemble'].sum():,.0f}")
print(f"  Min-Max Range: {overall_forecasts['Ensemble'].min():,.0f} - {overall_forecasts['Ensemble'].max():,.0f}")

# Historical comparison
historical_mean = data['Quantity Invoiced'].mean()
historical_total_last_12 = data['Quantity Invoiced'].tail(12).sum()

print(f"\nHistorical Comparison:")
print(f"  Historical Mean Monthly: {historical_mean:,.0f}")
print(f"  Historical Last 12-Month Total: {historical_total_last_12:,.0f}")
print(f"  Forecast vs Historical Mean: {((overall_forecasts['Ensemble'].mean() / historical_mean - 1) * 100):+.1f}%")
print(f"  Forecast vs Last 12-Month Total: {((overall_forecasts['Ensemble'].sum() / historical_total_last_12 - 1) * 100):+.1f}%")

# Method comparison
print(f"\nMethod Comparison (12-Month Totals):")
for method in forecast_methods:
    if method in overall_forecasts.columns:
        total_forecast = overall_forecasts[method].sum()
        vs_ensemble = ((total_forecast / overall_forecasts['Ensemble'].sum() - 1) * 100)
        print(f"  {method}: {total_forecast:,.0f} ({vs_ensemble:+.1f}% vs Ensemble)")

# Seasonal analysis of forecast
# print(f"\nSeasonal Forecast Analysis:")
# overall_forecasts['Month'] = overall_forecasts['Date'].dt.month_name()
# monthly_forecast = overall_forecasts.groupby('Month')['Ensemble'].mean()
# print("  Average Monthly Forecast:")
# for month, value in monthly_forecast.items():
#     print(f"    {month}: {value:,.0f}")

# # Show key forecast periods
# print(f"\nKey Forecast Periods:")
# max_month = overall_forecasts.loc[overall_forecasts['Ensemble'].idxmax()]
# min_month = overall_forecasts.loc[overall_forecasts['Ensemble'].idxmin()]
# print(f"  Highest Forecast: {max_month['Date'].strftime('%Y-%m')} ({max_month['Ensemble']:,.0f})")
# print(f"  Lowest Forecast: {min_month['Date'].strftime('%Y-%m')} ({min_month['Ensemble']:,.0f})")

# Growth trajectory
forecast_growth = ((overall_forecasts['Ensemble'].iloc[-1] / overall_forecasts['Ensemble'].iloc[0] - 1) * 100)
print(f"  Forecast Growth (First to Last Month): {forecast_growth:+.1f}%")


# ## Export Results
# 
# Let's save the overall forecasts to CSV files for further analysis and reporting.

# In[ ]:


# Export forecast results
print("=== EXPORTING OVERALL FORECAST RESULTS ===")

# Overall forecast export
overall_export = overall_forecasts[['Date', 'ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']].copy()
overall_export.to_csv(data_location + 'modelGeneratedData/overall_sales_forecast.csv', index=False)
print(f"Overall forecasts exported to: overall_sales_forecast.csv")

# Summary report
summary_report = []
summary_report.append({
    'Level': 'Overall',
    'Segment': 'Total',
    'Total_12Month_Forecast': overall_forecasts['Ensemble'].sum(),
    'Average_Monthly_Forecast': overall_forecasts['Ensemble'].mean(),
    'Min_Monthly_Forecast': overall_forecasts['Ensemble'].min(),
    'Max_Monthly_Forecast': overall_forecasts['Ensemble'].max(),
    'Forecast_Growth_Rate': ((overall_forecasts['Ensemble'].iloc[-1] / overall_forecasts['Ensemble'].iloc[0] - 1) * 100),
    'Vs_Historical_Mean': ((overall_forecasts['Ensemble'].mean() / data['Quantity Invoiced'].mean() - 1) * 100)
})

# Add method-specific summaries
for method in ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet']:
    if method in overall_forecasts.columns:
        summary_report.append({
            'Level': 'Overall',
            'Segment': f'Total_{method}',
            'Total_12Month_Forecast': overall_forecasts[method].sum(),
            'Average_Monthly_Forecast': overall_forecasts[method].mean(),
            'Min_Monthly_Forecast': overall_forecasts[method].min(),
            'Max_Monthly_Forecast': overall_forecasts[method].max(),
            'Forecast_Growth_Rate': ((overall_forecasts[method].iloc[-1] / overall_forecasts[method].iloc[0] - 1) * 100),
            'Vs_Historical_Mean': ((overall_forecasts[method].mean() / data['Quantity Invoiced'].mean() - 1) * 100)
        })

summary_df = pd.DataFrame(summary_report)
summary_df.to_csv(data_location + 'modelGeneratedData/forecast_metrics_summary.csv', index=False)
print(f"Forecast summary report exported to: forecast_metrics_summary.csv")

# Create detailed monthly forecast breakdown
monthly_breakdown = overall_forecasts.copy()
monthly_breakdown['Year'] = monthly_breakdown['Date'].dt.year
monthly_breakdown['Month'] = monthly_breakdown['Date'].dt.month
monthly_breakdown['Month_Name'] = monthly_breakdown['Date'].dt.month_name()
monthly_breakdown['Quarter'] = monthly_breakdown['Date'].dt.quarter

# Calculate quarterly summaries
quarterly_summary = monthly_breakdown.groupby(['Year', 'Quarter']).agg({
    'ARIMA': 'sum',
    'SARIMA': 'sum', 
    'ExpSmoothing': 'sum',
    'Prophet': 'sum',
    'Ensemble': 'sum'
}).reset_index()
quarterly_summary['Quarter_Label'] = quarterly_summary['Year'].astype(str) + '-Q' + quarterly_summary['Quarter'].astype(str)

quarterly_summary.to_csv(data_location + 'modelGeneratedData/quarterly_forecast_summary.csv', index=False)
print(f"Quarterly forecast summary exported to: quarterly_forecast_summary.csv")

print(f"\n=== EXPORT COMPLETE ===")
print(f"Total files exported: 3")
print(f"Export timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Display final summary
print(f"\n=== FINAL FORECAST SUMMARY ===")
display(summary_df)

print(f"\n=== QUARTERLY FORECAST BREAKDOWN ===")
display(quarterly_summary[['Quarter_Label', 'ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']])


# ## 2025 Actuals Comparison

# In[ ]:


# Compute the total Quantity Invoiced for each month of 2025
# 2025 Actuals Comparison so far
print("\n=== 2025 ACTUALS COMPARISON ===")

# Loading Quantity data in pandas DataFrames
# 2015 to 2025 Qty.csv
actual_quantity_df = pd.read_csv(data_location + "userProvidedData/2015-2025 Qty.csv", parse_dates=['Fiscal Hierarchy - Full Date'])
actual_quantity_df = actual_quantity_df[actual_quantity_df['Fiscal Hierarchy - Full Date'].dt.year == 2025]

# Ensure Quantity Invoiced is numeric
actual_quantity_df['Quantity Invoiced'] = actual_quantity_df['Quantity Invoiced'].astype(str).str.replace(',', '')
actual_quantity_df['Quantity Invoiced'] = pd.to_numeric(actual_quantity_df['Quantity Invoiced'], errors='coerce')

# # Filtering Later Quantity Data to match
# print(f"Before filtering quantity data shape: {actual_quantity_df.shape}")

# # Filter to InterCompany
# actual_quantity_df = actual_quantity_df[actual_quantity_df['CustomerSegment'] == 'InterCompany']
# print(f"After filtering to InterCompany segment, quantity data shape: {actual_quantity_df.shape}")

# Group by month and sum the Quantity Invoiced
actual_quantity_df = actual_quantity_df.groupby(actual_quantity_df['Fiscal Hierarchy - Full Date'].dt.to_period('M'))['Quantity Invoiced'].sum().reset_index()
actual_quantity_df['Fiscal Hierarchy - Full Date'] = actual_quantity_df['Fiscal Hierarchy - Full Date'].dt.to_timestamp()
# Rename columns for clarity
actual_quantity_df.rename(columns={'Fiscal Hierarchy - Full Date': 'Date', 'Quantity Invoiced': 'Actual_Quantity'}, inplace=True)

# Display the monthly actuals for 2025
display(actual_quantity_df.head())


# In[ ]:


# Create a CSV file with 2025 actuals and forecast comparison by combining with overall_export
comparison_df = overall_forecasts.merge(actual_quantity_df, on='Date', how='left')
# Drop the 'Level' and 'Segment' columns from overall_export
comparison_df = comparison_df.drop(columns=['Level', 'Segment'])
# Only use rows where Actual_Quantity is not NaN
comparison_df = comparison_df[~comparison_df['Actual_Quantity'].isna()]
# Export as CSV
comparison_df.to_csv(data_location + 'modelGeneratedData/2025_actuals_forecast_comparison.csv', index=False)
print(f"2025 actuals vs forecast comparison exported to: 2025_actuals_forecast_comparison.csv")

forecast_methods = ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet']

# Display comparison df
display(comparison_df)

# Calculate accuracy metrics for each method
accuracy_results = []
for method in forecast_methods:
    if method in comparison_df.columns:
        metrics = calculate_accuracy_metrics(
            comparison_df['Actual_Quantity'].values,
            comparison_df[method].values,
            method
        )
        if metrics:
            accuracy_results.append(metrics)

# Convert results to DataFrame and display
accuracy_df = pd.DataFrame(accuracy_results)
display(accuracy_df)

# Export accuracy metrics to CSV
accuracy_df.to_csv(data_location + 'modelGeneratedData/2025_forecast_accuracy_metrics.csv', index=False)


# In[ ]:


# Plot forcasts methods vs actuals
fig = go.Figure()
# Plot Actual Quantity
fig.add_trace(
    go.Scatter(x=comparison_df['Date'], y=comparison_df['Actual_Quantity'],
              mode='lines+markers', name='Actual Quantity',
              line=dict(color='black', width=2),
              marker=dict(size=4))
)

# Plot each forecast method as a dotted line
for method in forecast_methods:
    if method in comparison_df.columns:
        fig.add_trace(
            go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df[method],
                mode='lines+markers',
                name=f'{method} Forecast',
                line=dict(width=2, dash='dot'),
                marker=dict(size=4)
            )
        )

# Show the plot
fig.show()


# # Backtesting Analysis
# 
# Let's evaluate the performance of our forecasting models by backtesting on previous data. For 2024, we'll train models on data up to 2023 and test predictions against actual 2024 values.

# In[ ]:


# Backtesting Setup: Split data into train (2015-2023) and test (2024)
print("=== BACKTESTING SETUP ===")

# Define split date
BACKTEST_SPLIT_DATE = pd.to_datetime('2023-01-01')
# Forecast horizon
FORECAST_HORIZON = 12
print(f"Training data: Before {BACKTEST_SPLIT_DATE}")
print(f"Test data: {BACKTEST_SPLIT_DATE} onwards")

# Check if we have test data
data_test = data[(data['Date'] >= BACKTEST_SPLIT_DATE) & (data['Date'] < BACKTEST_SPLIT_DATE + pd.DateOffset(months=FORECAST_HORIZON))]
data_train = data[data['Date'] < BACKTEST_SPLIT_DATE]

# Display training and test data columns
print(f"Training data columns: {data_train.columns.tolist()}")
print(f"Test data columns: {data_test.columns.tolist()}")

print(f"Total data points: {len(data)}")
print(f"Training data points: {len(data_train)}")
print(f"Test data points: {len(data_test)}")
print(f"Test Data range: {data_test['Date'].min()} to {data_test['Date'].max()}")

if len(data_test) == 0:
    print("⚠️ No data available for backtesting!")
else:
    print(f"✅ Found {len(data_test)} data points for backtesting")
    print(f"Test months available: {sorted(data_test['Date'].dt.strftime('%Y-%m').unique())}")

# Create training overall aggregation
print("\n=== CREATING TRAINING AGGREGATION ===")

# Overall training data
print(f"Training data shape: {data_train.shape}")
print(f"Training period: {data_train['Date'].min()} to {data_train['Date'].max()}")

# Create actual aggregations for comparison
print("\n=== CREATING ACTUAL AGGREGATIONS ===")

if len(data_test) > 0:
    # Overall actuals
    overall_actual = data_test.groupby('Date').agg({
        'Quantity Invoiced': 'sum',
    }).reset_index()
    
    print(f"Actual data shape: {overall_actual.shape}")
    print(f"Actual period: {overall_actual['Date'].min()} to {overall_actual['Date'].max()}")
    
    print(f"\nOverall monthly totals:")
    for _, row in overall_actual.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m')}: {row['Quantity Invoiced']:,.0f}")
        
    # Basic statistics
    print(f"\nBacktesting Statistics:")
    print(f"  Training samples: {len(data_train)}")
    print(f"  Test samples: {len(overall_actual)}")
    print(f"  Training mean: {data_train['Quantity Invoiced'].mean():,.0f}")
    print(f"  Test mean: {overall_actual['Quantity Invoiced'].mean():,.0f}")
    print(f"  Training std: {data_train['Quantity Invoiced'].std():,.0f}")
    print(f"  Test std: {overall_actual['Quantity Invoiced'].std():,.0f}")
else:
    print("No data available for comparison")


# In[ ]:


# Generate Backtesting Forecasts (Train on 2015-2023, Predict 2024)
if len(data_test) > 0:
    print("=== OVERALL FORECASTING BACKTESTING ===")
    
    # Determine how many months of data we have
    backtest_months = len(overall_actual)
    backtest_dates = pd.date_range(start=BACKTEST_SPLIT_DATE, periods=backtest_months, freq='MS')
    
    print(f"Generating overall forecasts for {backtest_months} months")
    
    # Generate Overall Backtest Forecast using the reusable function
    print("\nGenerating Overall Backtest Forecast...")
    overall_backtest_forecasts = forecast_overall(
        data=data_train,
        exog_vars=exog_vars,
        forecast_steps=backtest_months,
        forecast_dates=backtest_dates
    )
    
    # Store backtest forecasts for overall level
    backtest_forecasts = overall_backtest_forecasts.copy()
    
    # Add actual values
    backtest_forecasts = backtest_forecasts.merge(
        overall_actual[['Date', 'Quantity Invoiced']], 
        on='Date', 
        how='left'
    )
    backtest_forecasts.rename(columns={'Quantity Invoiced': 'Actual'}, inplace=True)
    
    print(f"Backtest forecast range:")
    print(f"  ARIMA: {backtest_forecasts['ARIMA'].min():,.0f} - {backtest_forecasts['ARIMA'].max():,.0f}")
    print(f"  SARIMA: {backtest_forecasts['SARIMA'].min():,.0f} - {backtest_forecasts['SARIMA'].max():,.0f}")
    print(f"  ExpSmoothing: {backtest_forecasts['ExpSmoothing'].min():,.0f} - {backtest_forecasts['ExpSmoothing'].max():,.0f}")
    print(f"  Prophet: {backtest_forecasts['Prophet'].min():,.0f} - {backtest_forecasts['Prophet'].max():,.0f}")
    print(f"  Ensemble: {backtest_forecasts['Ensemble'].min():,.0f} - {backtest_forecasts['Ensemble'].max():,.0f}")
    print(f"Actual range: {overall_actual['Quantity Invoiced'].min():,.0f} - {overall_actual['Quantity Invoiced'].max():,.0f}")
    
    print("\nBacktest vs Actual comparison:")
    display(backtest_forecasts)
    
    # Calculate forecast errors for quick assessment
    if 'Actual' in backtest_forecasts.columns:
        for method in ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']:
            if method in backtest_forecasts.columns:
                mae = np.mean(np.abs(backtest_forecasts['Actual'] - backtest_forecasts[method]))
                mape = np.mean(np.abs((backtest_forecasts['Actual'] - backtest_forecasts[method]) / backtest_forecasts['Actual'])) * 100
                print(f"  {method} - MAE: {mae:,.0f}, MAPE: {mape:.2f}%")
    
else:
    print("⚠️ Skipping backtesting - no 2024 data available")


# In[ ]:


# Calculate Backtesting Accuracy Metrics
if len(data_test) > 0:
    print("\n=== BACKTESTING ACCURACY METRICS ===")
    
    # Calculate metrics for each method
    accuracy_results = []
    
    # Check which methods are available in the forecast data
    methods = ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']
    method_names = ['ARIMA', 'SARIMA', 'Exponential Smoothing', 'Prophet', 'Ensemble']
    
    for method, name in zip(methods, method_names):
        if method in backtest_forecasts.columns:
            metrics = calculate_accuracy_metrics(
                backtest_forecasts['Actual'].values,
                backtest_forecasts[method].values,
                name
            )
            if metrics:
                accuracy_results.append(metrics)
    
    # Convert to DataFrame
    accuracy_df = pd.DataFrame(accuracy_results)
    
    print("Overall Backtesting Accuracy Results:")
    print("="*80)
    for _, row in accuracy_df.iterrows():
        print(f"\n{row['Method']} Performance:")
        print(f"  MAE (Mean Absolute Error): {row['MAE']:,.0f}")
        print(f"  RMSE (Root Mean Square Error): {row['RMSE']:,.0f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {row['MAPE']:.2f}%")
        print(f"  R² (Coefficient of Determination): {row['R2']:.4f}")
        print(f"  Bias: {row['Bias']:,.0f} ({row['Bias_Percent']:.2f}%)")
        print(f"  Mean Actual: {row['Mean_Actual']:,.0f}")
        print(f"  Mean Predicted: {row['Mean_Predicted']:,.0f}")
    
    # Find best performing method
    best_method = accuracy_df.loc[accuracy_df['MAPE'].idxmin(), 'Method']
    best_mape = accuracy_df.loc[accuracy_df['MAPE'].idxmin(), 'MAPE']
    
    print(f"\n🏆 Best Performing Method: {best_method} (MAPE: {best_mape:.2f}%)")
    
    # Export accuracy results
    accuracy_df.to_csv(data_location + 'modelGeneratedData/backtesting_accuracy_metrics.csv', index=False)
    print(f"\n📊 Accuracy metrics exported to: backtesting_accuracy_metrics.csv")
    
    display(accuracy_df)
else:
    print("⚠️ Skipping accuracy calculation - no 2024 data available")


# ### Backtesting Visualization

# In[ ]:


# Backtesting Visualization
if len(data_test) > 0:
    print("\n=== BACKTESTING VISUALIZATION ===")
    
    # Create comprehensive backtesting visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Forecast vs Actual (Overall)', 
            'Forecast Accuracy by Method (MAPE)',
            'Residuals Analysis (Ensemble)',
            'Method Performance Comparison'
        ),
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Forecast vs Actual
    fig.add_trace(
        go.Scatter(
            x=backtest_forecasts['Date'], 
            y=backtest_forecasts['Actual'],
            mode='lines+markers', 
            name='Actual',
            line=dict(color='darkblue', width=3),
            marker=dict(size=6)
        ), row=1, col=1
    )
    
    # Add forecast methods
    colors_bt = ['red', 'green', 'orange', 'purple', 'black']
    methods_bt = ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']
    method_labels = ['ARIMA', 'SARIMA', 'Exp Smoothing', 'Prophet', 'Ensemble']
    
    for method, label, color in zip(methods_bt, method_labels, colors_bt):
        if method in backtest_forecasts.columns:
            fig.add_trace(
                go.Scatter(
                    x=backtest_forecasts['Date'], 
                    y=backtest_forecasts[method],
                    mode='lines+markers', 
                    name=f'{label} Forecast',
                    line=dict(color=color, width=3 if method == 'Ensemble' else 2, 
                             dash='solid' if method == 'Ensemble' else 'dash'),
                    marker=dict(size=6 if method == 'Ensemble' else 4)
                ), row=1, col=1
            )
    
    # Plot 2: MAPE Comparison
    if len(accuracy_results) > 0:
        fig.add_trace(
            go.Bar(
                x=[r['Method'] for r in accuracy_results],
                y=[r['MAPE'] for r in accuracy_results],
                name='MAPE %',
                marker_color=['red' if r['Method'] == best_method else 'lightblue' for r in accuracy_results],
                showlegend=False
            ), row=1, col=2
        )
    
    # Plot 3: Residuals (Actual - Ensemble Forecast)
    if 'Ensemble' in backtest_forecasts.columns:
        residuals = backtest_forecasts['Actual'] - backtest_forecasts['Ensemble']
        fig.add_trace(
            go.Scatter(
                x=backtest_forecasts['Date'], 
                y=residuals,
                mode='lines+markers', 
                name='Residuals (Actual - Ensemble)',
                line=dict(color='red', width=2),
                marker=dict(size=5),
                showlegend=False
            ), row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # Plot 4: Method Performance Metrics
    if len(accuracy_results) > 0:
        methods = [r['Method'] for r in accuracy_results]
        mae_values = [r['MAE'] for r in accuracy_results]
        
        # Create a normalized view of MAE for comparison
        fig.add_trace(
            go.Bar(
                x=methods,
                y=mae_values,
                name='MAE',
                marker_color=['gold' if m == best_method else 'lightcoral' for m in methods],
                showlegend=False
            ), row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Overall Forecasting Backtesting Analysis: Performance & Accuracy Assessment",
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Method", row=2, col=2)
    
    fig.update_yaxes(title_text="Quantity", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=2)
    
    fig.show()
    
    # Additional Analysis: Directional Accuracy
    if 'Ensemble' in backtest_forecasts.columns and len(backtest_forecasts) > 1:
        print("\n=== DIRECTIONAL ACCURACY ANALYSIS ===")
        
        # Calculate month-over-month changes
        actual_changes = backtest_forecasts['Actual'].diff().dropna()
        forecast_changes = backtest_forecasts['Ensemble'].diff().dropna()
        
        # Calculate directional accuracy (same sign of change)
        directional_accuracy = np.mean((actual_changes * forecast_changes) > 0) * 100
        
        print(f"Directional Accuracy: {directional_accuracy:.1f}%")
        print("(Percentage of time forecast correctly predicted direction of change)")
        
        # Additional directional analysis
        correct_direction = (actual_changes * forecast_changes) > 0
        print(f"\nDirectional Analysis:")
        print(f"  Months with correct direction: {correct_direction.sum()}/{len(correct_direction)}")
        print(f"  Months with incorrect direction: {(~correct_direction).sum()}/{len(correct_direction)}")
        
        # Show specific months with wrong direction
        if (~correct_direction).any():
            wrong_dates = backtest_forecasts['Date'].iloc[1:][~correct_direction]
            print(f"  Months with wrong direction: {', '.join(wrong_dates.dt.strftime('%Y-%m'))}")
        
        # Normality test on residuals
        if len(residuals) > 3:
            from scipy import stats
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                print(f"\nResiduals Normality Test (Shapiro-Wilk):")
                print(f"  Test Statistic: {shapiro_stat:.4f}")
                print(f"  P-value: {shapiro_p:.4f}")
                normality_result = "Normal" if shapiro_p > 0.05 else "Not Normal"
                print(f"  Result: {normality_result} at 5% significance level")
            except:
                print(f"\nResiduals Normality Test: Could not be performed")
        
        # Mean Absolute Deviation
        mad = np.mean(np.abs(residuals))
        print(f"\nMean Absolute Deviation: {mad:,.0f}")
        
        # Forecast bias analysis
        bias = np.mean(residuals)
        print(f"Forecast Bias: {bias:,.0f}")
        if bias > 0:
            print("  Interpretation: Forecast tends to underpredict")
        elif bias < 0:
            print("  Interpretation: Forecast tends to overpredict")
        else:
            print("  Interpretation: Forecast is unbiased")
        
        # Percentage bias
        bias_pct = (bias / np.mean(backtest_forecasts['Actual'])) * 100
        print(f"Percentage Bias: {bias_pct:.2f}%")
    
    print("\n=== BACKTESTING VISUALIZATION COMPLETE ===")
    
else:
    print("⚠️ Skipping backtesting visualization - data unavailable")


# ### Backtesting Summary Report

# In[ ]:


# Generate Comprehensive Backtesting Summary Report
if len(data_test) > 0:
    print("\n=== GENERATING BACKTESTING SUMMARY REPORT ===")
    
    # Create comprehensive summary report
    summary_report_data = []
    
    # Overall performance summary
    summary_report_data.append({
        'Section': 'Overall Performance',
        'Metric': 'Best Method',
        'Value': best_method,
        'Details': f'MAPE: {best_mape:.2f}%'
    })
    
    # Add detailed metrics for best method
    best_method_metrics = accuracy_df[accuracy_df['Method'] == best_method].iloc[0]
    
    key_metrics = [
        ('MAE', 'Mean Absolute Error', f"{best_method_metrics['MAE']:,.0f}"),
        ('RMSE', 'Root Mean Square Error', f"{best_method_metrics['RMSE']:,.0f}"),
        ('MAPE', 'Mean Absolute Percentage Error', f"{best_method_metrics['MAPE']:.2f}%"),
        ('R2', 'R-Squared', f"{best_method_metrics['R2']:.4f}"),
        ('Bias', 'Forecast Bias', f"{best_method_metrics['Bias']:,.0f}"),
        ('Bias_Percent', 'Bias Percentage', f"{best_method_metrics['Bias_Percent']:.2f}%")
    ]
    
    for metric_code, metric_name, value in key_metrics:
        summary_report_data.append({
            'Section': 'Best Method Metrics',
            'Metric': metric_name,
            'Value': value,
            'Details': f'{best_method} performance'
        })
    
    # Data coverage
    summary_report_data.append({
        'Section': 'Data Coverage',
        'Metric': 'Training Period',
        'Value': f"{data_train['Date'].min().strftime('%Y-%m')} to {data_train['Date'].max().strftime('%Y-%m')}",
        'Details': f"{len(data_train)} months"
    })
    
    summary_report_data.append({
        'Section': 'Data Coverage',
        'Metric': 'Test Period',
        'Value': f"{data_test['Date'].min().strftime('%Y-%m')} to {data_test['Date'].max().strftime('%Y-%m')}",
        'Details': f"{len(overall_actual)} months"
    })
    
    # Model comparison insights
    if len(accuracy_results) > 1:
        worst_method = accuracy_df.loc[accuracy_df['MAPE'].idxmax(), 'Method']
        worst_mape = accuracy_df.loc[accuracy_df['MAPE'].idxmax(), 'MAPE']
        improvement = worst_mape - best_mape
        
        summary_report_data.append({
            'Section': 'Model Insights',
            'Metric': 'Method Improvement',
            'Value': f"{improvement:.2f}% MAPE reduction",
            'Details': f"{best_method} vs {worst_method}"
        })
        
        # Add all method performances
        for result in accuracy_results:
            summary_report_data.append({
                'Section': 'Method Performance',
                'Metric': f"{result['Method']} MAPE",
                'Value': f"{result['MAPE']:.2f}%",
                'Details': f"MAE: {result['MAE']:,.0f}, R²: {result['R2']:.4f}"
            })
    
    # Statistical significance
    if 'Ensemble' in backtest_forecasts.columns:
        residuals = backtest_forecasts['Actual'] - backtest_forecasts['Ensemble']
        
        # Shapiro-Wilk test for normality of residuals
        try:
            from scipy import stats
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_result = "Normal" if shapiro_p > 0.05 else "Non-normal"
            
            summary_report_data.append({
                'Section': 'Statistical Tests',
                'Metric': 'Residuals Normality',
                'Value': normality_result,
                'Details': f"Shapiro-Wilk p-value: {shapiro_p:.4f}"
            })
        except:
            summary_report_data.append({
                'Section': 'Statistical Tests',
                'Metric': 'Residuals Normality',
                'Value': "Could not test",
                'Details': "Test not available"
            })
        
        # Mean absolute deviation
        mad = np.mean(np.abs(residuals - np.mean(residuals)))
        summary_report_data.append({
            'Section': 'Statistical Tests',
            'Metric': 'Mean Absolute Deviation',
            'Value': f"{mad:,.0f}",
            'Details': "Residuals spread measure"
        })
        
        # Directional accuracy
        if len(backtest_forecasts) > 1:
            actual_changes = backtest_forecasts['Actual'].diff().dropna()
            forecast_changes = backtest_forecasts['Ensemble'].diff().dropna()
            directional_accuracy = np.mean((actual_changes * forecast_changes) > 0) * 100
            
            summary_report_data.append({
                'Section': 'Directional Analysis',
                'Metric': 'Directional Accuracy',
                'Value': f"{directional_accuracy:.1f}%",
                'Details': "Correct direction prediction rate"
            })
    
    # Forecast characteristics
    summary_report_data.append({
        'Section': 'Forecast Characteristics',
        'Metric': 'Training Data Mean',
        'Value': f"{data_train['Quantity Invoiced'].mean():,.0f}",
        'Details': "Average monthly quantity in training"
    })
    
    summary_report_data.append({
        'Section': 'Forecast Characteristics',
        'Metric': 'Test Data Mean',
        'Value': f"{overall_actual['Quantity Invoiced'].mean():,.0f}",
        'Details': "Average monthly quantity in test period"
    })
    
    if 'Ensemble' in backtest_forecasts.columns:
        summary_report_data.append({
            'Section': 'Forecast Characteristics',
            'Metric': 'Ensemble Forecast Mean',
            'Value': f"{backtest_forecasts['Ensemble'].mean():,.0f}",
            'Details': "Average monthly forecast"
        })
    
    # Create summary DataFrame
    summary_df_report = pd.DataFrame(summary_report_data)
    
    # Export summary report
    summary_df_report.to_csv(data_location + 'modelGeneratedData/backtesting_summary_report.csv', index=False)

    # Create a csv with the actuals and overall forecast of each method for each month
    export_df = backtest_forecasts.copy()
    
    # Clean up columns for export
    export_columns = ['Date', 'Actual']
    for method in ['ARIMA', 'SARIMA', 'ExpSmoothing', 'Prophet', 'Ensemble']:
        if method in export_df.columns:
            export_columns.append(method)
    
    export_df = export_df[export_columns]
    export_df.to_csv(data_location + 'backtesting_actuals_vs_forecast_by_method.csv', index=False)
    print('Exported actuals and forecasts by method to: backtesting_actuals_vs_forecast_by_method.csv')

    # Display formatted summary
    print(f"\n{'='*80}")
    print(f"                 OVERALL FORECASTING BACKTESTING SUMMARY")
    print(f"{'='*80}")
    
    current_section = ""
    for _, row in summary_df_report.iterrows():
        if row['Section'] != current_section:
            current_section = row['Section']
            print(f"\n🔹 {current_section.upper()}")
            print(f"{'-'*50}")
        
        print(f"  {row['Metric']:<30}: {row['Value']:<20} ({row['Details']})")
    
    print(f"\n{'='*80}")
    print(f"📊 Complete backtesting summary exported to: backtesting_summary_report.csv")
    print(f"🎯 Overall forecasting backtesting analysis completed successfully!")
    
    # Display final summary table
    print(f"\n📋 FINAL BACKTESTING RESULTS:")
    summary_table = accuracy_df[['Method', 'MAPE', 'MAE', 'R2', 'Bias_Percent']].round(2)
    print(summary_table.to_string(index=False))
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"  • Best performing method: {best_method} with {best_mape:.2f}% MAPE")
    print(f"  • Test period covered: {len(overall_actual)} months")
    print(f"  • Training period: {len(data_train)} months")
    if 'Ensemble' in backtest_forecasts.columns:
        ensemble_bias = np.mean(backtest_forecasts['Actual'] - backtest_forecasts['Ensemble'])
        bias_direction = "under-predicts" if ensemble_bias > 0 else "over-predicts" if ensemble_bias < 0 else "is unbiased"
        print(f"  • Ensemble forecast {bias_direction} by {abs(ensemble_bias):,.0f} units on average")
    
    display(summary_df_report.head(15))
else:
    print("⚠️ Summary report generation skipped - data unavailable")

