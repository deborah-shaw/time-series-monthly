#!/usr/bin/env python
# coding: utf-8

# ## Time_Series_Ori_v1_DS
# 
# New notebook

# # Hierarchical Sales Forecasting: Overall, Customer Segment & SubCategory
# 
# This notebook implements grouped time series forecasting at three hierarchical levels:
# 1. **Overall** - Total quantity across all segments
# 2. **Customer Segment** - Aggregated by customer segments
# 3. **SubCategory** - Aggregated by product subcategories
# 
# We'll use multiple forecasting approaches and ensure hierarchical consistency.

# In[16]:


import mlflow 
mlflow.set_experiment("Time_Series_v1_DS")


# In[1]:


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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# In[2]:


# Check whether running in Fabric or locally, and set the data location accordingly
if "AZURE_SERVICE" in os.environ:
    is_fabric = True
    data_location = "/lakehouse/default/Files/"
    print("Running in Fabric, setting data location to /lakehouse/default/Files/")
else:
    is_fabric = False
    data_location = ""
    print("Running locally, setting data location to current directory")


# In[3]:


# Load the combined sales economic data
data = pd.read_csv(data_location + 'forecasting/userProvidedData/combined_sales_economic_data_segmented.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {data.shape}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
print(f"Unique Customer Segments: {data['CustomerSegment'].nunique()}")
print(f"Unique SubCategories: {data['SubCategoryName'].nunique()}")
print(f"\nCustomer Segments: {sorted(data['CustomerSegment'].unique())}")
print(f"\nSubCategories: {sorted(data['SubCategoryName'].unique())}")
print(f"\nData types:")
print(data.dtypes)
print(f"\nFirst few rows:")
data.head()


# ## Data Preparation for Hierarchical Forecasting
# 
# We'll create three levels of aggregation:
# 1. **Level 0 (Overall)**: Total quantity across all segments and subcategories
# 2. **Level 1 (Customer Segment)**: Aggregated by customer segment
# 3. **Level 2 (SubCategory)**: Aggregated by product subcategory

# In[4]:


# Create hierarchical aggregations
print("=== CREATING HIERARCHICAL AGGREGATIONS ===")

# Holds columns and their aggregation functions
column_aggregations = {
    'Total_Quantity': 'sum',
    'Transaction_Count': 'sum',
    'Unique_Customers': 'sum',
    'Unique_Products': 'sum',
    # Economic indicators (take mean as they're external factors)
    'PP_Spot': 'mean',
    'Resin': 'mean',
    'WTI_Crude_Oil': 'mean',
    'Natural_Gas': 'mean',
    'Electricity Price': 'mean',
    'Gas Price': 'mean',
    'Energy_Average': 'mean',
    'PPI_Freight': 'mean',
    'PMI_Data': 'mean',
    'Factory_Utilization': 'mean',
    'Capacity_Utilization': 'mean',
    'Beverage': 'mean', # Additional economic indicator
    'Household_consumption': 'mean',
    'packaging': 'mean',
    'Diesel': 'mean',
    'PPI_Delivery': 'mean',
    'Oil-to-resin': 'mean',
    'Electricity Price (Lag6)': 'mean',
    'Gas Price (Lag6)': 'mean'
}

# Define exogenous variables for modeling
exog_vars = [
    'PP_Spot',
    'Resin',
    'PMI_Data',
    'Natural_Gas',
    'WTI_Crude_Oil',
    'Factory_Utilization',
    'packaging',
    'Energy_Average',
    'Electricity Price (Lag6)',
    'Gas Price (Lag6)'
]

    # 'PPI_Delivery' slightly negative
    # 'PMI_Data', major positive
    # 'PPI_Freight', negative
    # 'Factory_Utilization',
    # 'Capacity_Utilization', negative
    # 'Beverage', minor negative
    # 'Household_consumption', major negative
    # 'packaging' minor positive
    # 'Diesel', minor positive
    # 'Natural_Gas' major positive
    # 'Electricity Price (Lag6)', positive
    # 'Gas Price (Lag6)' positive


# Level 0: Overall aggregation (sum across all segments and subcategories)
overall_ts = data.groupby('Date').agg(column_aggregations).reset_index()
overall_ts['Level'] = 'Overall'
overall_ts['Segment'] = 'Total'

# Level 1: Customer Segment aggregation
segment_ts = data.groupby(['Date', 'CustomerSegment']).agg(column_aggregations).reset_index()
segment_ts['Level'] = 'CustomerSegment'
segment_ts['Segment'] = segment_ts['CustomerSegment']

# Level 2: SubCategory aggregation
subcategory_ts = data.groupby(['Date', 'SubCategoryName']).agg(column_aggregations).reset_index()
subcategory_ts['Level'] = 'SubCategoryName'
subcategory_ts['Segment'] = subcategory_ts['SubCategoryName']

print(f"Overall time series shape: {overall_ts.shape}")
print(f"Customer segment time series shape: {segment_ts.shape}")
print(f"SubCategory time series shape: {subcategory_ts.shape}")

# Display summary statistics
print("\n=== LEVEL SUMMARY ===")
print(f"Overall total quantity range: {overall_ts['Total_Quantity'].min():,.0f} - {overall_ts['Total_Quantity'].max():,.0f}")
print(f"Customer segments: {segment_ts['CustomerSegment'].unique()}")
print(f"SubCategories: {subcategory_ts['SubCategoryName'].unique()}")

overall_ts.head()


# In[5]:


# Visualize the hierarchical time series
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Overall Total Quantity', 'Customer Segments', 'SubCategories'),
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]]
)

# Plot Overall
fig.add_trace(
    go.Scatter(x=overall_ts['Date'], y=overall_ts['Total_Quantity'],
              mode='lines+markers', name='Overall Total',
              line=dict(color='darkblue', width=3)),
    row=1, col=1
)

# Plot Customer Segments
colors = px.colors.qualitative.Set1
for i, segment in enumerate(segment_ts['CustomerSegment'].unique()):
    segment_data = segment_ts[segment_ts['CustomerSegment'] == segment]
    fig.add_trace(
        go.Scatter(x=segment_data['Date'], y=segment_data['Total_Quantity'],
                  mode='lines+markers', name=segment,
                  line=dict(color=colors[i % len(colors)])),
        row=2, col=1
    )

# Plot SubCategories
colors2 = px.colors.qualitative.Set2
for i, subcat in enumerate(subcategory_ts['SubCategoryName'].unique()):
    subcat_data = subcategory_ts[subcategory_ts['SubCategoryName'] == subcat]
    fig.add_trace(
        go.Scatter(x=subcat_data['Date'], y=subcat_data['Total_Quantity'],
                  mode='lines+markers', name=subcat,
                  line=dict(color=colors2[i % len(colors2)])),
        row=3, col=1
    )

fig.update_layout(
    height=900,
    title_text="Hierarchical Time Series: Sales Quantity Analysis",
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
)

fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Total Quantity", row=1, col=1)
fig.update_yaxes(title_text="Total Quantity", row=2, col=1)
fig.update_yaxes(title_text="Total Quantity", row=3, col=1)

fig.show()

# Statistical summary
print("\n=== STATISTICAL SUMMARY ===")
print("Overall Statistics:")
print(overall_ts['Total_Quantity'].describe())

print("\nCustomer Segment Statistics:")
print(segment_ts.groupby('CustomerSegment')['Total_Quantity'].describe())

print("\nSubCategory Statistics:")
print(subcategory_ts.groupby('SubCategoryName')['Total_Quantity'].describe())


# # Hierarchical Forecasting Implementation
# 
# Functions defining the forecasting

# ## Forecasting Models
# 
# We'll implement multiple forecasting approaches:
# 1. **ARIMA** - Auto-regressive Integrated Moving Average
# 2. **SARIMA** - Seasonal ARIMA with economic indicators
# 3. **Exponential Smoothing** - Holt-Winters method
# 4. **Ensemble** - Weighted combination of methods

# In[6]:


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
                'lower Total_Quantity': forecast * 0.9,
                'upper Total_Quantity': forecast * 1.1
            })
            return forecast, conf_int, float('inf')

def forecast_sarima(series, exog=None, steps=12, order=(1,1,1), seasonal_order=(1,1,1,12)):
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
            'lower Total_Quantity': forecast - 1.96 * std_resid,
            'upper Total_Quantity': forecast + 1.96 * std_resid
        })
        
        return forecast, conf_int, fitted_model.aic
    except:
        # Fallback to ARIMA
        return forecast_arima(series, steps)

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
            weights = [1/3, 1/3, 1/3]

    print(f"Model weights - ARIMA: {weights[0]:.3f}, SARIMA: {weights[1]:.3f}, EXP: {weights[2]:.3f}")
    
    ensemble = sum(f * w for f, w in zip(forecasts, weights))
    return ensemble

print("Forecasting functions defined successfully!")


# ## Hierarchical Forecasting Functions

# In[7]:


def forecast_hierarchical_level(data, level_column, level_name, exog_vars, forecast_steps, forecast_dates):
    """
    Generic function to forecast at any hierarchical level (segments or subcategories)
    
    Parameters:
    - data: DataFrame with the time series data for the level
    - level_column: Column name that contains the grouping variable (e.g., 'CustomerSegment', 'SubCategoryName')
    - level_name: Name for the level (e.g., 'CustomerSegment', 'SubCategoryName')
    - exog_vars: List of exogenous variables to use in forecasting
    - forecast_steps: Number of steps to forecast
    - forecast_dates: Date range for forecasts
    
    Returns:
    - DataFrame with forecasts for all groups in the level
    """
    print(f"\n=== {level_name.upper()} FORECASTING ===")
    
    forecasts_list = []
    
    for group in data[level_column].unique():
        print(f"\nForecasting for {level_name.lower()}: {group}")
        
        # Filter data for this group
        group_data = data[data[level_column] == group].set_index('Date')
        group_series = group_data['Total_Quantity']
        group_exog = group_data[exog_vars]
        
        if len(group_series) < 3:  # Need minimum data points
            print(f"  Insufficient data for {group}, using naive forecast")
            group_ensemble = pd.Series([group_series.mean()] * forecast_steps)
        else:
            # Generate forecasts
            arima_forecast, _, arima_aic = forecast_arima(group_series, forecast_steps)
            sarima_forecast, _, sarima_aic = forecast_sarima(group_series, group_exog, forecast_steps)
            exp_forecast, _, exp_aic = forecast_exponential_smoothing(group_series, forecast_steps)
            
            # Create ensemble
            aics = [arima_aic, sarima_aic, exp_aic]
            group_ensemble = ensemble_forecast(
                [arima_forecast, sarima_forecast, exp_forecast],
                aics
            )
        
        # Store forecast
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Ensemble': group_ensemble.values,
            'Level': level_name,
            'Segment': group,
            level_column: group
        })
        
        forecasts_list.append(forecast_df)
        print(f"  Forecast range: {group_ensemble.min():,.0f} - {group_ensemble.max():,.0f}")
    
    # Combine all forecasts
    combined_forecasts = pd.concat(forecasts_list, ignore_index=True)
    
    print(f"\nTotal {level_name.lower()} forecasts generated: {len(forecasts_list)}")
    print(f"{level_name} forecast total range: {combined_forecasts['Ensemble'].min():,.0f} - {combined_forecasts['Ensemble'].max():,.0f}")
    
    return combined_forecasts


# In[8]:


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
    overall_series = data.set_index('Date')['Total_Quantity']
    overall_exog = data.set_index('Date')[exog_vars]
    
    # Generate forecasts using different methods
    print("Generating ARIMA forecast...")
    overall_arima_forecast, overall_arima_conf, overall_arima_aic = forecast_arima(overall_series, forecast_steps)
    
    print("Generating SARIMA forecast...")
    overall_sarima_forecast, overall_sarima_conf, overall_sarima_aic = forecast_sarima(overall_series, overall_exog, forecast_steps)
    
    print("Generating Exponential Smoothing forecast...")
    overall_exp_forecast, overall_exp_conf, overall_exp_aic = forecast_exponential_smoothing(overall_series, forecast_steps)
    
    # Create ensemble forecast
    aics = [overall_arima_aic, overall_sarima_aic, overall_exp_aic]    
    overall_ensemble_forecast = ensemble_forecast(
        [overall_arima_forecast, overall_sarima_forecast, overall_exp_forecast], 
        aics
    )
    
    # Store overall forecasts
    overall_forecasts = pd.DataFrame({
        'Date': forecast_dates,
        'ARIMA': overall_arima_forecast.values,
        'SARIMA': overall_sarima_forecast.values,
        'ExpSmoothing': overall_exp_forecast.values,
        'Ensemble': overall_ensemble_forecast.values,
        'Level': 'Overall',
        'Segment': 'Total'
    })
    
    print(f"Overall forecast range: {overall_ensemble_forecast.min():,.0f} - {overall_ensemble_forecast.max():,.0f}")
    
    return overall_forecasts


# In[9]:


def hierarchical_reconciliation(overall_forecasts, segment_forecasts, subcategory_forecasts, 
                               segment_ts, subcategory_ts, forecast_dates):
    """
    Perform top-down hierarchical reconciliation to ensure forecast consistency
    
    Parameters:
    - overall_forecasts: DataFrame with overall level forecasts
    - segment_forecasts: DataFrame with customer segment forecasts
    - subcategory_forecasts: DataFrame with subcategory forecasts
    - segment_ts: Historical segment time series data
    - subcategory_ts: Historical subcategory time series data
    - forecast_dates: Date range for forecasts
    
    Returns:
    - Tuple of (reconciled_segment_forecasts, reconciled_subcategory_forecasts)
    """
    print("=== HIERARCHICAL RECONCILIATION ===")
    
    # Check consistency before reconciliation
    print("\nBefore Reconciliation:")
    for date in forecast_dates[:3]:  # Check first 3 dates
        overall_val = overall_forecasts[overall_forecasts['Date'] == date]['Ensemble'].iloc[0]
        segment_sum = segment_forecasts[segment_forecasts['Date'] == date]['Ensemble'].sum()
        subcat_sum = subcategory_forecasts[subcategory_forecasts['Date'] == date]['Ensemble'].sum()
        
        print(f"  {date.strftime('%Y-%m')}: Overall={overall_val:,.0f}, Segments Sum={segment_sum:,.0f}, SubCats Sum={subcat_sum:,.0f}")
    
    # Calculate historical proportions for reconciliation
    print("\nCalculating historical proportions...")
    
    # Customer segment proportions
    segment_props = {}
    for segment in segment_ts['CustomerSegment'].unique():
        segment_total = segment_ts[segment_ts['CustomerSegment'] == segment]['Total_Quantity'].sum()
        overall_total = segment_ts['Total_Quantity'].sum()
        segment_props[segment] = segment_total / overall_total
    
    # SubCategory proportions
    subcat_props = {}
    for subcat in subcategory_ts['SubCategoryName'].unique():
        subcat_total = subcategory_ts[subcategory_ts['SubCategoryName'] == subcat]['Total_Quantity'].sum()
        overall_total = subcategory_ts['Total_Quantity'].sum()
        subcat_props[subcat] = subcat_total / overall_total
    
    print(f"Customer Segment Proportions: {segment_props}")
    print(f"SubCategory Proportions: {subcat_props}")
    
    # Apply top-down reconciliation
    print("\nApplying top-down reconciliation...")
    
    # Reconcile segment forecasts
    segment_forecasts_reconciled = segment_forecasts.copy()
    for idx, row in segment_forecasts_reconciled.iterrows():
        overall_val = overall_forecasts[overall_forecasts['Date'] == row['Date']]['Ensemble'].iloc[0]
        segment_forecasts_reconciled.loc[idx, 'Ensemble_Reconciled'] = overall_val * segment_props[row['CustomerSegment']]
    
    # Reconcile subcategory forecasts
    subcategory_forecasts_reconciled = subcategory_forecasts.copy()
    for idx, row in subcategory_forecasts_reconciled.iterrows():
        overall_val = overall_forecasts[overall_forecasts['Date'] == row['Date']]['Ensemble'].iloc[0]
        subcategory_forecasts_reconciled.loc[idx, 'Ensemble_Reconciled'] = overall_val * subcat_props[row['SubCategoryName']]
    
    # Verify reconciliation
    print("\nAfter Reconciliation:")
    for date in forecast_dates[:3]:
        overall_val = overall_forecasts[overall_forecasts['Date'] == date]['Ensemble'].iloc[0]
        segment_sum = segment_forecasts_reconciled[segment_forecasts_reconciled['Date'] == date]['Ensemble_Reconciled'].sum()
        subcat_sum = subcategory_forecasts_reconciled[subcategory_forecasts_reconciled['Date'] == date]['Ensemble_Reconciled'].sum()
        
        print(f"  {date.strftime('%Y-%m')}: Overall={overall_val:,.0f}, Segments Sum={segment_sum:,.0f}, SubCats Sum={subcat_sum:,.0f}")
    
    print("\nReconciliation completed!")
    
    return segment_forecasts_reconciled, subcategory_forecasts_reconciled


# # Running Hierarchical Forecasting

# In[10]:


# Set forecasting parameters
FORECAST_STEPS = 12  # 12 months ahead
START_DATE = overall_ts['Date'].max() + pd.DateOffset(months=1)
FORECAST_DATES = pd.date_range(start=START_DATE, periods=FORECAST_STEPS, freq='MS')

# Generate overall forecasts using the reusable function
overall_forecasts = forecast_overall(
    data=overall_ts,
    exog_vars=exog_vars,
    forecast_steps=FORECAST_STEPS,
    forecast_dates=FORECAST_DATES
)

overall_forecasts.head()


# In[11]:


# Generate customer segment forecasts using the generic function
segment_forecasts = forecast_hierarchical_level(
    data=segment_ts, 
    level_column='CustomerSegment', 
    level_name='CustomerSegment',
    exog_vars=exog_vars,
    forecast_steps=FORECAST_STEPS,
    forecast_dates=FORECAST_DATES
)


# In[12]:


# Generate subcategory forecasts using the generic function
subcategory_forecasts = forecast_hierarchical_level(
    data=subcategory_ts, 
    level_column='SubCategoryName', 
    level_name='SubCategoryName',
    exog_vars=exog_vars,
    forecast_steps=FORECAST_STEPS,
    forecast_dates=FORECAST_DATES
)


# ## Hierarchical Consistency & Reconciliation
# 
# We need to ensure that the sum of forecasts at lower levels equals the forecast at higher levels. This is called hierarchical reconciliation.

# In[13]:


# Apply hierarchical reconciliation using the reusable function
segment_forecasts_reconciled, subcategory_forecasts_reconciled = hierarchical_reconciliation(
    overall_forecasts=overall_forecasts,
    segment_forecasts=segment_forecasts,
    subcategory_forecasts=subcategory_forecasts,
    segment_ts=segment_ts,
    subcategory_ts=subcategory_ts,
    forecast_dates=FORECAST_DATES
)


# ## Forecast Visualization & Results
# 
# Let's visualize the hierarchical forecasts and compare them with historical data.

# In[14]:


# Create comprehensive forecast visualization
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Overall Forecast', 'Customer Segment Forecasts', 'SubCategory Forecasts'),
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]]
)

# Overall forecast plot
fig.add_trace(
    go.Scatter(x=overall_ts['Date'], y=overall_ts['Total_Quantity'],
              mode='lines+markers', name='Historical Total',
              line=dict(color='darkblue', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=overall_forecasts['Date'], y=overall_forecasts['Ensemble'],
              mode='lines+markers', name='Overall Forecast',
              line=dict(color='red', width=3, dash='dash')),
    row=1, col=1
)

# Customer segment forecasts
segment_colors = px.colors.qualitative.Set1
for i, segment in enumerate(segment_ts['CustomerSegment'].unique()):
    # Historical data
    segment_hist = segment_ts[segment_ts['CustomerSegment'] == segment]
    fig.add_trace(
        go.Scatter(x=segment_hist['Date'], y=segment_hist['Total_Quantity'],
                  mode='lines', name=f'{segment} (Historical)',
                  line=dict(color=segment_colors[i % len(segment_colors)], width=1),
                  showlegend=False),
        row=2, col=1
    )
    
    # Forecast data
    segment_forecast = segment_forecasts_reconciled[segment_forecasts_reconciled['CustomerSegment'] == segment]
    fig.add_trace(
        go.Scatter(x=segment_forecast['Date'], y=segment_forecast['Ensemble_Reconciled'],
                  mode='lines+markers', name=f'{segment} Forecast',
                  line=dict(color=segment_colors[i % len(segment_colors)], width=2, dash='dash')),
        row=2, col=1
    )

# SubCategory forecasts
subcat_colors = px.colors.qualitative.Set2
for i, subcat in enumerate(subcategory_ts['SubCategoryName'].unique()):
    # Historical data
    subcat_hist = subcategory_ts[subcategory_ts['SubCategoryName'] == subcat]
    fig.add_trace(
        go.Scatter(x=subcat_hist['Date'], y=subcat_hist['Total_Quantity'],
                  mode='lines', name=f'{subcat} (Historical)',
                  line=dict(color=subcat_colors[i % len(subcat_colors)], width=1),
                  showlegend=False),
        row=3, col=1
    )
    
    # Forecast data
    subcat_forecast = subcategory_forecasts_reconciled[subcategory_forecasts_reconciled['SubCategoryName'] == subcat]
    fig.add_trace(
        go.Scatter(x=subcat_forecast['Date'], y=subcat_forecast['Ensemble_Reconciled'],
                  mode='lines+markers', name=f'{subcat} Forecast',
                  line=dict(color=subcat_colors[i % len(subcat_colors)], width=2, dash='dash')),
        row=3, col=1
    )

fig.update_layout(
    height=1000,
    title_text="Hierarchical Sales Forecasting: Historical vs Predicted",
    showlegend=True,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
)

fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Total Quantity")

fig.show()

# Print forecast summary
print("\n=== FORECAST SUMMARY ===")
print(f"Forecast Period: {FORECAST_DATES[0].strftime('%Y-%m')} to {FORECAST_DATES[-1].strftime('%Y-%m')}")
print(f"\nOverall Forecast Summary:")
print(f"  Mean Monthly Forecast: {overall_forecasts['Ensemble'].mean():,.0f}")
print(f"  Total 12-Month Forecast: {overall_forecasts['Ensemble'].sum():,.0f}")
print(f"  Min-Max Range: {overall_forecasts['Ensemble'].min():,.0f} - {overall_forecasts['Ensemble'].max():,.0f}")

print(f"\nCustomer Segment Forecast Summary:")
for segment in segment_forecasts_reconciled['CustomerSegment'].unique():
    segment_data = segment_forecasts_reconciled[segment_forecasts_reconciled['CustomerSegment'] == segment]
    total_forecast = segment_data['Ensemble_Reconciled'].sum()
    print(f"  {segment}: {total_forecast:,.0f} (12-month total)")

print(f"\nSubCategory Forecast Summary:")
for subcat in subcategory_forecasts_reconciled['SubCategoryName'].unique():
    subcat_data = subcategory_forecasts_reconciled[subcategory_forecasts_reconciled['SubCategoryName'] == subcat]
    total_forecast = subcat_data['Ensemble_Reconciled'].sum()
    print(f"  {subcat}: {total_forecast:,.0f} (12-month total)")


# ## Export Results
# 
# Let's save the hierarchical forecasts to CSV files for further analysis and reporting.

# In[15]:


# Export forecast results
print("=== EXPORTING FORECAST RESULTS ===")

# Overall forecast export
overall_export = overall_forecasts[['Date', 'ARIMA', 'SARIMA', 'ExpSmoothing', 'Ensemble', 'Level', 'Segment']].copy()
overall_export.to_csv(data_location + 'hierarchical_forecast_overall.csv', index=False)
print(f"Overall forecasts exported to: hierarchical_forecast_overall.csv")

# Customer segment forecasts export
segment_export = segment_forecasts_reconciled[['Date', 'CustomerSegment', 'Ensemble', 'Ensemble_Reconciled', 'Level']].copy()
segment_export.to_csv(data_location + 'hierarchical_forecast_segments.csv', index=False)
print(f"Segment forecasts exported to: hierarchical_forecast_segments.csv")

# SubCategory forecasts export
subcategory_export = subcategory_forecasts_reconciled[['Date', 'SubCategoryName', 'Ensemble', 'Ensemble_Reconciled', 'Level']].copy()
subcategory_export.to_csv(data_location + 'hierarchical_forecast_subcategories.csv', index=False)
print(f"SubCategory forecasts exported to: hierarchical_forecast_subcategories.csv")

# Combined hierarchical forecasts
combined_forecasts = []

# Add overall forecasts
for _, row in overall_forecasts.iterrows():
    combined_forecasts.append({
        'Date': row['Date'],
        'Level': 'Overall',
        'Segment': 'Total',
        'Forecast_Value': row['Ensemble'],
        'Forecast_Method': 'Ensemble'
    })

# Add segment forecasts
for _, row in segment_forecasts_reconciled.iterrows():
    combined_forecasts.append({
        'Date': row['Date'],
        'Level': 'CustomerSegment',
        'Segment': row['CustomerSegment'],
        'Forecast_Value': row['Ensemble_Reconciled'],
        'Forecast_Method': 'Ensemble_Reconciled'
    })

# Add subcategory forecasts
for _, row in subcategory_forecasts_reconciled.iterrows():
    combined_forecasts.append({
        'Date': row['Date'],
        'Level': 'SubCategoryName',
        'Segment': row['SubCategoryName'],
        'Forecast_Value': row['Ensemble_Reconciled'],
        'Forecast_Method': 'Ensemble_Reconciled'
    })

combined_df = pd.DataFrame(combined_forecasts)
combined_df.to_csv(data_location + 'hierarchical_forecasts_all_levels.csv', index=False)
print(f"Combined hierarchical forecasts exported to: hierarchical_forecasts_all_levels.csv")

# Summary report
summary_report = []
summary_report.append({
    'Level': 'Overall',
    'Segment': 'Total',
    'Total_12Month_Forecast': overall_forecasts['Ensemble'].sum(),
    'Average_Monthly_Forecast': overall_forecasts['Ensemble'].mean(),
    'Min_Monthly_Forecast': overall_forecasts['Ensemble'].min(),
    'Max_Monthly_Forecast': overall_forecasts['Ensemble'].max()
})

for segment in segment_forecasts_reconciled['CustomerSegment'].unique():
    segment_data = segment_forecasts_reconciled[segment_forecasts_reconciled['CustomerSegment'] == segment]
    summary_report.append({
        'Level': 'CustomerSegment',
        'Segment': segment,
        'Total_12Month_Forecast': segment_data['Ensemble_Reconciled'].sum(),
        'Average_Monthly_Forecast': segment_data['Ensemble_Reconciled'].mean(),
        'Min_Monthly_Forecast': segment_data['Ensemble_Reconciled'].min(),
        'Max_Monthly_Forecast': segment_data['Ensemble_Reconciled'].max()
    })

for subcat in subcategory_forecasts_reconciled['SubCategoryName'].unique():
    subcat_data = subcategory_forecasts_reconciled[subcategory_forecasts_reconciled['SubCategoryName'] == subcat]
    summary_report.append({
        'Level': 'SubCategoryName',
        'Segment': subcat,
        'Total_12Month_Forecast': subcat_data['Ensemble_Reconciled'].sum(),
        'Average_Monthly_Forecast': subcat_data['Ensemble_Reconciled'].mean(),
        'Min_Monthly_Forecast': subcat_data['Ensemble_Reconciled'].min(),
        'Max_Monthly_Forecast': subcat_data['Ensemble_Reconciled'].max()
    })

summary_df = pd.DataFrame(summary_report)
summary_df.to_csv(data_location + 'forecast_metrics_summary.csv', index=False)
print(f"Forecast summary report exported to: forecast_metrics_summary.csv")

print(f"\n=== EXPORT COMPLETE ===")
print(f"Total files exported: 4")
print(f"Export timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Display final summary
print(f"\n=== FINAL SUMMARY ===")
print(summary_df.to_string(index=False))

