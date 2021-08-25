'''Name: Time Series Modeling

Description: Functions created to assist with the creation and evaluation of time series models.

By Ben McCarty (bmccarty505@gmail.com)'''

### --------------- Importing Dependencies --------------- ###

from sys import displayhook
import pmdarima as pmd
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

### --------------- Functions --------------- ###

# Creating train/test split for time series modeling
def ts_split(timeseries_df, threshold):
    """Creates train/test split for time series modeling.

    Args:
        timeseries_df (DataFrame): DataFrame or Series to be modeled
        threshold (float): threshold (as decimal percent) for splitting data

    Returns:
        train, test: Initial DataFrame/Series split into sub-Series for modeling
    """

    tts_cutoff = round(timeseries_df.shape[0]*threshold)
    train = timeseries_df.iloc[:tts_cutoff]
    test = timeseries_df.iloc[tts_cutoff:]

    ## Plot
    fig, ax = plt.subplots()
    ax = train.plot(label='Train')
    ax = test.plot(label='Test')
    ax.legend()
    ax.set_xlabel('Years')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Train/Test Split for Zipcode {timeseries_df.name}')
    ax.axvline(train.index[-1], linestyle=":")
    plt.show();
    
    return train, test

## Display model results
def model_results(ts_model):
    """Displays a fitted model's summary and plot diagnostics.

    Args:
        ts_model (model): fitted model for evaluation
    """    
    display(ts_model.summary())
    ts_model.plot_diagnostics();
    plt.tight_layout()

## Generate best model parameters via auto_arima
def auto_arima_model(timeseries_dataset,m=12,start_p=0,max_p=3
                        ,start_q=0,max_q=3,d=n_d,start_P=0,
                        start_Q=0, max_P=3, max_Q = 3, D = n_D):
    
    auto_arima_model = pmd.auto_arima(timeseries_dataset, start_p = start_p,
                                max_p = max_p,start_q = start_q,
                                max_q = max_q, d = d ,m = m,
                                start_P = start_P, start_Q = start_Q,
                                max_P = max_P, max_Q = max_Q, D = D)

    return auto_arima_model

## Use auto_arima to determine best parameters
## Then create new SARIMA model via Statsmodels with selected parameters
def create_best_model(timeseries_dataset,m=12,start_p=0,max_p=3
                        ,start_q=0,max_q=3,d=n_d,start_P=0,
                        start_Q=0, max_P=3, max_Q = 3, D = n_D):
    """Calculates best model parameters via auto-arima,
     then fits a new SARIMAX model for results.

    Args:
        timeseries_dataset (Series/DataFrame): dataset for modeling
        m (int, optional): The number of periods in each season (for seasonal differencing). Defaults to 12.
        start_p (int, optional): Starting value for "p". Defaults to 0.
        max_p (int, optional): Max value for "p". Defaults to 3.
        start_q (int, optional): Starting value for "pq. Defaults to 0.
        max_q (int, optional): Max value for "q". Defaults to 3.
        d ([type], optional): The order of first-differencing. Defaults to n_d.
        start_P (int, optional): Starting value for "P". Defaults to 0.
        start_Q (int, optional): Starting value for "Q". Defaults to 0.
        max_P (int, optional): Max value for "P". Defaults to 3.
        max_Q (int, optional): Max value for "Q". Defaults to 3.
        D ([type], optional): The order of the seasonal differencing. Defaults to n_D.

    Returns:
        auto_model, best_model: auto_arima-generated model with best parameters,
                                SARIMAX model using best parameters.
    """    

    auto_model = auto_arima_model(timeseries_dataset, start_p = start_p,
                                max_p = max_p,start_q = start_q,
                                max_q = max_q, d = d ,m = m,
                                start_P = start_P, start_Q = start_Q,
                                max_P = max_P, max_Q = max_Q, D = D)
      
    best_model = tsa.SARIMAX(timeseries_dataset,order=auto_model.order,
                             seasonal_order = auto_model.seasonal_order,
                             enforce_invertibility=False).fit()
    
    display(auto_model.summary())
    display(model_results(best_model))
    
    return auto_model, best_model




### --------------- Workflow --------------- ###

def ts_modeling_workflow(dataframe, zipcode, threshold = .85, m= 12, n_yrs_past=5, n_yrs_future=2):
    """Functionalizes total time series modeling workflow 
    starting with time series dataset through final forecasted data and ROI.

    Args:
        dataframe (DataFrame): Original dataframe from which to select series
        zipcode (string): Series name to model and forecast.
        threshold (float, optional): Threshold to determine train/test split. Defaults to .85.
        m (int, optional): The number of periods in each season (for seasonal differencing). Defaults to 12.
        n_yrs_past (int, optional): Number of past years for visualizations. Defaults to 5.
        n_yrs_future (int, optional): Number of future years for visualizations. Defaults to 2.
    """

    ## Select values for the selected zipcode
    zipcode_val = dataframe[zipcode].copy()
    zipcode_val

    ## Split dataset
    train, test = ts_split(zipcode_val, threshold)

    ## Determine d, D values for SARIMA model
    n_d = ndiffs(train)
    n_D = nsdiffs(train, m=m)

    ## Generating auto_arima model and SARIMAX model
    ## (based on best parameters from auto_arima model)
    auto_model, best_model = create_best_model(train, n_d=n_d, n_D=n_D)
  
    ## Using get_forecast to generate forecasted data
    forecast = best_model.get_forecast(steps=len(test))

    ## Saving confidence intervals and predicted mean for future
    forecast_df = forecast.conf_int()
    forecast_df.columns = ['Lower CI','Upper CI']
    forecast_df['Forecast'] = forecast.predicted_mean
    forecast_df.head(5)

    fig,ax = plt.subplots(figsize=(13,6))

    ## Plotting training, testing datasets
    last_n_lags=12*n_yrs_past
    train.iloc[-last_n_lags:].plot(label='Training Data')
    test.plot(label='Test Data')

    ## Plotting forecasted data and confidence intervals
    forecast_df['Forecast'].plot(ax=ax,label='Forecast')
    ax.fill_between(forecast_df.index,forecast_df['Lower CI'],
                    forecast_df['Upper CI'],color='b',alpha=0.4)
    ax.set(xlabel='Time')
    ax.set(ylabel='Sale Price ($)')
    ax.set_title('Original Data and Forecasted Data')
    ax.legend();

    ## Fitting best model using whole dataset
    best_model_overall = tsa.SARIMAX(zipcode_val,order=auto_model.order,
                            seasonal_order = auto_model.seasonal_order,
                            enforce_invertibility=False).fit()

    ## Showing results
    display(model_results(best_model_overall))

    ## Using get_forecast to generate forecasted data
    forecast = best_model_overall.get_forecast(steps=12*n_yrs_future)

    ## Saving confidence intervals and predicted mean for future
    forecast_df = forecast.conf_int()
    forecast_df.columns = ['Lower CI','Upper CI']
    forecast_df['Forecast'] = forecast.predicted_mean
    display(forecast_df)

    ## Plotting original data and forecasted results
    fig,ax = plt.subplots(figsize=(13,6))

    ## Plotting original data
    zipcode_val.plot(label='Original Data')

    ## Plotting forecasted data and confidence intervals
    forecast_df['Forecast'].plot(ax=ax,label='Forecast')
    ax.fill_between(forecast_df.index,forecast_df['Lower CI'],
                    forecast_df['Upper CI'],color='b',alpha=0.4)
    ax.set(xlabel='Time')
    ax.set(ylabel='Sale Price ($)')
    ax.set_title('Original and Forecasted Data')
    ax.legend();

    ## Calculating investment cost and ROI
    investment_cost = forecast_df.iloc[0,2]
    roi_df = (forecast_df - investment_cost)/investment_cost*100
    display(roi_df)

    roi_final = roi_df.iloc[-1]
    roi_final.name = zipcode_val.name.astype('str')
    display(roi_final)