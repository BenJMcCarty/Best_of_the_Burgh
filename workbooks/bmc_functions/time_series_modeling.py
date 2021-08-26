'''Name: Time Series Modeling

Description: Functions created to assist with the creation and evaluation of time series models.

By Ben McCarty (bmccarty505@gmail.com)'''

### ----- Importing Dependencies ----- ###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.tsa.api as tsa

import pmdarima as pmd
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

### ----------------------------------- Functions ----------------------------------- ###

## --------------- Stationary Methods --------------- ##

def adf_test(ts, p = .05):
    zipdf_results = tsa.stattools.adfuller(ts)
    
    index_label = [f'Results: {ts.name}']
    labels = ['Test Stat','P-Value','Number of Lags Used','Number of Obs. Used',
            'Critical Thresholds', 'AIC Value']
    results_dict  = dict(zip(labels,zipdf_results))

    ## Saving results to a dictionary and adding T/F indicating stationarity
    results_dict[f'p < {p}'] = results_dict['P-Value'] < p
    results_dict['Stationary'] = results_dict[f'p < {p}']

    ## Creating DataFrame from dictionary
    if isinstance(index_label,str):
        index_label = [index_label]
    results_dict = pd.DataFrame(results_dict,index=index_label)
    results_dict = results_dict[['Test Stat','P-Value','Number of Lags Used',
                                 'Number of Obs. Used','P-Value',f'p < {p}',
                                 'Stationary']]
    
    return results_dict

def remove_trends(timeseries, method, window = 4):
    if method == 'diff':
        results = timeseries.diff().dropna()
    elif method == 'log':
        results = np.log(timeseries)
    elif method == 'rolling' or method == 'rolling mean':
        results = timeseries - timeseries.rolling(window = window).mean()
        results.dropna(inplace=True)
    elif method == 'ewm' or method == 'EWM':
        results = timeseries-timeseries.ewm(4).mean()
        results.dropna(inplace=True)
    
    print("|","---"*7,f"{method.title()} Effect on Zipcode {timeseries.name}",
      "-----"*6,"|",'\n\n')
    print("|","---",f"Zipcode {timeseries.name}","---","|","\n")
    print(results)
    print('\n\n',"|","----"*5,f"ADF Results for Zipcode {timeseries.name}",
          "-----"*6,"|")
    display(adf_test(results))

    print('\n\n','|',"---"*8,f"Visualizing {method.title()} Effect","----"*8,
          "|")
    fig, ax = plt.subplots()
    ax = results.plot(label=f'{timeseries.name}')
    ax.legend()
    ax.set_xlabel('Years')
    ax.set_ylabel('Price ($)')
    
    if method != 'ewm' and method != 'EWM':
        ax.set_title(f'{method.title()} Effect on Zipcode {timeseries.name}')
    else:
        ax.set_title(f'{method.capitalize()} Effect on Zipcode \
                                                        {timeseries.name}')
    plt.show()
    
    return results

def plot_acf_pacf(data, figsize=(12,6), lags=52, suptitle=None, sup_x = .53, sup_y = 1):
    """Plot pacf and acf using statsmodels
    
    Adapted from: https://github.com/flatiron-school/Online-DS-FT-022221-\
    Cohort-Notes/blob/master/Phase_4/topic_38_time_series_models/topic_38-\
    time_series_models_v3_SG.ipynb"""
    
    fig,axes=plt.subplots(nrows=2, figsize = figsize)
    
    tsa.graphics.plot_acf(data,ax=axes[0],lags=lags)
    tsa.graphics.plot_pacf(data,ax=axes[1],lags=lags)
    
    ## Add gridlines and y-labels
    [ax.grid(axis='both',which='both') for ax in axes]
    [ax.set_ylabel('Corr. Strength') for ax in axes]
    
    if suptitle is not None:
        fig.suptitle(suptitle,x = sup_x, y=sup_y,fontweight='bold',fontsize=15)
        
    fig.tight_layout()

    return fig,axes

### --------------- Modeling --------------- ###

# Creating train/test split for time series modeling
def ts_split(dataframe, threshold=.85, show_vis=False):
    """Creates train/test split for time series modeling.

    Args:
        timeseries_df (DataFrame): DataFrame or Series to be modeled
        threshold (float): Threshold (as decimal percent) for splitting data. Defaults to .85.
        show_vis (boolean): Whether to show a visualization of the split data. Detaults to False.

    Returns:
        train, test: Initial DataFrame/Series split into sub-Series for modeling
    """

    tts_cutoff = round(dataframe.shape[0]*threshold)
    train = dataframe.iloc[:tts_cutoff]
    test = dataframe.iloc[tts_cutoff:]

    if show_vis == True:
        fig,ax = plt.subplots()
        ax = train.plot(label='Training Data')
        test.plot(ax=ax, label='Testing Data')
        ax.set_xlabel('Years')
        ax.set_ylabel('Sale Price ($)')
        ax.set_title(f'Zipcode {dataframe.name}: Train/Test Split')
        ax.axvline(train.index[-1], linestyle=":", label=f'Split Point: {train.index[-1].year}'+'-'+f'{train.index[-1].month}')
        ax.legend()
        plt.show()

        return train, test, fig, ax
    else:
        return train, test

## Display model results
def model_performance(ts_model, show_vis = False):
    """Displays a fitted model's summary and plot diagnostics.

    Args:
        ts_model (model): fitted model for evaluation
    """    
    
    fig = ts_model.plot_diagnostics()
    plt.close()
    
    if show_vis is True:
        fig = ts_model.plot_diagnostics()
        plt.tight_layout()

    return ts_model.summary(), fig

## Generate best model parameters via auto_arima
def auto_arima_model(timeseries_dataset, m = 12, start_p=0,max_p=5,
                        start_q=0,max_q=5,start_P=0,
                        start_Q=0, max_P=5, max_Q = 5):
    
    """Fits an auto_arima model to a given timeseries dataset.

    Args:
        timeseries_dataset (Series/DataFrame): dataset for modeling
        m (int): The number of periods in each season (for seasonal differencing).
        start_p (int, optional): Starting value for "p". Defaults to 0.
        max_p (int, optional): Max value for "p". Defaults to 3.
        start_q (int, optional): Starting value for "pq. Defaults to 0.
        max_q (int, optional): Max value for "q". Defaults to 3.
        start_P (int, optional): Starting value for "P". Defaults to 0.
        start_Q (int, optional): Starting value for "Q". Defaults to 0.
        max_P (int, optional): Max value for "P". Defaults to 3.
        max_Q (int, optional): Max value for "Q". Defaults to 3.

    Returns:
        auto_arima_model: Fitted auto_arima model for use in SARIMAX modeling.
    """    

    # ## Determine d, D values for SARIMA model
    n_d = ndiffs(timeseries_dataset)
    n_D = nsdiffs(timeseries_dataset, m=m)

    auto_arima_model = pmd.auto_arima(timeseries_dataset,m = m,
                                start_p = start_p,max_p = max_p,
                                start_q = start_q, max_q = max_q,
                                start_P = start_P, max_P = max_P,
                                start_Q = start_Q, max_Q = max_Q,
                                d = n_d, D = n_D, error_action="ignore")

    return auto_arima_model

## Use auto_arima to determine best parameters
## Then create new SARIMA model via Statsmodels with selected parameters
def create_best_model(timeseries_dataset,m=12,start_p=0,max_p=5,
                        start_q=0,max_q=5,start_P=0,
                        start_Q=0, max_P=5, max_Q = 5, show_vis=False):

    """Calculates best model parameters via auto-arima,
     then fits a new SARIMAX model for results.

    Args:
        timeseries_dataset (Series/DataFrame): dataset for modeling
        m (int): The number of periods in each season (for seasonal differencing)
        start_p (int, optional): Starting value for "p". Defaults to 0.
        max_p (int, optional): Max value for "p". Defaults to 3.
        start_q (int, optional): Starting value for "pq. Defaults to 0.
        max_q (int, optional): Max value for "q". Defaults to 3.
        start_P (int, optional): Starting value for "P". Defaults to 0.
        start_Q (int, optional): Starting value for "Q". Defaults to 0.
        max_P (int, optional): Max value for "P". Defaults to 3.
        max_Q (int, optional): Max value for "Q". Defaults to 3.
        show_vis (boolean, optional): Whether to show the model summary and plot diagnostics. Defaults to False.

    Returns:
        auto_model, best_model: auto_arima-generated model with best parameters,
                                SARIMAX model using best parameters.
    """

    auto_model_best = auto_arima_model(timeseries_dataset,m = m,
                                 start_p = start_p,max_p = max_p,
                                 start_q = start_q,max_q = max_q,
                                 start_P = start_P, start_Q = start_Q,
                                 max_P = max_P, max_Q = max_Q)
      
    best_model = tsa.SARIMAX(timeseries_dataset,order=auto_model_best.order,
                             seasonal_order = auto_model_best.seasonal_order,
                             enforce_invertibility=False).fit()
    
    if show_vis is True:
        display(auto_model_best.summary())
        display(model_performance(best_model))
    
    return auto_model_best, best_model

## Using get_forecast to generate forecasted data
def forecast_and_ci(model, test_data):
    """Generate forecast for a given model

    Args:
        model: fitted SARIMAX model
        test_data (Series): Test data
    """    
    forecast = model.get_forecast(steps=len(test_data))
    forecast_df = forecast.conf_int()
    forecast_df.columns = ['Lower CI','Upper CI']
    forecast_df['Forecast'] = forecast.predicted_mean

    return forecast_df

## Plotting training, testing datasets
def plot_forecast_ttf(train, test, forecast_df, n_yrs_past=5, show_vis = False):
    fig,ax = plt.subplots()

    last_n_lags=12*n_yrs_past
    train.iloc[-last_n_lags:].plot(label='Training Data')
    test.plot(label='Test Data')

    ## Plotting forecasted data and confidence intervals
    forecast_df['Forecast'].plot(ax=ax,label='Forecast', color='g')
    ax.fill_between(forecast_df.index,forecast_df['Lower CI'],
                    forecast_df['Upper CI'],color='y',alpha=0.275)
    ax.set(xlabel='Years')
    ax.set(ylabel='Sale Price ($)')
    ax.set_title(f'Zipcode {train.name}: Validating Forecasted Data')
    ax.axvline(test.index[0], linestyle=":",
     label=f'Beginning of Forecast: {test.index[0].year}'+'-'+f'{test.index[0].month}',
      color='k')
    ax.legend(loc='upper left')
    
    if show_vis is True:
        plt.show()
    else:
        plt.close()

    return fig, ax

## Plotting training, testing datasets
def plot_forecast_final(zipcode_val, forecast_full, show_vis = False):
    ## Plotting original data and forecasted results
    fig,ax = plt.subplots()

    ## Plotting original data
    zipcode_val.plot(ax=ax, label='Original Data')

    ## Plotting forecasted data and confidence intervals
    forecast_full['Forecast'].plot(ax=ax,label='Forecast', color='g')
    ax.fill_between(forecast_full.index,forecast_full['Lower CI'],
                    forecast_full['Upper CI'],color='y',alpha=0.275)
    ax.set(xlabel='Years')
    ax.set(ylabel='Sale Price ($)')
    ax.set_title(f'Zipcode {zipcode_val.name}: Original Data and Forecast Data')
    ax.axvline(zipcode_val.index[-1], linestyle=":",
     label=f'Beginning of Forecast: {zipcode_val.index[-1].year}'+'-'+f'{zipcode_val.index[-1].month}',
      color='k')
    ax.legend(loc='upper left')
    
    if show_vis is True:
        plt.show()
    else:
        plt.close()

    return fig, ax


### --------------- Workflow --------------- ###

def ts_modeling_workflow(dataframe, zipcode, threshold = .85, m= 12, n_yrs_past=5, show_vis = False):
    """Functionalizes total time series modeling workflow 
    starting with time series dataset through final forecasted data and ROI.

    Args:
        dataframe (DataFrame): Original dataframe from which to select series
        zipcode (string): Series name to model and forecast.
        threshold (float, optional): Threshold to determine train/test split. Defaults to .85.
        m (int, optional): The number of periods in each season (for seasonal differencing). Defaults to 12.
        n_yrs_past (int, optional): Number of past years for visualizations. Defaults to 5.
    """

    ## Select values for the selected zipcode
    zipcode_val = dataframe[zipcode].copy()
    zipcode_val

    ## Split dataset
    if show_vis == True:
            train, test, split_vis, _ = ts_split(zipcode_val, threshold, show_vis = show_vis)
    else:
        train, test = ts_split(zipcode_val, threshold, show_vis = show_vis)

    ## Generating auto_arima model and SARIMAX model
    ## (based on best parameters from auto_arima model)
    auto_model_train, best_model_train = create_best_model(timeseries_dataset = train, m=m)
    
    ## Savind training model results
    summary_train, diag_train = model_performance(best_model_train)

    ## Generating dataframe to store forecast results
    forecast_train = forecast_and_ci(best_model_train, test)

    ## Plotting forecast results against train/test split
    training_frcst, _  = plot_forecast_ttf(train=train, test=test, forecast_df = forecast_train, n_yrs_past=n_yrs_past, show_vis=show_vis)

    ## Fitting best model using whole dataset
    best_model_full = tsa.SARIMAX(zipcode_val,order=auto_model_train.order,
                            seasonal_order = auto_model_train.seasonal_order,
                            enforce_invertibility=False).fit()

    
    summary_full, diag_full = model_performance(best_model_full)

    ## Using get_forecast to generate forecasted data
    forecast_full = forecast_and_ci(best_model_full, test)

    ## Plotting original data and forecast results
    final_frcst, _ = plot_forecast_final(zipcode_val, forecast_full, show_vis = show_vis)

    ## Calculating investment cost and ROI across dataframe
    investment_cost = forecast_full.iloc[0,2]
    roi_df = (forecast_full - investment_cost)/investment_cost*100
    
    ## Pulling ROI for final forecasted date
    roi_final = roi_df.iloc[-1]
    roi_final.name = zipcode_val.name.astype('str')
    
    if show_vis is True:
        return forecast_full, roi_final, split_vis, summary_train, diag_train, summary_full, diag_full, training_frcst, final_frcst
    else:
        plt.close()
        return forecast_full, roi_final, summary_train, diag_train, summary_full, diag_full, training_frcst, final_frcst

def make_dict(dataframe, zipcode, m=12, show_vis = True):

    zip_tsa_results = {}
    metrics = {}
    forecast_vis = {}
    
    if show_vis == False:
        forecast_full, roi_final, summary_train, diag_train, summary_full,diag_full, training_frcst, final_frcst = ts_modeling_workflow\
            (dataframe = dataframe, zipcode = zipcode, m=m, show_vis = show_vis)
    
    else:
        forecast_full, roi_final, split_vis, summary_train, diag_train, summary_full, diag_full, training_frcst, final_frcst = ts_modeling_workflow\
            (dataframe = dataframe, zipcode = zipcode, m=m, show_vis = show_vis)
    

    
    metrics['train'] = [summary_train, diag_train]
    metrics['full'] = [summary_full, diag_full] 
    forecast_vis['train'] = training_frcst
    forecast_vis['full'] = final_frcst
    if show_vis == True:
        forecast_vis['split'] = split_vis
    
    zip_tsa_results['forecasted_prices'] = forecast_full
    zip_tsa_results['roi'] = roi_final
    zip_tsa_results['model_metrics'] = metrics
    zip_tsa_results['model_visuals'] = forecast_vis
    
    return zip_tsa_results