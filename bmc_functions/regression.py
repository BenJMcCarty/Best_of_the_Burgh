'''Name: Regression

Description: Functions created to assist with the creation and evaluation of regression models.

By Ben McCarty (bmccarty505@gmail.com)'''

def diagnose_model(model, figsize=(10,5)):
    """ ---
    
    Argument:
        * model: provide the linear regression model for diagnostics
    
    Keyword Argument:
        * figsize: default (10,5); can increase/decrease for larger/smaller
    ---
    
    * Display the summary details of the provided model
    * Create two scatter plots to test assumptions of linearity
        * Predictions: verifying homoscedasticity (no cone-shapes)
        * Residuals: confirming normal distribution of residuals
    ---
    
    """

    print(model.summary())
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    axes[0].scatter(model.predict(), model.resid)
    axes[0].axhline()
    axes[0].set_xlabel('Model Predictions')
    axes[0].set_ylabel('Model Residuals')
    axes[0].set_title('Testing for Homoscedasticity')

    sms.graphics.qqplot(data=model.resid, fit=True, line = "45", ax=axes[1])
    
    plt.tight_layout()
    
    return


def create_model(data, cont, cat, target):
    """Creates a linear regression model using Statsmodels OLS and 
    evaluates assumptions of linearity by plotting residuals for homoscedasticity
    and a Q-Q plot for normality.

    Args:
        data (pd.DataFrame): Source DataFrame
        cont (list): List of strings indicating which column names to treat as continuous data
        cat (list): List of strings indicating which column names to treat as categorical data

    Returns:
        model: Statsmodels OLS Linear Regression model
    """    

    cont_features = '+'.join(cont)

    cat_features = '+'.join([f'C({x})' for x in cat])

    f = f'{target}~+{cont_features}+{cat_features}'

    print(f)

    model = smf.ols(formula=f, data=data).fit()
   
    diagnose_model(model)
    
    return model


def plot_param_coef(model, kind = 'barh', figsize = (10,5)):
    """Plotting a figure to visualize parameter coefficients

    Args:
        model (Statsmodels OLS model object): linear regression model details to plot
        kind (str, optional): Plot type. Defaults to 'barh'.
        figsize (tuple, optional): Figure size. Defaults to (10,5).
    """
 
    ## Getting coefficients as a Series
    params = model.params[1:]
    params.sort_values(inplace=True)

    plt.figure(figsize=figsize) # Used if large number of params
    ax = params.plot(kind=kind)
    ax.axvline()
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Features')
    ax.set_title('Comparing Feature Coefficients')
    
    plt.tight_layout()
    
    return

def plot_p_values(model, kind = 'barh', figsize = (10,5), alpha = .05):
    """Plots a figure to visualize parameter p-values exceeding stated alpha.

    Args:
        model (Statsmodels OLS model object): Model details to plot.
        kind (str, optional): Plot type. Defaults to 'barh'.
        figsize (tuple, optional): Figure size. Defaults to (10,5).
        alpha (float, optional): Significance level (p-value). Defaults to .05.
    """   

    pv = model.pvalues[1:]
    pv_high = pv[pv > alpha]
    pv_low = pv[pv <= alpha]
    pv_high.sort_values(ascending=False, inplace=True)
    
    if len(pv_high) > 0:
        plt.figure(figsize=figsize) # Used if large number of params
        ax = pv_high.plot(kind=kind)
        ax = pv_low.plot(kind=kind)
        ax.axvline()
        plt.suptitle(f'P-Values')
        
    if len(pv_low) > 0:
        plt.figure(figsize=figsize) # Used if large number of params
        ax = pv_low.plot(kind=kind)
        ax.axvline()
        plt.suptitle(f'P-Values Below {alpha}')        

    ## Not used; keeping just in case        
    # else:
        # print(f'There are no p-values above {alpha}.')
        
    plt.tight_layout()
    
    return

def review_model(model):
    """Combines earlier functions into one all-purpose function for reviewing
    model performance.

    Args:
        model (Statsmodels OLS model object): Model details to plot.
    """    
    
    diagnose_model(model)
    
    plot_param_coef(model)
    
    plot_p_values(model)
    
    return

def eval_perf_train(model, X_train=None, y_train=None):
    """Evaluates the performance of a model on training data

    Metrics:
    Mean Absolute Error (MAE)
    Mean Squared Error(MSE)
    Root Mean Squared Error (RMSE)
    R^2

    Args:
        model (fit & trasformed model): model created via Statsmodels or SKLearn
        X_train (2D array): X_train data from train/test split
        y_train (1D array): y_train data from train/test split
    """

    # if X_train != None and y_train != None:

    y_hat_train = model.predict(X_train)
    
    train_mae = metrics.mean_absolute_error(y_train, y_hat_train)
    train_mse = metrics.mean_squared_error(y_train, y_hat_train)
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_hat_train))
    train_r = metrics.r2_score(y_train, y_hat_train)

    print('Evaluating Performance on Training Data:\n')
    print(f'Train Mean Absolute Error: {train_mae:,.2f}')
    print(f'Train Mean Squared Error:  {train_mse:,.2f}\n')
    print(f'Train Root Mean Squared Error: {train_rmse:,.2f}')
    print(f'Train R-Square Value: {round(train_r,2)}')

    # if X_test != None and y_test != None:

        # y_hat_test = model.predict(X_test)

        # test_mae = metrics.mean_absolute_error(y_test, y_hat_test)
        # test_mse = metrics.mean_squared_error(y_test, y_hat_test)
        # test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat_test))
        # test_r = metrics.r2_score(y_test, y_hat_test)

        # print('Evaluating Performance on Testing Data:\n')
        # print(f'Test Mean Absolute Error: {test_mae:,.2f}')
        # print(f'Test Mean Squared Error:  {test_mse:,.2f}\n')
        # print(f'Test Root Mean Squared Error: {test_rmse:,.2f}')
        # print(f'Test R-Square Value: {round(test_r,2)}')

def eval_perf_test(model, X_test, y_test):
    """Evaluate the performance of a given model on the testing data

    Args:
        model (transformed model): model created via Statsmodels or SKLearn
        X_test (2D array): X_test data from train/test split
        y_test (1D array): y_train data from train/test split
    """

    y_hat_test = model.predict(X_test)

    test_mae = metrics.mean_absolute_error(y_test, y_hat_test)
    test_mse = metrics.mean_squared_error(y_test, y_hat_test)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat_test))
    test_r = metrics.r2_score(y_test, y_hat_test)

    print('Evaluating Performance on Testing Data:\n')
    print(f'Test Mean Absolute Error: {test_mae:,.2f}')
    print(f'Test Mean Squared Error:  {test_mse:,.2f}\n')
    print(f'Test Root Mean Squared Error: {test_rmse:,.2f}')
    print(f'Test R-Square Value: {round(test_r,2)}')

def plot_coefs(data, x_label, y_label, title, kind = 'barh', style = 'seaborn-darkgrid',
               figsize = (10, 8)):
    """Generates plots to visualize model coefficients.

    Args:
        data (pd.Series): Model coefficients as a Pandas Series
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Visualization title
        kind (str, optional): [description]. Defaults to 'barh'.
        style (str, optional): [description]. Defaults to 'seaborn-darkgrid'.
        figsize (tuple, optional): [description]. Defaults to (10, 8).

    Returns:
        Matplotlib.pyplt ax: generated visualization
    """

    with plt.style.context(style):
    
        ax = data.plot(kind=kind, figsize = figsize, rot=45)
              
        if kind == 'barh':
            
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
            ax.set_yticklabels(ax.get_yticklabels(), ha='right')
            ax.axvline(color='k')
            ax.set(xlabel = x_label, ylabel = y_label, title = title)
            
        else:
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
            ax.set_xticklabels(ax.get_xticklabels(), ha='right')
            ax.axhline(color='k')
            ax.set(xlabel = x_label, ylabel = y_label, title = title)

    return ax

def eval_perf_total(model, X_train, y_train, X_test, y_test):
    """Evaluates the performance of a model on training data

    Metrics:
    Mean Absolute Error (MAE)
    Mean Squared Error(MSE)
    Root Mean Squared Error (RMSE)
    R^2

    Args:
        model (fit & trasformed model): model created via Statsmodels or SKLearn
        X_train (2D array): X_train data from train/test split
        y_train (1D array): y_train data from train/test split
    """

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    train_mae = metrics.mean_absolute_error(y_train, y_hat_train)
    train_mse = metrics.mean_squared_error(y_train, y_hat_train)
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_hat_train))
    train_r = metrics.r2_score(y_train, y_hat_train)

    print('Evaluating Performance on Training Data:\n')
    print(f'    Train Mean Absolute Error: {train_mae:,.2f}')
    print(f'    Train Mean Squared Error:  {train_mse:,.2f}\n')
    print(f'Train Root Mean Squared Error: {train_rmse:,.2f}')
    print(f'Train R-Square Value: {round(train_r,2)}')

    print('\n'+'---'*25+'\n')

    test_mae = metrics.mean_absolute_error(y_test, y_hat_test)
    test_mse = metrics.mean_squared_error(y_test, y_hat_test)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat_test))
    test_r = metrics.r2_score(y_test, y_hat_test)

    print('Evaluating Performance on Testing Data:\n')
    print(f'    Test Mean Absolute Error: {test_mae:,.2f}')
    print(f'    Test Mean Squared Error:  {test_mse:,.2f}\n')
    print(f'Test Root Mean Squared Error: {test_rmse:,.2f}')
    print(f'Test R-Square Value: {round(test_r,2)}')

def get_model_coefs(model, index):

    model_coefs = pd.Series(model['regressor'].coef_, index=index)
    model_coefs['intercept'] = model['regressor'].intercept_
    
    return model_coefs
