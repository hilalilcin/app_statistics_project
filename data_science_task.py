# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:20:36 2024

@author: hilal
"""
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import resource
import math



def number_of_users_predictions(df,latest_hours,next_hours):
    model = SARIMAX(df['users'].tail(latest_hours))# defining sarimax model 
    model_fit = model.fit()
    # Future Prediction
    pred_users = model_fit.predict(len(df["users"].tail(latest_hours)),len(df["users"].tail(latest_hours))+(next_hours-1))
    
    #Original Prediction Values(without rounding)
    print('Original Prediction Values')
    print(pred_users)
    
    # Rounded Prediction Values
    pred_users = pred_users.round()
    print('Rounded Prediction Values')
    print(pred_users)
    
    # Performance Metrics
    print('Performance Metrics')
    performance_evaluation = model_fit.predict(1,len(df["users"].tail(latest_hours)))
    mse = mean_squared_error(df["users"].tail(latest_hours),performance_evaluation)
    mae = mean_absolute_error(df["users"].tail(latest_hours),performance_evaluation)
    rmse = math.sqrt(mse)

    print('MSE={:.2f} '.format(mse))  
    print('RMSE={:.2f} '.format(rmse))
    print('MAE={:.2f} '.format(mae)) 
    print('***************************************')
    
    #Data Visualization 
    plt.figure(figsize=(20, 3))
    plt.plot(df['users'].tail(latest_hours), color='blue',label='User count')
    plt.plot(pred_users, '--bo',color='red', label='Prediction')
    plt.legend(loc='best')
    plt.show()
    
    return df


# for pipeline
def time_preprocessing(df):
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df 



if __name__ == "__main__":

    time_start = time.perf_counter()

    # reading csv
    df = pd.read_csv("app.csv",sep = ';')

   
    pipeline = df.pipe(time_preprocessing).pipe(number_of_users_predictions, latest_hours = 24, next_hours = 6).pipe(number_of_users_predictions,
    latest_hours = (24*5), next_hours = 24)

    time_execution = (time.perf_counter() - time_start)

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

    print("\nTotal duration and the memory usage of the overall system\n")
    print ("Total Duration: %5.1f secs / Memory Usage: %5.1f MByte" % (time_execution,memory_usage))

    

  
    
    
    

    

    