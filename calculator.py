import numpy as np
import pandas as pd

def date_to_int(date):
    dd, mm, year = date.split("/")
    if len(dd)==1: dd = '0'+dd
    if len(mm)==1: mm = '0'+mm
    int_date = int(year+mm+dd)
    return int_date

# divide the data into groups by concentration
# parameters: 
# data -> input data (only 4 columns: date, latitude, longitude, concentration)
# value_partition -> partition the data according to this list. 
# ##Format: [a1,a2,a3,a4] <-> [a1,a2), [a2,a3), [a3,a4) ##
# names -> a list of string containing the name of each partition
# value_partition has length n+1 while names has length n
def pre_process(data,value_partition,names):
    data = data[['StateWellNumber','Date','LatitudeDD','LongitudeDD','ParameterValue']]
    # eliminate null vales
    data = data.dropna()
    # if there are more than 1 record on the same day, take mean
    data = data.groupby(["StateWellNumber","Date"]).mean()
    data = data.reset_index()
    # change the date object into int with length 8 ##Format: yearmmdd##
    data['Date'] = data['Date'].apply(date_to_int)
    data['type'] = np.nan

    for i in range(len(value_partition)):
        if i==len(value_partition)-1: break
        lwb, upb = value_partition[i], value_partition[i+1]
        data.loc[(data['ParameterValue']<upb) & (data['ParameterValue']>=lwb),'type'] = names[i]
    # return the dataframe with Date in ascending order
    data  = data.sort_values("Date")
    return data

# given a list of dataframe of different chemicals, select the dates all of them have records
def select_timestamp():
    pass

