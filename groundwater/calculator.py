import numpy as np
import pandas as pd

from numpy import ndarray

def date_to_int(date):
    mm, dd, year = date.split("/")
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


def get_dict(arr:ndarray,key_idx:int):
    '''
    Given an n*k array A and key_idx K, return a map from key to (n-1)*k' ndarray,\\ 
    where k' is the num of rows in A with the specified key.\\
    The K-th column of A is the list of keys, which is used to construct the map.
    '''
    assert(len(arr.shape)==2)
    _,ncols=arr.shape
    if(key_idx<0): key_idx=ncols+key_idx
    key_list=arr[:,key_idx]
    data_arr=arr[:,[i for i in range(ncols) if i!=key_idx]]
    keys=list(set(key_list))
    dic={k:None for k in keys}
    for k in dic:
        dic[k]=data_arr[key_list==k]
    return dic

def get_colocations(dic:dict,colo_type:list,thres:float,curr_type=None,colo_arr=None):
    '''
    Given dic of data obtained by get_dict(), colocation type, and threshold,\\
    return the n*d ndarray of colocations,\\
    where n is num of colocations, and d is the length of colo_type.\\
    curr_type of the colo_type for colo_arr, which can be used for recursion.\\
    Note that curr_type can only be None or a prefix of colo_type.\\
    Return None if no colocations are found.
    '''
    if curr_type==None: curr_type=list()
    assert(str(colo_type).startswith(str(curr_type)[:-1]))
    if len(colo_type)==len(curr_type): return colo_arr

    type1 = colo_type[len(curr_type)] # new type of object
    objs_t1=np.arange(len(dic[type1])) # indices of type1 objects
    pos_t1=dic[type1] # get positions of type1 objects

    # base case
    if len(curr_type)==0: # add all objects of type1 within the window
        curr_type.append(type1)
        colo_arr=np.array(objs_t1).reshape(-1,1)
        return get_colocations(dic,colo_type,thres,curr_type,colo_arr)
    
    # general case
    mask=np.ones((len(colo_arr),len(objs_t1))) # mask[i,j] checks whether colo_arr[i] and objs_t1[j] form a colocation
    for i,type2 in enumerate(curr_type):
        objs_t2=colo_arr[:,i]
        pos_t2=dic[type2][objs_t2] # get positions of type2 objects
        # pairwise distances of shape = (# of type2) * (# of type1)
        dists=np.sqrt(-2*pos_t2.dot(pos_t1.T)+np.sum(pos_t2**2,axis=1,keepdims=True)+np.sum(pos_t1**2,axis=1))
        mask*=(dists<=thres) # check whether type1 and type2 has distance less than thres
    curr_type.append(type1)
    colo_arr=np.array(
        [list(colo_arr[idx_colo])+[objs_t1[idx_t1]] 
        for idx_colo,row in enumerate(mask) 
            for idx_t1,val in enumerate(row) 
                if val==True]) # generate new colo_arr for curr_type
    if len(colo_arr)==0: return None # if empty, return None
    return get_colocations(dic,colo_type,thres,curr_type,colo_arr)

def get_part_index(dic:dict,colo_type:list,part_type,colo_arr:np.ndarray):
    '''
    Given dic of data obtained by get_dict(), colocation type, and participation type A,\\
    colo_arr computed by get_colocations,\\
    return the participation index of A for the colocation type.\\
    '''
    assert(len(colo_arr.shape)==2)
    idx=colo_type.index(part_type)
    return len(set(colo_arr[:,idx].tolist()))/len(dic[part_type])

def count_objs(df,date,type):
    filt = df[(df['Date']==date)&(df['type']==type)]
    return filt.shape[0]
    

# write a csv file out
# input parameter: a dataframe df, a list of type called types, year you want to study (default=2002)
def vectorize(df,types,year=2002,dates = []):
    df['year'] =  df['Date']/10000
    df['year'] = df['year'].astype('int')
    df = df[df['year']==year]
    # make sure all the date in different df are the same!!
    if len(dates) ==0:
        dates = df['Date'].unique()
    
    to_return = []
    for date in dates:
        # the return data structure is a list of T x 3 pandas dataframe
        to_select = df[df['Date']==date]
        type = to_select['type']
        x = to_select['LongitudeDD']
        y = to_select['LatitudeDD']
        lists = list(zip(type,x,y))
        cur_df = pd.DataFrame(lists,columns = ['type','x','y'])
        to_return.append(cur_df)
    return to_return

def add_vectorization(vec1,vec2):
    vec_sum = []
    for i in range(len(vec1)):
        df_to_add = vec1[i].append(vec2[i],ignore_index=True)
        vec_sum.append(df_to_add)
    return vec_sum

