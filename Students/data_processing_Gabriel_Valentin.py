
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
def remove_Outliers(data, col, treshold = 1.5):
    if type(col) is str :
        Q3 = data[col].quantile(0.75)
        Q1 = data[col].quantile(0.25)
        IQR = Q3 - Q1
        lower_range = Q1 - treshold*IQR
        upper_range = Q3 + treshold*IQR
        outlier_free_list = [x for x in data[col] if ((x > lower_range) & (x < upper_range))]
        filtered_data = data.loc[data[col].isin(outlier_free_list)]
        return  filtered_data
    elif len(col)>1:
        filtered_data = remove_Outliers(data, col[0])
        for i in col[1:]:
            filtered_data = remove_Outliers(filtered_data, i)       
        return filtered_data

def remove_Z_outliers(data, col, treshold = 3):
    if type(col) is str :
        mean = data[col].mean()
        std = data[col].std()
        outlier_free_list = [x for x in data[col] if (np.abs(x-mean)/std <treshold)]
        filtered_data = data.loc[data[col].isin(outlier_free_list)]
        return  filtered_data
    else:
        filtered_data = remove_Z_outliers(data, col[0], treshold)
        for i in col[1:]:
            filtered_data = remove_Z_outliers(filtered_data, i, treshold)       
        return filtered_data



def upper_cut(data, col, treshold = 0.2):
    if type(col) is str :
        res = data.drop(data.loc[data[col] > treshold].index)
        return res
    elif len(col)>1:
        filtered_data = upper_cut(data, col[0], treshold)
        for i in col[1:]:
            filtered_data = upper_cut(filtered_data, i, treshold)     
        return filtered_data

def under_cut(data, col, treshold = 0.2):
    if type(col) is str :
        res = data.drop(data.loc[data[col] < treshold].index)
        return res
    elif len(col)>1:
        filtered_data = upper_cut(data, col[0], treshold)
        for i in col[1:]:
            filtered_data = upper_cut(filtered_data, i, treshold)     
        return filtered_data
    
def processing(T_measures = "../sampling_points/Point034/point034_T_measures.csv", 
               P_measures ="../sampling_points/Point034/point034_P_measures.csv", 
               config = "../configuration/pressure_sensors/P508.csv"):
    with open(T_measures, 'r') as file: 
        all_lines = file.readlines()
    file.close()
    if all_lines[0][0] != '"' or all_lines[0].strip()[-1] != '"':
        print(all_lines[0][-1], all_lines[0][0])
        all_lines.pop(0)   
    with open(T_measures, 'w') as file: 
        for line in all_lines: 
            file.write(line)
    capteur_riviere = pd.read_csv(P_measures)
    capteur_ZH = pd.read_csv(T_measures)
    etalonage_capteur_riv = pd.read_csv(config)
    np.random.seed(0)
    capteur_ZH.head()
    capteur_riviere.rename(columns = {'Unnamed: 1' : 'dates', 'Unnamed: 2' : 'tension_V', 'Unnamed: 3' : 'temperature_stream_C'}, inplace = True)
    capteur_ZH.rename(columns = {'Titre de tracé : T520' : '#', 'Date Heure, GMT+01:00': 'dates' , 'Temp., °C (LGR S/N: 10117166, SEN S/N: 10117166, LBL: Température)':'temperature_depth_1_C' , 'Temp., °C (LGR S/N: 10117166, SEN S/N: 10117166, LBL: Température).1':'temperature_depth_2_C' , 'Temp., °C (LGR S/N: 10117166, SEN S/N: 10117166, LBL: Température).2':'temperature_depth_3_C' , 'Temp., °C (LGR S/N: 10117166, SEN S/N: 10117166, LBL: Température).3':'temperature_depth_4_C'}, inplace = True)
    print(capteur_riviere.dtypes)
    capteur_riviere['tension_V']=pd.to_numeric(capteur_riviere['tension_V'], errors  ='coerce')
    capteur_riviere['temperature_stream_C']=pd.to_numeric(capteur_riviere['temperature_stream_C'], errors  ='coerce')
    print(capteur_riviere.dtypes)
  
    etalonage_capteur_riv.dtypes
    dUH = pd.to_numeric(etalonage_capteur_riv.loc[3,'P508'])
    dUT = pd.to_numeric(etalonage_capteur_riv.loc[4,'P508'])
    intercept = pd.to_numeric(etalonage_capteur_riv.loc[2,'P508'])
    capteur_riviere['charge_M'] = ((capteur_riviere['tension_V'])-(capteur_riviere['temperature_stream_C'])*dUT-intercept)/(dUH)
    capteur_riviere['dates'] = pd.to_datetime(capteur_riviere['dates'] , infer_datetime_format = True, errors = 'coerce')
    capteur_ZH['dates'] = pd.to_datetime(capteur_ZH['dates'] , infer_datetime_format = True, errors = 'coerce')
    print(capteur_riviere.info())
    print(capteur_ZH.info())
    capteur_ZH.describe()
    capteur_riviere.describe()
    capteur_riviere['temperature_stream_C'].hist()
    print('La charge minimale est', capteur_riviere['charge_M'].min(), 'et la charge maximale est', capteur_riviere['charge_M'].max())
    print('La temperature minimale est', capteur_riviere['temperature_stream_C'].min(), 'et la temperature maximale est', capteur_riviere['temperature_stream_C'].max())
    print("Raw data")
    capteur_riviere['charge_M'].hist()
    capteur_riviere.boxplot(column = ['charge_M'])
    capteur_riviere.boxplot(column = ['temperature_stream_C'])
    #capteur_ZH.boxplot(column = ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'])
    capteur_ZH.hist(column = ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'])
    capteur_ZH.plot.scatter(x = 'temperature_depth_1_C' , y = 'temperature_depth_2_C')
    capteur_ZH.plot.scatter(x = 'temperature_depth_2_C' , y = 'temperature_depth_3_C')
    def id(data, col, treshold):
        return data
    mode = input("Remove temp outlier method (None, up_cut, un_cut, Z_score, IQ) :" )
    methods = {'up_cut' : upper_cut, 'un_cut' : under_cut, 'Z_score' : remove_Z_outliers, 'IQ' : remove_Outliers, 'None' : id}
    treshold = input("Threshold (press enter for default):")
    if treshold == '':
        filtered_temp = methods[mode](capteur_ZH, col =  ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'])
    else:
        treshold = int(treshold)
        filtered_temp = methods[mode](capteur_ZH, col =  ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'], treshold = treshold)
    
    #filtered_temp.boxplot(column = ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'])
    filtered_temp.hist(column = ['temperature_depth_1_C','temperature_depth_2_C','temperature_depth_3_C','temperature_depth_4_C'])
    filtered_temp.plot.scatter(x = 'temperature_depth_1_C' , y = 'temperature_depth_2_C')
    filtered_temp.plot.scatter(x = 'temperature_depth_2_C' , y = 'temperature_depth_3_C')
    
    mode = input("Remove Pressure outlier method (None, up_cut, un_cut, Z_score, IQ) :" )

    treshold = input("Threshold (press enter for default):")
    if treshold == '':
        filtered_p = methods[mode](capteur_riviere, col =  ['temperature_stream_C','charge_M'])
    else:
        treshold = int(treshold)
        filtered_p = methods[mode](capteur_riviere, col =  ['temperature_stream_C','charge_M'], treshold = treshold)
    filtered_p['charge_M'].hist()
    filtered_p.boxplot(column = ['charge_M'])
    filtered_p.boxplot(column = ['temperature_stream_C'])
    return filtered_p,filtered_temp

p, temp = processing()
temp.boxplot()
