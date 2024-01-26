import pandas as pd
import numpy as np 

# Routine to normalize between 0 and 1 (Following the atmospheric turbulence paper)

def norm_data(x):
    for ii in range(len(x)):
        x[ii] = (x[ii] - np.min(x))/(np.max(x) - np.min(x))
    return x
    
# Load csv files 

df_weather=pd.read_csv('Weather Data/input_weatherData.csv')
df_scintillometer=pd.read_csv('Weather Data/output_scintillometerData.csv')

# Get all rows where the indices match

weather_index = df_weather["Index"].to_numpy()

input_data = []
output_data = []

# In the spirit of the paper, we load the RH, Sol. Irradiance, Temperature, Pressure 

for ii,ind in enumerate(weather_index):
    if(df_scintillometer["Cn2"][ind] == "ERROR"  or df_weather["Temp (°C)"][ii] == "ERROR" or df_weather["Rel Hum (%)"][ii] == "ERROR" or df_weather["Stn Press (kPa)"][ii] == "ERROR"):
        print(f"Error detected at ii = {ii}, ind = {ind} ")
    else:
        input_data.append([eval(df_weather["Temp (°C)"][ii]), eval(df_weather["Rel Hum (%)"][ii]), eval(df_weather["Stn Press (kPa)"][ii])])
        output_data.append([np.log10(eval(df_scintillometer["Cn2"][ind])), df_scintillometer["Fried"][ind]])
        
input_data = np.array(input_data)
output_data = np.array(output_data)

shape_input = np.shape(input_data)
shape_output = np.shape(output_data)

# Normalize input & output data

for ii in range(shape_input[1]):
    input_data[:,ii] = norm_data(input_data[:,ii])

for ii in range(shape_output[1]):
    output_data[:,ii] = norm_data(output_data[:,ii])

