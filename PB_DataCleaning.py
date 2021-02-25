#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#Import production data set
production_df = pd.read_excel('Initial_Data.xlsx')
production_df.head()

#Create new column that splits batch number into PA or NJ
batch_number = production_df.BatchNumber.str[:2]
production_df['Batch Location'] = np.where(batch_number == 'PA', 'Bethlehem, PA', 'Middlesex, NJ')

#Work with the StartDate column
production_df['Year'] = pd.DatetimeIndex(production_df['StartDate']).year
production_df['Month'] = pd.DatetimeIndex(production_df['StartDate']).month
production_df['Day'] = pd.DatetimeIndex(production_df['StartDate']).day
production_df['Time'] = pd.DatetimeIndex(production_df['StartDate'].time

#Calculate difference in DRY QUANTITY and create new column
production_df['DryQty_Difference'] = (production_df['ScheduledDryQty'] - production_df['ActualDryQty']).round(2)

#Drop Bulk Density & Moisture Target *Execute only if we dont need these columns*
### production_df.drop(['Bulk Density', 'Moisture Target'], axis = 1)

#Drop rows in 'Yield Percentage' that contain values > 110 *confirm threshold of outlier*
#production_df = production_df[production_df.YieldPercentage < 110]

##### Import weather data #####
middlesex_weather = pd.read_excel('Initial_Data.xlsx', sheet_name = 'MiddlesexWeather')
bethlehem_weather = pd.read_excel('Initial_Data.xlsx', sheet_name = 'BethlehemWeather')