#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#Import data set
production_df = pd.read_excel('Initial_Data.xlsx')
production_df.head()

#df shape: (9970,16)
production_df.shape

#columns
production_df.columns

#Batch Number: retrieve first two characters of the string
batch_number = production_df.BatchNumber.str[:2]

#Add new column based on batch number to determine location
production_df['Batch Location'] = np.where(batch_number == 'PA', 'Bethlehem, PA', 'Middlesex, NJ')

#Work with the StartDate column
production_df['Year'] = pd.DatetimeIndex(production_df['StartDate']).year
production_df['Month'] = pd.DatetimeIndex(production_df['StartDate']).month
production_df['Day'] = pd.DatetimeIndex(production_df['StartDate']).day
production_df['Time'] = pd.DatetimeIndex(production_df['StartDate']).time

### Unique value counts of columns ###

 #different type of dryers
production_df['Dryer'].value_counts()
production_df.Dryer.value_counts().plot(kind = 'bar', title = 'Dryer Types')

#Flow
production_df.Flow.value_counts().plot(kind = 'pie', autopct = '%1.1f%%',  title = 'Flow Types', colormap = 'vlag' )

#Hygroscopicity
production_df['Hygroscopicity'].value_counts().plot(kind = 'bar', title = 'Hygroscopicity Types', colormap = 'vlag')

#Year 
production_df['Year'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%')

#Month
production_df['Month'].value_counts().plot(kind = 'bar', title = 'Months')

#Calculate difference in DRY QUANTITY and create new column
production_df['DryQty_Difference'] = (production_df['ScheduledDryQty'] - production_df['ActualDryQty']).round(2)


##### Import weather data #####
middlesex_weather = pd.read_excel('Initial_Data.xlsx', sheet_name = 'MiddlesexWeather')
bethlehem_weather = pd.read_excel('Initial_Data.xlsx', sheet_name = 'BethlehemWeather')
