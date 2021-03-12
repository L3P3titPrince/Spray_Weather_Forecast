from class_31_hyperparameters import HyperParamters

import pandas as pd
import numpy as np
# for time sum()
import datetime

class MergeData(HyperParamters):
    """:arg
    For now, we have differenty hypothesis to view our data
    We will use self.MERGE_WAY to choose different hypothesis

    row_proliferate will be the core components. Every other merge function will need call self.df_multi variable first


    1.self.MERGE_WAY:one
        without further process, just using dt_est to merge
    2.self.MERGE_WAY:

    """
    def __init__(self):
        HyperParamters.__init__(self)

    def convert_timedelta(self, element):
        """:arg
        """
        time_delta = datetime.timedelta(hours = element)
        return time_delta


    def row_proliferate(self, df_product):
        """:arg
        Because weather data is split by hour, so we need transform each records into every hour records

        Args:
        -----
        df_product:DataFrame

        """
        # first create a new column ['hours'] from ['DryingTime_Hrs'] round to bigger integer
        df_product['hours'] = np.ceil(df_product['DryingTime_Hrs']).astype(int)
        # Second, repeat each records/rows by ['hours'] times
        # which means you will have how many running hours, you will have how many rows for each column
        self.df_multi = pd.DataFrame(np.repeat(df_product.values, df_product['hours'], axis=0),
                                      columns = df_product.columns)
        # Thrid, adjust ['hours'] to ascending rank model, for instance, [3,3,3] to [0,1,2]
        # this is used for next step add to ['dt_est'] time
        # groupby each old record, for instance, for first records, groupby will be a three row table
        # for second records, groupby['StartData'] will be a nine row table
        # Then, extract ['hours'] column, for each pd.Series, apply a fomulation
        # addition, in here, ['hours'] will not be a single row, instead, it will be a three element Series
        # for this three/nine series, we cacualte their cumulation sum cumsum() [3,3,3] will be [3,6,9]
        # for instance [9.9.9.9...9] will be [9,18,27,,,,81]
        # and we divide by ['hours'] count, which will be [3,9,...], last minute 1 to get a column for timestamp add
        self.df_multi['hour_add'] = self.df_multi.groupby(['StartDate'])['hours'].apply(lambda x:x.cumsum()/x.count())-1
        # our weather segmentation is one hour
        # one_hour = datetime.timedelta(hours = 1)
        self.df_multi['hour_add'] = self.df_multi['hour_add'].apply(self.convert_timedelta)
        self.df_multi['dt_est'] = self.df_multi['dt_est'] + self.df_multi['hour_add']

        return self.df_multi


    def merge_location(self, df_product, df_nj_weather, df_pa_weather):
        """:arg
        split by location, just merge all together by ['dt_est']
        """
        # Identify location by ['Batchnumber']
        df_nj_pro = df_product[df_product['BatchNumber'].str.contains('NJ', regex=False)]
        # them merge them into a new DataFrame
        df_nj = pd.merge(df_nj_pro, df_nj_weather, how='left', on=['dt_est'])
        df_pa_pro = df_product[df_product['BatchNumber'].str.contains('PA', regex=False)]
        df_pa = pd.merge(df_pa_pro, df_pa_weather, how='left', on=['dt_est'])

        return df_nj, df_pa



