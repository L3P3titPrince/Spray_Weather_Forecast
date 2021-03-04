from class_31_hyperparameters import HyperParamters
from class_33_eda import EDA

# using for timestap convert
import pandas as pd
import numpy as np
from scipy import stats
# recording running time in each function
from time import time



class PreProcess(HyperParamters):
    """:arg
    Actually, not only production data need to be clean.

    tz_convert(): used to convert string to timestamp format
    """

    def __init__(self):
        """:arg
        Inheriret from HyperParamters
        """
        HyperParamters.__init__(self)


    def tz_convert(self, series):
        """:arg
        Args:
        ------
        serise:DataFrame.series
            each element from DataFrame columns

        Returns:
        -----------
        series_est:timestamp

        """
        series_est = pd.Timestamp(series, unit='s', tz='America/New_York')

        return series_est

    def round_to_hour(self,timestamp):
        """
        We notice df_product['StartDat'] is string type not DataFrame.timestamp.
        And weather data are seperated by hours, so we need transform string time data round to nearest hour

        Args:
        --------
        timestamp
        """
        # due to excel convert to csv, all format are lost and original timestamp has transformed to 'string'
        # so we add a duplicate process to convert 'string' to 'timestamp'
        timestamp = pd.Timestamp(timestamp)
        # because of summer time and winter time rules, some UTC date can't conver to EST
        try:
            date_hour = timestamp.round(freq='H').tz_localize(tz='America/New_York', nonexistent='shift_forward')
        except:
            print('This data cant convert correctly {}'.format(timestamp.round(freq='H')))
            date_hour = timestamp.round(freq='H').tz_localize(tz='America/New_York',
                                                              ambiguous='NaT',
                                                              nonexistent='shift_forward')
            print('This date replaced by {}'.format('NaT'))

        return date_hour

    def drop_na(self, df, list_col):
        """
        Args:
        ------
        df:DataFrame

        Returns:
        --------
        df_clean:DataFrame
        """
        df_clean = df.dropna(subset=list_col)
        print("After dropping na, {} rows has decreased to {} rows".format(df.shape[0], df_clean.shape[0]))

        return df_clean


    def del_outlier(self, column, abs=False):
        """:arg
        We use z-score to delete outliers in our data.
        When ads=True, it will delete small outliers. When ads=False, it only delete greater than 110%
        """
        if abs:
            z_score = np.abs(stats.zscore(df_product['YieldPercentage'], nan_policy='omit'))
        z_score = np.abs(stats.zscore(df_product['YieldPercentage'], nan_policy='omit'))
        # set a z-score threshold, any greater than this value will be eliminate
        z_threshold = 3
        # filter that ooutlier rows in dataframe
        index_outlier = np.where(z_score > z_threshold)
        # print out result
        print('We have {} data points are outliers'.format(len(index_outlier[0])))
        # drop these rows by index
        df_11 = df_product.drop(index_outlier[0])



    def clean_data(self, df_product, df_nj_weather, df_pa_weather, list_col_missing_product):
        """:arg
        Delete columns we believe didn't use in the future.

        Args:
        --------
        df_product:DataFrame

        Returns:
        --------
        df_product:DataFrame
            droped missing value, add a new column['dt_est'] for merge purpose
        df_nj_weather:DataFrame
            cleaned DataFrame table

        """
        print("*" * 50, "Start clean_data()", "*" * 50)
        start_time = time()


        #***************Drop non-realted columns**********************
        # *************Drop by columns**************
        # ['Bulk Density'] have 50% missing data
        df_product = df_product.drop(self.DROP_COL, axis=1)
        print("These columns have been dropped {}".format(self.DROP_COL))
        
        class_eda = EDA()
        list_col_missing_product = class_eda.missing_plot(df_product)

        #**************Drop by rows**********************
        # we get the name of columns that has missing value from eda part
        df_product = self.drop_na(df_product, list_col_missing_product)

        #*****************Drop by outliers****************


        # some weather data might also need clearn


        # convert weather data ['dt'] (unix time) to Eastern Stardard Time(EST)
        # pass an argument(series) to function tz_convert()
        df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(self.tz_convert)
        # another way to use apply()
        # df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(lambda x: pd.TimeStamp(x, unit='s', tz='America/New_York'))
        df_pa_weather['dt_est'] = df_pa_weather['dt'].apply(self.tz_convert)

        # for merge purpose, make sure two column have same name
        df_product['dt_est'] = df_product['StartDate'].apply(self.round_to_hour)



        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End clean_data() with {} second".format(cost_time), "*" * 40, end='\n\n')

        return df_product, df_nj_weather, df_pa_weather
