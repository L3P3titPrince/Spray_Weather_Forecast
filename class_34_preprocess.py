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

    def tz_y_m_d(self, ts):
        """:arg
        We use this function to convert Timestamp to year/quarter/month/day

        Args:
        ------
        ts:pd.Timestamp
            ['dt_est']

        Returns:
        -------
        year:int
        quarter:int
        month:int
        day:int
        """
        year = ts.year
        quarter = ts.quarter
        month = ts.month
        day = ts.day

        return year, quarter, month, day


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


    def z_outlier(self, df, str_col, z_threshold  = 3, abs=False):
        """
        We use z-score to delete outliers in our data. for example ['YieldPrecentage'] and ['Rate']
        When ads=True, it will delete smallest and biggest outliers.
        When ads=False, it will only delete outliers ['YieldPercentage']> 130%, which means remain [0,130]

        Args:
        ------
        df:DataFrame
            Any dataframe
        str_col:str
            the name of column you want to detect outliers and drop

        Returns:
        --------
        df_clean:DataFrame
            df has been dropped
        df_outlier:DataFrame
            The rows that has been detected as outliers
        """
        if abs:
            # if abs=True, we will absolute z-score and find the both side outliers
            z_score = np.abs(stats.zscore(df[str_col], nan_policy='omit'))
            print("We will drop too large and too small outliers")
        else:
            # if abs=False (default setting), we only delete the too large outlers
            z_score = stats.zscore(df[str_col], nan_policy='omit')
            print("we will only drop too large outliers")
        # set a z-score threshold, any greater than this value will be eliminate
        # z_threshold = 3
        # filter that ooutlier rows in dataframe
        # index_outlier = np.where(z_score > z_threshold, z_score)
        # get the index of these outliers
        index_outlier = df[z_score > z_threshold].index
        # restore these outliers in df_outlier for further review
        df_outlier = df.loc[index_outlier,:]
        print('We have {} data points are outliers, which between {} and {}'.format(len(index_outlier),
                                                                                    df_outlier[str_col].min(),
                                                                                    df_outlier[str_col].max()))
        # drop these rows by index
        df_clean = df.drop(axis=0, index=index_outlier)
        print("After drop outlier, the {} column remain range between {} and {}".format(str_col,
                                                                                        df_clean[str_col].min(),
                                                                                        df_clean[str_col].max()))
        return df_clean, df_outlier



    def iqr_outlier(self, df, str_col, iqr_threshold=3):
        """:arg
        Except z-socre, we also have also have 'Interquartile Range Method' to deal with non-Gaussian distribution

        Args:
        -----
        df:pd.DataFrame

        """
        print('Start using IQR to eliminate outliers.')
        # caculate interquartile range
        q75, q25 = np.percentile(df[str_col], 75), np.percentile(df[str_col], 25)
        # IQR can be used to identify outliers by defining limits on the sample values that are a factore k
        iqr = q75 - q25
        # of the IQR below the 25th percentile or above the 75th percentile. The commmon value of k is 1.5
        # A factor k of 3 or more can be used to identify values that are extrmel outliers
        cut_off = iqr * iqr_threshold
        # get the lower bound and upper bound
        lower_bound, upper_bound = q25-cut_off, q75+cut_off
        # so we can use these bounds to identify outliers and remaining data
        df_outlier = df.loc[(df[str_col]>upper_bound) | (df[str_col]<lower_bound)]
        print('In {} column using {} as threshold'.format(str_col,iqr_threshold))
        print('We have {} data points are outliers, which between {} and {}'.format(df_outlier.shape[0],
                                                                                    df_outlier[str_col].min(),
                                                                                    df_outlier[str_col].max()))
        # then we calculate filter out outlers result
        df_clean = df.loc[(df[str_col]<upper_bound) & (df[str_col]>lower_bound)]
        print("After drop outlier, the {} column remain range between {} and {}".format(str_col,
                                                                                        df_clean[str_col].min(),
                                                                                        df_clean[str_col].max()))

        print('Complete IQR eliminate outliers Function.')

        return df_clean, df_outlier

    def drop_na(self, df, list_col):
        """
        Args:
        ------
        df:DataFrame
            The DF has delete some objective columns
        list_col:list
            Generated by EDA() part, like['dt_est','Rate']

        Returns:
        --------
        df_clean:DataFrame
            complete cleaning
        df_dropped:DataFrame
            Contain the rows have been dropped by dropna()
        """
        # sometime, we also need review these columns that has been deleted by rows, so we restore them
        # isnull() will return every element, NaN=True, Value=False
        # any() axis=0, reduced the index, return a Serise whose index is the original column labesl
        # any(axis=1), reduced the columns, return a Serise whose index is the original index label
        # we use isnull().any(axis=1) to check each rows have missing data or not
        df_dropped = df[df.isnull().any(axis=1)]
        # according to eda result, we got columns name that has missing values
        df_clean = df.dropna(subset=list_col, axis=0)
        print("These {} columns still have missing data".format(list_col))
        print("After dropping na, {} rows has decreased to {} rows".format(df.shape[0], df_clean.shape[0]))

        return df_clean, df_dropped

    def convert_time(self, df_product, df_nj_weather, df_pa_weather):
        """:arg
        """
        # **************Convert all time into same timestamp format***************************
        # It should clean data first by columns then by rows, but missing data will appear when you convert time
        # From out put, we can now, 11-05 and 11-06 two records got Nan on ['dt_est'] column,
        # so we need drop them by row in the next part
        # convert weather data ['dt'] (unix time) to Eastern Stardard Time(EST)
        # pass an argument(series) to function tz_convert()
        df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(self.tz_convert)
        # according to Timestamp split into year, quarter, month and day for further merge action
        df_nj_weather['year'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[0])
        df_nj_weather['quarter'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[1])
        df_nj_weather['month'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[2])
        df_nj_weather['day'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[3])
        # another way to use apply()
        # df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(lambda x: pd.TimeStamp(x, unit='s', tz='America/New_York'))
        df_pa_weather['dt_est'] = df_pa_weather['dt'].apply(self.tz_convert)
        # according to Timestamp split into year, quarter, month and day for further merge action
        df_pa_weather['year'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[0])
        df_pa_weather['quarter'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[1])
        df_pa_weather['month'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[2])
        df_pa_weather['day'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[3])
        # for merge purpose, make sure two column have same name
        df_product['dt_est'] = df_product['StartDate'].apply(self.round_to_hour)
        # ***********************End***************************************************

        return df_product, df_nj_weather, df_pa_weather


    def drop_col(self, df_product, df_nj_weather, df_pa_weather):
        """:arg
        From initial observation, we know some columns are not useful data analysis infomation,
        So, we directly delete them by columns according to EDA() results

        """
        class_eda = EDA()
        # ***************Drop non-realted columns**********************
        # *************Drop by columns**************
        # ['Bulk Density'] have 50% missing data
        df_product = df_product.drop(self.PRODUCT_DROP, axis=1)
        print("In Production sheet, these columns have been dropped {}".format(self.PRODUCT_DROP))
        # Drop un-related colunms from weather data
        df_nj_weather = df_nj_weather.drop(self.WEATHER_DROP, axis=1)
        # some weather data might also need clearn
        list_missing_nj = class_eda.missing_plot(df_nj_weather)
        # axis=0/index axis=1/columns
        df_nj_weather = df_nj_weather.drop(list_missing_nj, axis=1)
        list_missing_pa = class_eda.missing_plot(df_pa_weather)
        df_pa_weather = df_pa_weather.drop(list_missing_pa, axis=1)
        # ******************End************************************
        return df_product, df_nj_weather, df_pa_weather


    def drop_outlier(self, df_product):
        """:arg

        """
        #*****************Drop by outliers****************
        # after drop non-related columns, convert timestamp format, drop missing data by rows,
        # we need to drop outliers in some necessary columns
        # we might need automate generate outlier by some EDA() models
        str_col='YieldPercentage'
        df_product, df_outlier_1 = self.z_outlier(df_product, str_col, z_threshold = self.YEILD_THRESHOLD, abs=False)
        # ['Rate'] data have a lot of extramly large data. According to human judgement, >2000 might be delete.
        # But when we use statistical method to testify our data outliers
        # In z-score function, if we set value to 3, ['Rate']>1275 row=3 will be delete
        # if we set value to 2, ['Rate']>973 row=247 will be delete
        df_product, df_outlier_2 = self.z_outlier(df_product, 'Rate', z_threshold = self.IQR_THRESHOLD, abs=False)
        # concate outliers into one data frame
        frames = [df_outlier_1, df_outlier_2]
        df_outlier = pd.concat(frames)
        #*****************************End**********************

        return df_product, df_outlier




    def clean_modify(self, df_product, df_nj_weather, df_pa_weather):
        """:arg
        """
        # ********************Adidtional modified*****************************
        # we have a few small changes based on some inputing error
        # For example ['Food Addit'] == ['Food Additive']
        df_product.loc[df_product['ProdLine'] == 'Food Addit', 'ProdLine'] = 'Food Additive'
        # PA only proedure food related (non-Fragance) kind custome material
        # we find a record in PA location index=3299, Batchnumber = PASD340354
        # It could be correct record but for now we delete this line
        ind = df_product[(df_product['ProdLine'] == 'Fragrance') & (df_product['BatchNumber'].str.contains('PA'))].index
        # we can use former index to drop, beacuse it's two long for one line code.
        df_product.drop(index=ind, inplace=True)

        # when we deal with product data, we find some dryer in the wrong location.
        # Generally, NJ has Dryer 01 -04, PA has Dryer 06-11.
        # For index=7491, BatchNumber ='NJSD362799' this records originally palced in NJ and then swithed over
        # to be run in PA. Typically, Spray-Tek protocal is cancel the NJ order and re-submit it as a PA order
        # In this records, the NJ order was not cancelled. The dyer number is accurate and the location is incorrect
        df_nj_1 = df_product[df_product['BatchNumber'].str.contains('NJ')]
        # then we find the index of these error recording columns
        idx_1 = df_nj_1.loc[(df_nj_1['Dryer'] == 'Dryer 06') | (df_nj_1['Dryer'] == 'Dryer 09')].index
        # idx_1 = df_nj_1.loc[df_nj_1['Dryer'].isin(['Dryer 06','Dryer 09'])].index
        # The best way to replace/assign value is use .loc[] to locate and then assign new value 'PA000'
        df_product.loc[idx_1, 'BatchNumber'] = 'PA000'
        # vice-versa, we find Dryer01 in PA, it should be in NJ
        # there index=(2761, 3245), BatchNumber=('PASD337379', 'PASD337014')
        df_pa_1 = df_product[df_product['BatchNumber'].str.contains('PA')]
        idx_2 = df_pa_1.loc[(df_pa_1['Dryer'] == 'Dryer 01')].index
        df_product.loc[idx_2, 'BatchNumber'] = 'NJ000'
        # anthoer way, If you want to assign multi-values to a pd.Series, the other head also should be pd.Serise
        # df_product.loc[idx_2, 'BatchNumber'] = df_product.loc[idx_2, 'BatchNumber'].map(lambda x: 'NJ111').values

        # *************************End**********************************************
        return df_product, df_nj_weather, df_pa_weather




    def clean_data(self, df_product, df_nj_weather, df_pa_weather):
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

        class_eda = EDA()


        #**************Convert all time into same timestamp format***************************
        # It should clean data first by columns then by rows, but missing data will appear when you convert time
        # From out put, we can now, 11-05 and 11-06 two records got Nan on ['dt_est'] column,
        # so we need drop them by row in the next part
        # convert weather data ['dt'] (unix time) to Eastern Stardard Time(EST)
        # pass an argument(series) to function tz_convert()
        df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(self.tz_convert)
        # according to Timestamp split into year, quarter, month and day for further merge action
        df_nj_weather['year'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[0])
        df_nj_weather['quarter'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[1])
        df_nj_weather['month'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[2])
        df_nj_weather['day'] = df_nj_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[3])
        # another way to use apply()
        # df_nj_weather['dt_est'] = df_nj_weather['dt'].apply(lambda x: pd.TimeStamp(x, unit='s', tz='America/New_York'))
        df_pa_weather['dt_est'] = df_pa_weather['dt'].apply(self.tz_convert)
        # according to Timestamp split into year, quarter, month and day for further merge action
        df_pa_weather['year'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[0])
        df_pa_weather['quarter'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[1])
        df_pa_weather['month'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[2])
        df_pa_weather['day'] = df_pa_weather['dt_est'].apply(lambda x: self.tz_y_m_d(x)[3])
        # for merge purpose, make sure two column have same name
        df_product['dt_est'] = df_product['StartDate'].apply(self.round_to_hour)
        #***********************End***************************************************


        #***************Drop non-realted columns**********************
        # *************Drop by columns**************
        # ['Bulk Density'] have 50% missing data
        df_product = df_product.drop(self.PRODUCT_DROP, axis=1)
        print("In Production sheet, these columns have been dropped {}".format(self.PRODUCT_DROP))
        # Drop un-related colunms from weather data
        df_nj_weather = df_nj_weather.drop(self.WEATHER_DROP, axis = 1)
        # some weather data might also need clearn
        list_missing_nj = class_eda.missing_plot(df_nj_weather)
        df_nj_weather = df_nj_weather.drop(list_missing_nj, axis = 1)
        list_missing_pa = class_eda.missing_plot(df_pa_weather)
        df_pa_weather = df_pa_weather.drop(list_missing_pa, axis = 1)
        #******************End************************************


        #***********************Drop missing data********************************
        list_col_missing_product = class_eda.missing_plot(df_product)
        #**************Drop by rows**********************
        # we get the name of columns that has missing value from eda part
        df_product,df_dropped = self.drop_na(df_product, list_col_missing_product)
        #******************End*****************


        #*****************Drop by outliers****************
        # after drop non-related columns, convert timestamp format, drop missing data by rows,
        # we need to drop outliers in some necessary columns
        # we might need automate generate outlier by some EDA() models
        str_col='YieldPercentage'
        df_product, df_outlier_1 = self.z_outlier(df_product, str_col, z_threshold = self.YEILD_THRESHOLD, abs=False)
        # ['Rate'] data have a lot of extramly large data. According to human judgement, >2000 might be delete.
        # But when we use statistical method to testify our data outliers
        # In z-score function, if we set value to 3, ['Rate']>1275 row=3 will be delete
        # if we set value to 2, ['Rate']>973 row=247 will be delete
        df_product, df_outlier_2 = self.z_outlier(df_product, 'Rate', z_threshold = self.IQR_THRESHOLD, abs=False)
        # concate outliers into one data frame
        frames = [df_outlier_1, df_outlier_2]
        df_outlier = pd.concat(frames)
        #*****************************End**********************


        #********************Adidtional modified*****************************
        # we have a few small changes based on some inputing error
        # For example ['Food Addit'] == ['Food Additive']
        df_product.loc[df_product['ProdLine'] == 'Food Addit', 'ProdLine'] = 'Food Additive'
        # PA only proedure food related (non-Fragance) kind custome material
        # we find a record in PA location index=3299, Batchnumber = PASD340354
        # It could be correct record but for now we delete this line
        ind=df_product[(df_product['ProdLine'] == 'Fragrance') & (df_product['BatchNumber'].str.contains('PA'))].index
        # we can use former index to drop, beacuse it's two long for one line code.
        df_product.drop(index=ind, inplace=True)

        # when we deal with product data, we find some dryer in the wrong location.
        # Generally, NJ has Dryer 01 -04, PA has Dryer 06-11.
        # For index=7491, BatchNumber ='NJSD362799' this records originally palced in NJ and then swithed over
        # to be run in PA. Typically, Spray-Tek protocal is cancel the NJ order and re-submit it as a PA order
        # In this records, the NJ order was not cancelled. The dyer number is accurate and the location is incorrect
        df_nj_1 = df_product[df_product['BatchNumber'].str.contains('NJ')]
        # then we find the index of these error recording columns
        idx_1 = df_nj_1.loc[(df_nj_1['Dryer'] == 'Dryer 06') | (df_nj_1['Dryer'] == 'Dryer 09')].index
        # idx_1 = df_nj_1.loc[df_nj_1['Dryer'].isin(['Dryer 06','Dryer 09'])].index
        # The best way to replace/assign value is use .loc[] to locate and then assign new value 'PA000'
        df_product.loc[idx_1, 'BatchNumber'] = 'PA000'
        # vice-versa, we find Dryer01 in PA, it should be in NJ
        # there index=(2761, 3245), BatchNumber=('PASD337379', 'PASD337014')
        df_pa_1 = df_product[df_product['BatchNumber'].str.contains('PA')]
        idx_2 = df_pa_1.loc[(df_pa_1['Dryer'] == 'Dryer 01')].index
        df_product.loc[idx_2, 'BatchNumber'] = 'NJ000'
        # anthoer way, If you want to assign multi-values to a pd.Series, the other head also should be pd.Serise
        # df_product.loc[idx_2, 'BatchNumber'] = df_product.loc[idx_2, 'BatchNumber'].map(lambda x: 'NJ111').values

        #*************************End**********************************************


        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End clean_data() with {} second".format(cost_time), "*" * 40, end='\n\n')

        return df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier

