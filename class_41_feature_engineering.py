from class_31_hyperparameters import HyperParamters

import numpy as np
# calculate slope
import statsmodels.api as sm
from tqdm import tqdm

class FeatureEngineer(HyperParamters):
    """
    If we directly use our original features, we can't get useful relationship
    So, we need using feature engineering to get more significent



    """
    def __init__(self):
        """

        """
        HyperParamters.__init__(self)

    def get_trend(self, series_feature):
        """
        get the slope of time series, if the slope is not significant, set it to 0

        In linear regression (ordinary least squares), y = theta_0 + theta_1 * x
        H0: theta_1 is zero, y is constant
        H1: theta 1 is not zero. y is linear
        When P-value of theta_1 is smaller than 0.05, we reject H0 and accept, believe this function is liearn.
        If p-value > 0.05, we accept H0, reject H1, believe X is not related to Y and Y is constant.

        For example, if we got three sequence [51,45,47], we use OLS() to fit(). Finally we got a negtive
        slope y =49-2x, but they are not that significent and p-value['x1'] is 0.54 > 0.05, not significant.

        In this function, only significant data trend will be return with non-zero value. If p-value>0.05,
        we will return zero and igonre the slope and trend for this features

        Args:
        -----------------
        series_feature


        Returns:
        -----------------
        trend[float] (slope)
        """
        # create a empty parameters theta vector for sm.OLS to train.
        theta = np.arange(0, len(series_feature))
        # add constant in theta parameters, first column is constant [1,1,,,,1], second column is theta parameters
        theta = sm.add_constant(theta)
        # in OLS() function, first is Y (target) , second is X
        result = sm.OLS(series_feature, theta).fit()
        # if result p-value < 0.05, reject H0 (theta_1 is zero, Y and x not relevant)
        # accept H1, theta_1(slope) is not zero, y is linear, y and x are significant relevant
        if result.pvalues['x1'] < 0.05:
            # we output the slope of this simple linear model
            return result.params['x1']
        else:
            # we think the trend/fluctuate in this sequence is not sigificent, so we output 0 as flatten result
            return 0



    def feature_eng(self, df_product, df_nj, df_pa):
        """
        This is feature engineering function, we need reserve all
        max / min / mean / midlle / trend / std

        We want use parameters to get the characteristic、traits of this sequence
        Yes, this is more like a time sequence, we might can use RNN to try to import them like a sequence
        And also, it will not be a single RNN model, we have several featers/models and then conncate them together

        Args:
        -------
        df_product:DataFrame
            It should be a cleaned df, in big picture, it will be df_product_6 after most of preprocessing
        df_pa:DataFrame
            This df contains all info product and weather

        Returns:
        --------
        df_prod_nj
        """

        # Add a model to check

        # first we need use cleaned product DataFrame to build a new dataset for Neual network
        # set some colunms we need to use in Neural Network / PCA / Decision Tree
        #     col_names = ['Index_ID','CustItem','Dryer','dt_est', 'Rate']
        df_prod_nj = df_product.loc[df_product['BatchNumber'].str.contains('NJ', regex=False)]
        df_prod_pa = df_product.loc[df_product['BatchNumber'].str.contains('PA', regex=False)]

        # Add a model to check
        if (df_prod_nj.shape[0] == df_nj.groupby(['Index_ID']).count().shape[0]):
            print('YES', df_prod_nj.shape[0], df_nj.groupby(['Index_ID']).count().shape[0])
        else:
            print('NO', df_prod_nj.shape[0], df_nj.groupby(['Index_ID']).count().shape[0])

        # Add a model to check
        if (df_prod_pa.shape[0] == df_pa.groupby(['Index_ID']).count().shape[0]):
            print('YES', df_prod_pa.shape[0], df_pa.groupby(['Index_ID']).count().shape[0])
        else:
            print('NO', df_prod_pa.shape[0], df_pa.groupby(['Index_ID']).count().shape[0])

        # DataFrome of PA with feature engineering,
        # df_product_fe = df_product[.loc[:,col_names]

        # create a list contain all the columns we need to process with feature engineering
        list_col = ['humidity', 'temp', 'temp_min', 'temp_max', 'pressure']

        for i in tqdm(range(0, len(list_col))):
            # print(list_col[i])
            col_name = list_col[i]
            print(col_name)
            # In order to avoid chain assign, which will confused about using a view or a copy
            # we just to clear state that we made a dataframe copy first
            df_nj_groupby = df_nj.groupby(by=['Index_ID'])[col_name]
            df_pa_groupby = df_pa.groupby(by=['Index_ID'])[col_name]
            # for each new column, name will be it attributes+"max/min/mean/middle/std"
            # this is best way to add new column without settingwithcopywarning. loc() is only for change exist col
            # max()
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name+"_max"), value=df_nj_groupby.max())
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name+"_max"), value=df_pa_groupby.max())
            # MIN
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name+"_min"), value=df_nj_groupby.min())
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name+"_min"), value=df_pa_groupby.min())
            # mean
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name+"_mean"), value=df_nj_groupby.mean())
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name+"_mean"), value=df_pa_groupby.mean())
            # middle / median
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name+"_median"), value=df_nj_groupby.median())
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name+"_median"), value=df_pa_groupby.median())
            # trend
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name+"_trend"), value=df_nj_groupby.apply(self.get_trend))
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name+"_trend"), value=df_pa_groupby.apply(self.get_trend))
            # standard deviation
            df_prod_nj.insert(df_prod_nj.shape[1], (col_name + "_std"), value=df_nj_groupby.std())
            df_prod_pa.insert(df_prod_pa.shape[1], (col_name + "_std"), value=df_pa_groupby.std())
            # # MAX
            # df_prod_nj.loc[:, (col_name + "_max")] = df_nj_groupby.max()
            # df_prod_pa.loc[:, (col_name + "_max")] = df_pa_groupby.max()

        return df_prod_nj, df_prod_pa


    def test_get_trend(self, df_nj):
        """
        In my full stack system, I need build a test model to make sure this model is correct when modifed
        We only use the first 3
        :return:
        """
        # get the ['humidity'] sequence of index_id=0
        hum_01 = df_nj[df_nj['Index_ID']=='0']['humidity']
        print("True Value ['51.0', '45.0', '47.0'], Test Value {}".format(hum_01))


        return None