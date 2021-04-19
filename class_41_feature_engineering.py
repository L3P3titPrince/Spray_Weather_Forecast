from class_31_hyperparameters import HyperParamters

import numpy as np
# calculate slope
import statsmodels.api as sm
from tqdm import tqdm
# split data into train, validataion and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
            # there are some data points that too small to have std, so, this featuures will suspend
            # df_prod_nj.insert(df_prod_nj.shape[1], (col_name + "_std"), value=df_nj_groupby.std())
            # df_prod_pa.insert(df_prod_pa.shape[1], (col_name + "_std"), value=df_pa_groupby.std())
            # # MAX
            # df_prod_nj.loc[:, (col_name + "_max")] = df_nj_groupby.max()
            # df_prod_pa.loc[:, (col_name + "_max")] = df_pa_groupby.max()
            df_prod_nj.name = "df_prod_nj"
            df_prod_pa.name = "df_prod_pa"

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

    def split_df(self,df, col_1='Dryer', col_2='ProdLine'):
        """
        we use this to split each dataframe into subplot and output variable name is assgined by there categorical
        for example, If i want to check dryer 08 and its ProdLine ='Flavors', you can get a dataframe which
        vairable name is df_08_Flavors
        But problems is that I can't return them all, so I choose to use dict to restore vairable name and
        dataframe as key and value

        Args:
        --------
        df:pd.DataFrame

        col_1:string
            Default is ['Dryer'] and it contains white space in category string, so if you want to
            change to other category, please remain col_1 and chagne col_2
        col_2:string
            defualt is ['ProdLine']

        Return:
        ---------

        """
        # get the groupby/unique list result of this dataframe, output is pandas.series
        # For now, we only can support two layer, and default arr_1 is ['Dryer']
        arr_1 = df[col_1].unique()
        # default is ['ProdLine'] and might change to ['CustItem']
        arr_2 = df[col_2].unique()

        # create new empty list to restore dataframe variable name and its row numbers
        list_var_name, list_shape_row, dict_df= [], [], {}

        # iterate from first layer ['Dryer']
        for dryer in arr_1:
            for prod in arr_2:
                # get the number of dryer because there is white space in this column
                str_num = str(dryer.split(' ')[1])
                str_var = "df" + "_" + str_num + "_" + prod
                # use combine string as new dataframe variable name
                locals()[str_var] = df[(df[col_1]==dryer) & (df[col_2]==prod)]
                # restore the variable name to new list, and using this list to get
                list_var_name.append(str_var)
                # and also retore the row number for plot
                list_shape_row.append(df[(df[col_1]==dryer) & (df[col_2]==prod)].shape[0])
                # consider we need to output as return, we can't directly return these new df
                # so we contain them into a dictionary and return this new dictionary
                dict_new = {str_var : df[(df[col_1]==dryer) & (df[col_2]==prod)]}
                dict_df.update(dict_new)

        # create new dataframe for pie plot
        df_pie = pd.DataFrame(list(zip(list_var_name, list_shape_row)), columns=['DataFrame', 'RowNumber'])
        # plot a pie pplot to demonstrate how imbalance data
        # plot = df_pie['RowNumber'].plot.pie(labels=df_pie['DataFrame'], autopct='%.2f', fontsize=10, figsize=(20, 20))
        df_dropped = df_pie[df_pie['RowNumber'] == 0]
        print("These df has zero records, so we don't display them: \n{}"
              .format(df_dropped['DataFrame']))
        class_eda = EDA()
        class_eda.squrify_treemap(df_pie[df_pie['RowNumber']!=0], bool_groupby=False)
        print("Its more easy to use records more than 100: \n{}"
              .format(df_pie[df_pie['RowNumber'] > 100]['DataFrame']))


        return dict_df, df_pie



    def ten_calssify(self, label_col):
        """
        This function will manually segement numerical。
        Input data should be between [0,1] normalization result
        For X part we use standardization to maintain outliers,
        for y part we use normalization to matatin equal seperate

        Args:
        ------
        label_col:pd.Series
            Typyical is one of columns in dataframe, but after normalization

        Return:
        ------
        categrocial:np.ndarray
            It will be a (n,10) matrxie, n will be the length of input pd.Series
        """
        # initial empty matrix
        categorical = np.zeros((len(label_col), 10), dtype='float32')
        # find the corresponding cell by (row, column) and set that as
        for idx, label in enumerate(label_col):
            if 0 <= label < 0.1:
                categorical[idx, 0] = 1
            elif 0.1 <= label < 0.2:
                categorical[idx, 1] = 1
            elif 0.2 <= label < 0.3:
                categorical[idx, 2] = 1
            elif 0.3 <= label < 0.4:
                categorical[idx, 3] = 1
            elif 0.4 <= label < 0.5:
                categorical[idx, 4] = 1
            elif 0.5 <= label < 0.6:
                categorical[idx, 5] = 1
            elif 0.6 <= label < 0.7:
                categorical[idx, 6] = 1
            elif 0.7 <= label < 0.8:
                categorical[idx, 7] = 1
            elif 0.8 <= label < 0.9:
                categorical[idx, 8] = 1
            elif 0.9 <= label <= 1.0:
                categorical[idx, 9] = 1
            else:
                print('ERROR', label)

        # test part, if our calcuatlion is correct, all row should be included and have exactly number one
        # If correct, nothing happen, if condition return false, AssertionError is raised
        assert np.sum(categorical, axis=1).sum() == len(label_col)
        #         (unique, counts) = np.unique(test_13, return_counts=True)
        return categorical


    def five_calssify(self, label_col):
        """
        This function will manually segement numerical。
        Input data should be between [0,1] normalization result
        For X part we use standardization to maintain outliers,
        for y part we use normalization to matatin equal seperate

        Args:
        ------
        label_col:pd.Series
            Typyical is one of columns in dataframe, but after normalization

        Return:
        ------
        categrocial:np.ndarray
            It will be a (n,10) matrxie, n will be the length of input pd.Series
        """
        # initial empty matrix
        categorical = np.zeros((len(label_col), 5), dtype='float32')
        # find the corresponding cell by (row, column) and set that as
        for idx, label in enumerate(label_col):
            if 0 <= label < 0.2:
                categorical[idx, 0] = 1
            elif 0.2 <= label < 0.4:
                categorical[idx, 1] = 1
            elif 0.4 <= label < 0.6:
                categorical[idx, 2] = 1
            elif 0.6 <= label < 0.8:
                categorical[idx, 3] = 1
            elif 0.8 <= label <= 1.0:
                categorical[idx, 4] = 1
            else:
                print('ERROR', label)

        # test part, if our calcuatlion is correct, all row should be included and have exactly number one
        # If correct, nothing happen, if condition return false, AssertionError is raised
        assert np.sum(categorical, axis=1).sum() == len(label_col)
        #         (unique, counts) = np.unique(test_13, return_counts=True)
        return categorical


    def feature_scaling(self, df):
        """
        Contain standardize for X part and normalize for y part

        Args:
        -------
        df:pd.DataFrame
            for now we only use dataframe with filter result, like df_08_Flavors

        Return:
        ------
        X_train_std:
            60% + Standardization
        """
        # in the future, we will find a way to use time as part our of our input, but now, we ignore the time
        # maybe we can only categorical quarter to four one-hot, and transfrom 24 hours to day and night ont-hot
        list_col = ['hours', 'humidity_max','humidity_min', 'humidity_mean', 'humidity_median', 'humidity_trend',
                  'temp_max', 'temp_min', 'temp_mean', 'temp_median', 'temp_trend',
                  'temp_min_max', 'temp_min_min', 'temp_min_mean', 'temp_min_median',
                  'temp_min_trend', 'temp_max_max', 'temp_max_min', 'temp_max_mean',
                  'temp_max_median', 'temp_max_trend', 'pressure_max', 'pressure_min',
                  'pressure_mean', 'pressure_median', 'pressure_trend']

        # create y target labels, due to scalar only process dataframe,so here we use two []
        # to make sure y is a dataframe not a pd.Serise which can't be direct use in MinMaxScaler()
        y = df[['Rate']].copy()
        X = df[list_col].copy()
        # 1. Split into train, validation, and test, train will be one group, test and valistion will be another
        X_train ,X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.4, random_state=1024)

        # 2. For X part, apply standardizaion, for y part, we use normalization
        # create standard scaler
        scaler_std = StandardScaler()
        # fit and transofrm only to train data
        X_train_std = scaler_std.fit_transform(X_train)
        # create Min_Max normliazation sclaer
        scaler_mms = MinMaxScaler()
        # fit and transofrm to
        y_train_mms = scaler_mms.fit_transform(y_train)

        # 3.Fit standardize and normlization to another part
        X_val_test_std = scaler_std.transform(X_val_test)
        y_val_test_mms = scaler_mms.transform(y_val_test)

        #3. split validatiaon and test into equal part
        X_val, X_test, y_val, y_test = train_test_split(X_val_test_std, y_val_test_mms,
                                                        test_size=0.5, random_state=1024)

        return X_train_std, X_val, X_test, y_train_mms, y_val, y_test

    def test_for_whole_dataset(self):
        """
        This is just for test
        :return:
        """
        # instert dryer_numbers by only extract numer from ['Dryer']
        df_prod_nj_2.insert(4, 'Dryer_num', df_prod_nj_2['Dryer'].str.extract('(\d+)'))
        # reverse Rate?
        new_rate  = ActualyQty/dryging_time

        return None