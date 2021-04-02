from class_31_hyperparameters import HyperParamters
from class_33_eda import EDA
from class_34_preprocess import PreProcess

import pandas as pd

class SplitCompile(HyperParamters):
    """

    """
    def __init__(self):
        """

        """
        HyperParamters.__init__(self)


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


    def preprocess(self, df):
        """
        Args:
        df
        This can be any

        :return:
        """
        # in the future, we will find a way to use time as part our of our input, but now, we ignore the time
        # maybe we can only categorical quarter to four one-hot, and transfrom 24 hours to day and night ont-hot
        list_nn = ['hours', 'humidity_max','humidity_min', 'humidity_mean', 'humidity_median', 'humidity_trend',
                  'temp_max', 'temp_min', 'temp_mean', 'temp_median', 'temp_trend',
                  'temp_min_max', 'temp_min_min', 'temp_min_mean', 'temp_min_median',
                  'temp_min_trend', 'temp_max_max', 'temp_max_min', 'temp_max_mean',
                  'temp_max_median', 'temp_max_trend', 'pressure_max', 'pressure_min',
                  'pressure_mean', 'pressure_median', 'pressure_trend',
                  'Rate']
        # only extract the columns in list_nn, there are numerical and
        df_nn = df[list_nn]



        # we might need to categorical each dryer into one-hot encoding
        list_dryer = ['Dryer 01', 'Dryer 02', 'Dryer 03', 'Dryer 04']

        return None



