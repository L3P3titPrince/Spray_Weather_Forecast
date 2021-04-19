from class_31_hyperparameters import HyperParamters
from class_32_import_data import ImportData
from class_34_preprocess import PreProcess
from class_33_eda import EDA
from class_35_merge_data import MergeData
from class_41_feature_engineering import FeatureEngineer
from class_42_split_compile import SplitCompile
from class_51_nn_models import NNModels

import pandas as pd

import bar_chart_race as bcr

def main():
    """
    We use this function to call process one by one.
    """

    """
    # ***********************import******************************
    class_import = ImportData()
    # If you don't have csv file, you should run this line first. Reading csv is more quick than excel
    # df_product_1, df_nj_weather_1, df_pa_weather_1 = class_import.import_data()
    df_product_1, df_nj_weather_1, df_pa_weather_1 = class_import.read_csv()

    # **************************EDA***************************
    class_eda = EDA()

    # *******************3.4 Preprocess**************************************
    class_pre = PreProcess()
    # convert timestmap into same format and generate detail time columns
    df_product_2, df_nj_weather_2, df_pa_weather_2 = class_pre.convert_time(df_product_1,
                                                                            df_nj_weather_1,
                                                                            df_pa_weather_1)
    # drop meanningless columns from hyperparameter and EDA result
    df_product_3, df_nj_weather_3, df_pa_weather_3 = class_pre.drop_col(df_product_2,
                                                                        df_nj_weather_2,
                                                                        df_pa_weather_2)
    # drop rows still have NaN, these just normal missing data caused by operation error
    list_col_missing_product = class_eda.missing_plot(df_product_3)
    # Drop by rows
    # we get the name of columns that has missing value from eda part
    df_product_4, df_dropped = class_pre.drop_na(df_product_3, list_col_missing_product)
    # drop outliers
    df_product_5 ,df_outlier = class_pre.drop_outlier(df_product_4)
    # There are some human issue we can modify and correct here
    df_product_6, df_nj_weather_6, df_pa_weather_6 = class_pre.clean_modify(df_product_5,
                                                                            df_nj_weather_3,
                                                                            df_pa_weather_3)

    # **********************WARNING*****************************
    # **********************WARNING*****************************
    # we should not add this code in final code, but we can temperaoly delete ['Rate']<20 for statistical result
    # Because running ['Rate'] might related to Dryer size. So we delete Dryer09 temporary
    df_product_6 = df_product_6.loc[df_product_6['Dryer']!='Dryer 09']
    df_product_6 = df_product_6.loc[df_product_6['Rate']>20]
    # **********************WARNING****************************
    # **********************WARNING****************************

    # df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier = class_pre.clean_data(df_raw_product,
    #                                                                                         df_raw_nj_weather,
    #                                                                                         df_raw_pa_weather)


    # *********************EDA*******************************
    # 1. Bar_Race_Chart
    # df_base, df_bcr= class_eda.bcr_dryer(df_product_6, df_nj_weather_6, df_pa_weather_6)
    # # we can also read it from files
    # # df_bcr = pd.read_csv('03_data/27_bcr.csv', index_col=0)
    # bcr_html = bcr.bar_chart_race(df=df_bcr.head(10), filename='30.mp4')


    #*******************3.5 MergeDate************************
    class_merge = MergeData()
    # if you need use self.df_multi in the next line, you need call function to product self.multi first
    df_multi = class_merge.row_proliferate(df_product_6)
    # identify this is only belong to
    df_nj, df_pa = class_merge.merge_location(df_product_6, df_nj_weather_6, df_pa_weather_6)

    #********************4.1 Feature Enginnering**************************
    class_fe = FeatureEngineer()
    # using feature_eng() function to produce a new df with new feature per records (not multi records but groupby)
    df_prod_nj, df_prod_pa = class_fe.feature_eng(df_product_6, df_nj, df_pa)
    """




    #--------------------------------WARNING-----------------------------------
    #--------------------------------WARNING-----------------------------------
    #********************************INTERUPT FOR SPPED UP RUNNING TIME ************************
    # restore into 03_data
    df_prod_nj.to_csv('03_data/34_df_prod_nj.csv', index=False)
    df_prod_pa.to_csv('03_data/35_df_prod_pa.csv', index=False)
    # to read fast
    # df_prod_nj = pd.read_csv('03_data/34_df_prod_nj.csv')
    # df_prod_pa = pd.read_csv('03_data/35_df_prod_pa.csv')
    #***********************************************************************************************
    #--------------------------------WARNING-----------------------------------
    #--------------------------------WARNING-----------------------------------


    #*******************4.2 Split into sub dataframe*************************
    class_sc = SplitCompile()
    dict_prod_nj, df_pie_nj = class_sc.split_df(df_prod_nj)
    dict_prod_pa, df_pie_pa = class_sc.split_df(df_prod_pa)
    # get the useful df from pervious result
    df_08_Flavors = dict_prod_pa['df_08_Flavors']
    # normalize and standardlize
    class_fe = FeatureEngineer()
    X_train, X_val, X_test, y_train, y_val, y_test = class_fe.feature_scaling(df_08_Flavors)
    # transform y part into one hot
    arr_y_train = class_fe.five_calssify(y_train)
    arr_y_val = class_fe.five_calssify(y_val)
    arr_y_test = class_fe.five_calssify(y_test)

    #**********************5.1 MLP model********************************
    class_nn = NNModels()
    model_2 = class_nn.mlp_model()
    hist_2 = model_2.fit(X_train, arr_y_train, validation_data=(X_val, arr_y_val), epochs=30, batch_size=32, verbose=1)

    return (df_product_1, df_nj_weather_1, df_pa_weather_1, df_product_2, df_nj_weather_2, df_pa_weather_2,
            df_product_3, df_nj_weather_3, df_pa_weather_3, df_product_4, df_product_5, df_dropped, df_outlier,
            df_product_6, df_nj_weather_6, df_pa_weather_6,
            df_multi, df_nj, df_pa, df_prod_nj, df_prod_pa, dict_prod_nj, df_pie_nj , dict_prod_pa, df_pie_pa,
            df_08_Flavors, X_train, X_val, X_test, y_train, y_val, y_test,
            arr_y_train, arr_y_val, arr_y_test, hist_2)



if __name__=="__main__":
    """:arg
    
    """
    (df_product_1, df_nj_weather_1, df_pa_weather_1, df_product_2, df_nj_weather_2, df_pa_weather_2,
     df_product_3, df_nj_weather_3, df_pa_weather_3, df_product_4, df_product_5, df_dropped, df_outlier,
     df_product_6, df_nj_weather_6, df_pa_weather_6,
     df_multi, df_nj, df_pa, df_prod_nj, df_prod_pa, dict_prod_nj, df_pie_nj , dict_prod_pa, df_pie_pa,
     df_08_Flavors, X_train, X_val, X_test, y_train, y_val, y_test,
     arr_y_train, arr_y_val, arr_y_test, hist_2) = main()

    print("OVER")