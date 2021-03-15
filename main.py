from class_31_hyperparameters import HyperParamters
from class_32_import_data import ImportData
from class_34_preprocess import PreProcess
from class_33_eda import EDA
from class_35_merge_data import MergeData

def main():
    """
    We use this function to call process one by one.
    """


    # ***********************import******************************
    class_import = ImportData()
    # df_product, df_nj_weather, df_pa_weather = class_import.import_data()
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


    # df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier = class_pre.clean_data(df_raw_product,
    #                                                                                         df_raw_nj_weather,
    #                                                                                         df_raw_pa_weather)

    #*******************3.5 MergeDate************************
    class_merge = MergeData()
    # if you need use self.df_multi in the next line, you need call function to product self.multi first
    df_multi = class_merge.row_proliferate(df_product_6)
    # identify this is only belong to
    df_nj, df_pa = class_merge.merge_location(df_product_6, df_nj_weather_6, df_pa_weather_6)


    return (df_product_1, df_nj_weather_1, df_pa_weather_1, df_product_2, df_nj_weather_2, df_pa_weather_2,
            df_product_3, df_nj_weather_3, df_pa_weather_3, df_product_4, df_product_5, df_dropped, df_outlier,
            df_product_6, df_nj_weather_6, df_pa_weather_6,
            df_multi, df_nj, df_pa,)



if __name__=="__main__":
    """:arg
    
    """
    (df_product_1, df_nj_weather_1, df_pa_weather_1, df_product_2, df_nj_weather_2, df_pa_weather_2,
     df_product_3, df_nj_weather_3, df_pa_weather_3, df_product_4, df_product_5, df_dropped, df_outlier,
     df_product_6, df_nj_weather_6, df_pa_weather_6,
     df_multi, df_nj, df_pa,) = main()

    print("OVER")