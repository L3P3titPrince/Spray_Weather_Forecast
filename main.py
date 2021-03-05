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
    df_product, df_nj_weather, df_pa_weather = class_import.read_csv()

    # *******************3.Preprocess**************************************
    class_pre = PreProcess()
    df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier = class_pre.clean_data(df_product,
                                                                                df_nj_weather,
                                                                                df_pa_weather)

    #*******************3.5 MergeDate************************
    class_merge = MergeData()
    df_nj, df_pa = class_merge.merge_one(df_product, df_nj_weather, df_pa_weather)
    df_multi = class_merge.row_proliferate(df_product)

    return (df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier,
            df_nj, df_pa, df_multi)



if __name__=="__main__":
    """:arg
    
    """
    (df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier,
     df_nj, df_pa, df_multi) = main()

    print("OVER")