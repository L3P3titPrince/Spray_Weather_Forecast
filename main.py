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
    # if you need use self.df_multi in the next line, you need call function to product self.multi first
    df_multi = class_merge.row_proliferate(df_product)
    # identify this is only belong to
    df_nj_loc, df_pa_loc = class_merge.merge_location(df_product, df_nj_weather, df_pa_weather)


    return (df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier,
            df_multi, df_nj_loc, df_pa_loc,)



if __name__=="__main__":
    """:arg
    
    """
    (df_product, df_nj_weather, df_pa_weather, df_dropped, df_outlier,
     df_multi, df_nj_loc, df_pa_loc,) = main()

    print("OVER")