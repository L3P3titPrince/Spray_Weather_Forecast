from class_31_hyperparameters import HyperParamters
from class_32_import_data import ImportData
from class_33_preprocess import PreProcess


def main():
    """
    We use this function to call process one by one.
    """


    # ***********************import******************************
    class_import = ImportData()
    df_product, df_nj_weather, df_pa_weather = class_import.import_data()

    # *******************3.Preprocess**************************************
    class_pre = PreProcess()
    df_product, df_nj_weather, df_pa_weather = class_pre.clean_data(df_product, df_nj_weather, df_pa_weather)


    return df_product, df_nj_weather, df_pa_weather

if __name__=="__main__":
    """:arg
    
    """
    (df_product, df_nj_weather, df_pa_weather) = main()
    print("OVER")