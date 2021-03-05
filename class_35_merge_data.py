from class_31_hyperparameters import HyperParamters

from pandas import pd

class MergeData(HyperParamters):
    """:arg
    For now, we have differenty hypothesis to view our data
    We will use self.MERGE_WAY to choose different hypothesis

    1.self.MERGE_WAY:one
        without further process, just using dt_est to merge
    2.self.MERGE_WAY:

    """
    def __init__(self):
        HyperParamters.__init__(self)

    def row_proliferate(self):

        return None



    def merge_one(self, df_product, df_nj_weather, df_pa_weather):
        """:arg
        Don't split, just merge all together by ['dt_est']
        """
        df_nj = pd.merge(df_product, df_nj_weather, how='left', on=['dt_est'])
        df_pa = pd.merge(df_product, df_pa_weather, how='left', on=['dt_est'])

        return df_nj, df_pa

