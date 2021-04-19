from class_31_hyperparameters import HyperParamters

# used for visulazation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import squarify

class EDA(HyperParamters):
    """:arg
    """
    def __init__(self):
        """:arg
        """
        HyperParamters.__init__(self)

    def missing_plot(self, df: pd.DataFrame):
        """:arg

        Used for missing data plot for count plot and percenatage plot
        """
        # not all columns contain missing value, we only plot the colunms have missing value
        # checking is there any missing value in this df and find threir column names and assign to col_missing_value
        col_missing_value = list(df.columns[df.isnull().any()])
        # count the missing value for each column
        df[col_missing_value].isnull().sum()
        # to hold the columns names
        list_labels=[]
        # to hold the count of missing values for each variable
        list_value_count=[]
        # to hold the percentage of missing values for each variable
        list_percent_count=[]
        #
        for col in col_missing_value:
            # add the column names into this list
            list_labels.append(col)
            # caculate each column missing value count and append to list
            list_value_count.append(df[col].isnull().sum())
            # df.shape[0] will give totoal row count
            list_percent_count.append(df[col].isnull().sum()/df.shape[0])


        # create two subplot with 1 row two columns
        fig, (ax1, ax2) = plt.subplots(1,2)
        # we use ax.barh() create horizontal bar chart, bar width is the count number of missing value
        ax1.barh(y = np.arange(len(list_labels)),width = np.array(list_value_count), height=0.5, color='blue')
        # set lenght of y labels
        ax1.set_yticks(np.arange(len(list_labels)))
        # set y labels
        ax1.set_yticklabels(list_labels, rotation='horizontal')
        ax1.set_xlabel('Count of missing values')
        ax1.set_title("Columns with missing value count")

        ax2.barh(y=np.arange(len(list_labels)), width = np.array(list_percent_count), height=0.5, color='red')
        ax2.set_yticks(np.arange(len(list_labels)))
        ax2.set_yticklabels(list_labels, rotation='horizontal')
        ax2.set_xlabel("Percentage of missing values")
        ax2.set_title("Columns with missing values")

        plt.show()
        # print(col_missing_value, type(col_missing_value))

        return col_missing_value


    def bcr_dryer(self, df_product, df_nj_weather, df_pa_weather):
        """
        This fucntion used for bar_chart_race library data reconstruction.
        This function will present the relationship between ['Dryer'] and time

        https://www.wikihow.com/Install-FFmpeg-on-Windows

        Args:
        -----
        df_product:pd.DataFrame
            We into

        Returns:
        ------
        df
        """
        # due to format unification, we need transform ['dt_date'] to its string type
        df_nj_weather.loc[:, ('str_date')] = df_nj_weather['dt_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_pa_weather.loc[:, ('str_date')] = df_pa_weather['dt_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        # direct merge on ['dt_est'] without split each row to several hours rows
        # because we only care about its accumualte results, so just extract combination of location and time
        df_nj = pd.merge(df_product, df_nj_weather, on=['dt_est'], how='left')
        df_pa = pd.merge(df_product, df_pa_weather, on=['dt_est'], how='left')


        # ***********This part logic have some problem, we need delete duplicate from weather first ************
        # in same day, it might have two weather condition, so when you merge data, it will appear duplicate records
        # we know it will be duplicate when we merge with product data because some duplicate weather data
        print('Before drop duplicate, {}, {}'.format(df_nj.shape, df_pa.shape))
        # ['StartDate'] is unipue, other columns more or less contain same records
        # we keep the first records and directly do this action on current DataFrame
        df_nj.drop_duplicates(subset=['StartDate'], keep='first', inplace=True)
        df_pa.drop_duplicates(subset=['StartDate'], keep='first', inplace=True)
        print('After drop duplicate, {}, {}'.format(df_nj.shape, df_pa.shape))


        # extract each dryer from df_product
        df_01 = df_nj.loc[df_nj['Dryer'] == 'Dryer 01']
        df_02 = df_nj.loc[df_nj['Dryer'] == 'Dryer 02']
        df_03 = df_nj.loc[df_nj['Dryer'] == 'Dryer 03']
        df_04 = df_nj.loc[df_nj['Dryer'] == 'Dryer 04']
        df_06 = df_pa.loc[df_pa['Dryer'] == 'Dryer 06']
        df_07 = df_pa.loc[df_pa['Dryer'] == 'Dryer 07']
        df_08 = df_pa.loc[df_pa['Dryer'] == 'Dryer 08']
        df_10 = df_pa.loc[df_pa['Dryer'] == 'Dryer 10']

        # in each day, there might be several orders in one dryer, so we need to sum one day data into one row
        series_01_cum = df_01.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_02_cum = df_02.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_03_cum = df_03.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_04_cum = df_04.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_06_cum = df_06.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_07_cum = df_07.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_08_cum = df_08.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()
        series_10_cum = df_10.groupby(by=['str_date'])['ActualDryQty'].sum().cumsum()

        # create a new time base DataFrame for next step merge with time range and type=datetime.time
        df_base = pd.DataFrame(data=pd.date_range(start='1/2/2016', end='12/31/2020'), columns=['dt'])
        # because next step we need to merge and using bar_race_chart librayr,
        # so we tranform date into same format and type and restore into new column ['str_date']
        df_base.loc[:, ('str_date')] = df_base['dt'].apply(lambda x: x.strftime('%Y-%m-%d'))
        # in brc() library, it use time (string type) as index
        df_base = df_base.set_index('str_date')
        # concatnate data by columns, so rows is time index from df_base with everyday for these 5 years
        # for each dryers, it might have someday don't work
        df_bcr = pd.concat([df_base, series_01_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_02_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_03_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_04_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_06_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_07_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_08_cum], axis=1)
        df_bcr = pd.concat([df_bcr, series_10_cum], axis=1)
        # only reserve dryers columns
        df_bcr.drop(labels=['dt'], axis=1, inplace=True)
        df_bcr.columns = ['Dryer 01', 'Dryer 02', 'Dryer 03', 'Dryer 04', 'Dryer 06', 'Dryer 07', 'Dryer 08',
                          'Dryer 10']
        # this is used to fullfill with contiuous data in days that dryers dont' have task
        df_bcr = df_bcr.interpolate()
        # full fill some early day, like 01/01/2016, that interpolate() can't full fill
        df_bcr.fillna(0, inplace=True)
        # it's more easy to read file and start processing to movies
        df_bcr.to_csv('03_data/27_bcr.csv', index=True)

        # because each row is already start from on day, so we can directly use ['ActualDrayQty'] to cumulate sum
        #     df_01.loc[:,('cumsum')] = df_01['ActualDryQty'].cumsum()
        #     df_02.loc[:,('cumsum')] = df_02['ActualDryQty'].cumsum()

        #     df_20 = pd.DataFrame(data=[df_14['dt_date'], df_14['cumsum']])
        #     df_21 = df_20.T
        #     df_21['str_date'] = df_21['dt_date'].apply(lambda x:x.strftime('%Y-%m-%d'))
        #     df_22 = df_21.set_index('str_date')

        #     # merge
        #     df_bcr = pd.merge(df_base, df_01, on=['str_date'], how='left')
        #     df_bcr = pd.merge(df_bc)

        # create a new DataFrame for  bar_race_chart
        #     df_bcr = pd.DataFrame()

        return df_base, df_bcr

    def squrify_treemap(self, df, col_name='ProdLine', bool_groupby = False):
        """
        Args:
        ------
        df:pd.DataFrame
            data source
        col_name:string
            the column you want to plot
        bool_groupby:boolean
            if bool_groupby = True, we need to groupby this dataframe, If False, we can directly use dataframe

        :return:
        """
        if bool_groupby:
            # create a new dataframe with groupby list and its corresponding count size
            df_map = df.groupby(by=[col_name]).size().reset_index(name='counts')
        else:
            df_map = df.copy()
        # create labels for squrify demonstration
        # defalut apply() direction is vertical, we push a column series to apply() function and default axis=0
        # In default, x will be two column features and x[0] will be the first element for this two column features
        # which are chemicl and 229(size/count of chemical)
        # So when you change axis=1, the output x become row direction output, x[0] will become all name of first column
        # Lable will be category name + its groupby size
        series_labels = df_map.apply(lambda x : str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
        # squrify library need into the box size with list format
        list_sizes = df_map.iloc[:,1].values.tolist()
        # create colors by length of existing labels, for example, we have 8 category, Spectral will provide 8 colors
        list_colors = [plt.cm.Spectral( i/float(len(series_labels))) for i in range(len(series_labels))]
        # set the plot size, weight=20 is just fit jupter notebook weight
        plt.figure(figsize = (20,8), dpi=80)
        squarify.plot(sizes = list_sizes, label = series_labels, color = list_colors, alpha=1.0)
        #title
        plt.title("Treemap of")
        plt.axis('off')
        plt.show()

        return None



