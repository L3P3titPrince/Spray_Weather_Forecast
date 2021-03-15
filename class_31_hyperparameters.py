class HyperParamters(object):
    """
    This class will be used to transmit hyperparameters between class.parameters
    Most of class can inherit this class and its hyperparameters

    ########################__Rules__##################################
    class file name = class + _ + number of sequence + function name
    class name = ClassType (Camel-Case)
    function name = function_name
    variable name = attribute_type_describe (Hungarian notation) # sometime I don't use attribute
    constant = UPPERCASE
    ###########################################################################


    ########################__Notation__############################################
    1. We might need split data into two place NJ and PA, but we also can try to merge two location into one,
    because they are not far away
    2. clean data, transform date format, join by date, Coeffience anaylsis
    3. Check kaggle format
    4. Check journals
    5. Traditional way is to find relationship between total running time with other weather features
    6. But we need to consider equipment service lift or depreciation will affect running time
    7. Sometimes, we don't use running time itself, we can use the statiscal version of this data
    For example, runing time - avg , variance of each running time data point
    8.
    #############################################################################

    1.change import data path


    There is a little space in class_34_proprocess, we elimiate some error record by mannully,
    This is not general proceedure, we can comments that part of code any time
    """

    def __init__(self):
        """:arg


        """
        self.TEST = 1

        # you can change this root path in this class and import_data() function will search from this root dictionary
        self.ROOTPATH = 'D:\\OneDrive\\03_Academic\\23_Github\\20_Stevens\\66-MGT-809-A\\03_data'


        # assign the column will be dropped in production table
        # we believe these columns are objective result or not useful features
        # ['Bulk Density'] is too objective
        # ['Moisture'] is decided by custome requirement, that might affect drying time
        # ['Hygroscopicity'] imbalance data distribution
        self.PRODUCT_DROP = ['Bulk Density', 'Moisture','Flow', 'Hygroscopicity']


        # these are the un-related columns in weather data
        # ['timezone'] will change from -18000 to -14400 because winter time to summer time, vsia. We can delete it
        self.WEATHER_DROP = ['dt_iso', 'timezone', 'city_name', 'lat', 'lon']
        # These columns are non-numerical data, if we need use these columns, we can add them back,
        # for now, I will delete them in this hyperparameters
        # self.WEATHER_DROP = ['dt_iso', 'timezone', 'city_name', 'lat', 'lon',
        #                      'weather_id', 'weather_main', 'weather_description', 'weather_icon']


        # sometimes operater repack powder back to dryer again because it's not meet particular size requirement
        # we use z-score as our defination, If we set threshold as 3, the yeild great than 130% will be eliminate
        # we also have a np.abs() for smaller outliers, which also affected by this threshold
        # when we set our threshold to 3, we eliminate the percentage great then 130%,
        # set to 1.5, then delete precentage greater than 111% (106 rows was dropped)
        # 3 - 130% /  1.5 - 111%   / 1 - 104.6%
        self.YEILD_THRESHOLD = 1.5
        # ['Rate'] is our target output, we more concens this colunm's statistcal results.
        # Typically rate can be as high as possible, but sometimes, the extramly high rate is operation error
        # We will try to identify ourlier by z-socre and Interquartile Range method
        # If we shrink our threshold to a small number, we will delete more outliers. But they might not be delete so early
        # For example, in index ['821', '2983','2999'] after manually double chek, they are operation error.
        # We can't do this to every ['Rate'] outliers. So we can do this drop process only for statiscal visulzation
        self.IQR_THRESHOLD = 2

