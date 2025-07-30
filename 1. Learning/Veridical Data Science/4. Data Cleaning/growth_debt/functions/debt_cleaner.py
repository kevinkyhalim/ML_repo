# define the function and add the following variables
# data: the debt data frame
# year_start: to find the column that corresponds to the year_start
# year_end: to find the column that corresponds to the year_end
import numpy as np 
def debt_cleaner(data, year_start, year_end):

    # Change all values of "no data" to NaN
    replace_dict = {'no data': np.nan}
    debt_clean = data.loc[:,:].replace(replace_dict)

    # Change the column name of DEBT (% of GDP) to Country Name
    debt_clean.rename(columns={'DEBT (% of GDP)': 'country'}, inplace=True)

    # Change the column heading to str data type
    debt_clean.columns = debt_clean.columns.astype(str)

    # Take out the first row of the data
    debt_clean = debt_clean.iloc[1:]

    # Pick only the 1st column, then the columns from year_start to year_end
    debt_clean = debt_clean.iloc[:, 
                                 np.r_[0, debt_clean.columns.get_loc(str(year_start)):
                                       debt_clean.columns.get_loc(str(year_end))+1]]

    return debt_clean
