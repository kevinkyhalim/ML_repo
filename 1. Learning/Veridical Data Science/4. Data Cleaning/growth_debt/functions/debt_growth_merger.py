# define the function and add the following variables
# debt data: the debt data frame
# growth data: to find the column that corresponds to the year_start
import numpy as np 
import pandas as pd
def debt_growth_merger(debt, growth):

    # Unpivoted debt
    debt_unpivot = pd.melt(debt, id_vars=debt.columns[0], 
                        value_vars=debt.columns[1:], 
                        var_name='year', 
                        value_name='debt-pct-gdp').sort_values(by=['country', 'year']).reset_index(drop=True)

    # Unpivot growth data
    growth_unpivot = pd.melt(growth, id_vars=growth.columns[0], 
                        value_vars=growth.columns[1:], 
                        var_name='year', 
                        value_name='growth-pct-gdp').sort_values(by=['country', 'year']).reset_index(drop=True)

    # Merge the two dataframes
    merged_data = pd.merge(debt_unpivot, growth_unpivot, on=['country', 'year'])
    
    return merged_data
