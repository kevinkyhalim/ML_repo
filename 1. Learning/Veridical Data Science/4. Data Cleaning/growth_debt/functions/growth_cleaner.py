# define the function and add the following variables
# data: the debt data frame
# year_start: to find the column that corresponds to the year_start
# year_end: to find the column that corresponds to the year_end
import numpy as np 
import pandas as pd
def growth_cleaner(data, year_start, year_end):

    # Change the column heading to str data type
    data.columns = data.columns.astype(str)

    data.rename(columns={'Country Name': 'country'}, inplace=True)

    # Pick only the 1st column, then the columns from year_start to year_end
    growth_clean = data.iloc[:, 
                                 np.r_[0, data.columns.get_loc(str(year_start)):
                                       data.columns.get_loc(str(year_end))+1]]

    growth_clean.iloc[:,0] = growth_clean['country'].replace({'China' : "China, People's Republic of",
                                        'Congo, Dem. Rep.': 'Congo, Dem. Rep. of the',
                                        'Congo, Rep.' : 'Congo, Republic of',
                                        'Czechia' : 'Czech Republic',
                                        "Cote d'Ivoire" : "Côte d'Ivoire",
                                        'Egypt, Arab Rep.' : 'Egypt',
                                        'Hong Kong SAR, China' : 'Hong Kong SAR',
                                        'Iran, Islamic Rep.' : 'Iran',
                                        'Korea, Rep.' : 'Korea, Republic of',
                                        'Lao PDR' : 'Lao P.D.R.',
                                        'Micronesia, Fed. Sts.' : 'Micronesia, Fed. States of',
                                        'North Macedonia' : 'North Macedonia ',
                                        'St. Kitts and Nevis' : 'Saint Kitts and Nevis',
                                        'St. Lucia' : 'Saint Lucia',
                                        'St. Vincent and the Grenadines' : 'Saint Vincent and the Grenadines',
                                        'South Sudan' : 'South Sudan, Republic of',
                                        'Syrian Arab Republic' : 'Syria',
                                        'Sao Tome and Principe' : 'São Tomé and Príncipe',
                                        'Turkiye' : 'Turkey',
                                        "Venezuela, RB" : 'Venezuela',
                                        "Yemen, Rep." : 'Yemen'})

    return growth_clean
