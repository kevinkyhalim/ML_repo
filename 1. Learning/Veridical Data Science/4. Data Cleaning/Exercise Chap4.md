<h1>Exercises</h1>

<h2>True or False Exercises</h2>
For each question, specify whether the answer is true or false (briefly justify your answers).
<ul>
<li>Data cleaning is an optional part of the DSLC.</li>
Answer: FALSE, it is quite an integral part of DSLC and every dataset should be determined if it needs to be cleaned.

<br>
<li>You should avoid modifying the original data file itself; instead, you should try to modify it only within your programming environment.</li>
Answer: TRUE, this is correct as keeping the original data as is ensures the reproducibility of analysis in the future.

<br>
<li>A clean dataset can contain missing values.</li>
Answer: TRUE, data cleaning is not necessarily the same as satisfying the formatting requirements of specific algorithms that may not be able to accept missing values. It can also be "clean" as long as it is properly formatted.

<br>
<li>A preprocessed dataset can contain missing values.</li>
Answer: TRUE, however it will depend on the algorithm that will be used later on and whether missing values can or cannot be processed using the algorithm.

<br>
<li>The best way to handle missing values is to remove the rows with missing values.</li>
Answer: FALSE, there are other ways to handle missing values such as imputing using existing data. In addition, this method will not work well if there are many rows with missing values and will result in having very little data to work with. Also, doing this will introduce bias to the data.

<br>
<li>The set of valid values for a variable is usually determined based on domain knowledge and how the data was collected.</li>
Answer: TRUE.

<br>
<li>An unusual or surprising value should be treated as invalid.</li>
Answer: FALSE, as long as the value is deemed possible within the domain and is possible based on how the data is collected then it should not be necessarily treated as invalid.

<br>
<li>There is a single correct way to clean every dataset.</li>
Answer: FALSE, there are many ways to clean every dataset (e.g. Whether a population value should be log transformed or not)

<br>
<li>It is generally recommended that you split your data into training, validation, and test sets before cleaning it.</li>
Answer: TRUE, because the training data set will inform the nature of cleaning / transformation to be done on the validation and test set.

<br>
<li>You should avoid exploring your data before you clean it.</li>
Answer: FALSE, exploring the data is part of cleaning the data!

<br>
<li>You must write separate cleaning and preprocessing functions.</li>
Answer: FALSE, both cleaning and preprocessing functions can be done in the same function.

<br>
<li>A different preprocessing function is needed for each algorithm that you apply.</li>
Answer: FALSE, different arguments to create different versions of the data can be done depending on the algorithm's formatting requirements.
</ul>

<h2>Conceptual Exercises</h2>
<ul>
<li>What do you think might be a possible cause of the missing values in the organ donation dataset discussed in this chapter?</li>
Answer: It could be that countries don't really have a method to properly count the numbers of donors or only sporadically collect numbers when needed, hence for the years in between there are no data.

<br>
<li>How can writing a data cleaning/preprocessing function help with conducting a stability analysis that assesses whether the data cleaning/preprocessing judgment calls affect the downstream results?</li>
Answer: Writing functions to help with data cleaning and preprocessing can quickly help in creating datasets with different methods of cleaning &/ preprocessing which can then be run through the same analysis as existing method to quickly compare the difference in results, if any, and proves the stability of the analysis to the cleaning &/ preprocessing judgement.

<br>
<li>Why it is not a good idea to try to automate the process of data cleaning (i.e., to create a general one-size-fits-all set of data cleaning tasks)?</li>
Answer: Since different data sets have different types of data and requires specific domain knowledge, having a non-automated data cleaning process ensures that each cleaning process adheres to the proper domain of each data set.

<br>
<li>What is the difference between data cleaning and preprocessing?</li>
Answer: Data cleaning involves reducing ambiguities in the data and converting it to a more usable (e.g., “tidy”) format. On the other hand, preprocessing ensure that the data is prepared for the application of a specific algorithm or analysis.

<br>
<li>Suppose that your job is to prepare a daily summary of the air quality in your area. To prepare your report each day, you consider the set of air quality measurements collected each day at 8 a.m. from a set of 10 sensors, all within a 20-mile radius of one another. Today, nine of the 10 sensors reported an Air Quality Index (AQI) within the range of 172–195 (very unhealthy), but one of the sensors reported an AQI of 33 (good).</li>
<ul>
<li>Would you consider this individual sensor’s measurement to be invalid? Is there any additional information that might help you answer this question?</li>
Answer: Not necessarily, as there might be something happening near one of the sensors that reported a good quality. We can either see the history of the past measurement by the sensor, inspect the sensor, check the local news / survey the area of the seemingly faulty sensor to further judge whether this value is valid or not.

<br>
<li>If you learned the sensor was broken, what, if any, action items would you introduce to handle this data point? Justify your choice.</li>
Answer: In the absent of any other news that might change my perception of the possible air quality level at the faulty sensor location, I would most likely use the average of the 2-3 nearest sensor to the faulty sensor.

<br>
</ul>
<li>The following two line plots show the total number of organ donations worldwide (computed as the sum of the donor counts from all countries) by year based on (a) the original version of the organ donation data where we have not completed the data or imputed the missing donor counts; and (b) a preprocessed version of the data where we have both completed the data (by adding in the missing rows) and imputed the missing donor counts (using the “average” imputation method described where we replace a country’s missing donor count with the average of the previous and the next non-missing donor counts for that country).</li>
<ul>
<li>Explain why preprocessing the data by completing it and imputing the missing donor counts changes the result shown in the figure.</li>
Answer:

<br>
<li>How does the takeaway message of plot (a) differ from plot (b)? Discuss the cause of these differences.</li>
Answer:

<br>
<li>Which version, (a) or (b), do you think is a better reflection of reality?</li>
Answer:

<br>
</ul>
<img src="https://vdsbook.com/04-data_cleaning_files/figure-html/unnamed-chunk-14-1.png", width=300>
<li>The histogram here shows the distribution of the reported age of diabetes diagnosis of a random sample of American diabetic adults collected in an annual health survey, called the National Health and Nutrition Examination Survey (NHANES), conducted by the Centers for Disease Control and Prevention (CDC).</li>
<ul>
<li>Identify two strange or surprising trends in this histogram. What do you think is causing these trends?</li>
Answer:

<br>
<li>Describe any data cleaning action items you might create to address them.</li>
Answer:

<br>
</ul>
<img src="https://vdsbook.com/04-data_cleaning_files/figure-html/unnamed-chunk-15-1.png", width = 500, height = 250>
<li>The table here shows a subset of the data on government spending on climate and energy, with the type of each column printed beneath the column name.</li>
<ul>
<li>What are the observational units?</li>
Answer: Most likely spending in Million USD

<br>
<li>Is the data in a tidy format?</li>
Answer: Not really, while the type is the correct data type, year could be under a datetime format while the spending can be converted to numeric by taking out the double aposthropes.

<br>
<li>Mentally walk through the data cleaning explorations for this data and identify at least two action items that you might want to implement to clean this data. Document any judgment calls that you make and note any alternative judgment calls that you could have made.</li>
Answer: As mentioned above, change the year column to a date time while the spending to be in numeric, specifically in million dollars. In addition, we can also pivot the table by having the "Type" values as columns so the table will be comprised of 3 columns, where it indicates the year, spending for energy and spending for climate for every year.

<br>
<li>Create one possible clean version of this dataset (you can just manually write down the table—that is, you don’t need to use a computer, but you can if you want to).</li>
Answer:

<br>

</ul>

| Year (numeric) | Type (categorical) | Spending (character) |
|----------------|--------------------|----------------------|
| 2013           | Energy             | "$15,616"            |
| 2013           | Climate            | "$2,501"             |
| 2014           | Energy             | "$17,113"            |
| 2014           | Climate            | "$2,538"             |
| 2015           | Energy             | "$19,378"            |
| 2015           | Climate            | "$2,524"             |
| 2016           | Energy             | "$20,134"            |
| 2016           | Climate            | "$2,609"             |
| 2017           | Energy             | "$19,595"            |
| 2017           | Climate            | "$2,800"             |

</ul>
<h2>Reading Exercises</h2>
Read “Tidy Data” by Hadley Wickham (2014) (available in the exercises/reading folder on the supplementary GitHub repository).

<h2>Project Exercises</h2>
<ul>
<li>[Non-coding project exercise] IMAGENET is a popular public image dataset commonly used to train image-based ML algorithms (such as algorithms that can detect dogs and cats in images). The goal of this question is to guide you through conducting Step 1 (Learning about the data collection process and the problem domain) of the data cleaning procedure described in this chapter using the information provided on the IMAGENET website and the Kaggle website. Note that the dataset, which is available on Kaggle, is more than 100 GB–you do not need to download it to answer this question.</li>
<ul>
<li>How was the data collected (e.g., how did they find the images and decide which ones to include)?</li>

<li>How have the images in the dataset been cleaned or reformatted?</li>

<li>How would you plan to assess the predictability of an analysis conducted on this data (e.g., do you have access to external data? Would you split the data into training, validation, or test sets? Would you use domain knowledge?)</li>

<li>Write a summary of how easy it was to find this information, any other relevant information that you learned, and any concerns you have about using algorithms that are based on this dataset for real-world decision making.</li>
</ul>
<li>Re-evaluating growth in the time of debt For this project, you are placing yourself in Thomas Herndon’s shoes. Your goal is to try and reproduce the findings of Reinhart and Rogoff’s study (introduced in Chapter 3) using historical public debt data (debt.xls) downloaded from the International Monetary Fund (IMF), as well as gross domestic product (GDP) growth data (growth.csv) downloaded from the World Bank.

These data files can be found in the exercises/growth_debt/data/ folder in the supplementary GitHub repository.

The file debt.xls contains the IMF data on the annual debt as a percentage of GDP data from 1800 to 2015 (a total of 216 years) for 189 countries.

The file growth.csv contains the World Bank data from 1960 through to 2021 (a total of 62 years) for 266 countries.

A template for the 01_cleaning.qmd file for R (or equivalent .ipynb files for Python) has been provided in the relevant exercises/growth_debt/ folder in the supplementary GitHub repository.
</li>

<ul>
<li>Fill in the template 01_cleaning.qmd (or equivalent .ipynb) file by implementing steps 1-3 of the data cleaning process outlined in this chapter. We recommend filtering the debt.xls data to 1960 and beyond and exploring the two data files (debt.xls and growth.csv) separately.
</li>

<li>To implement step 4 of the data cleaning process, write a separate cleaning function for each dataset. Your debt-cleaning function should create a clean debt dataset whose rows for Australia from 1960–1969 look like the left table below, and your growth-cleaning function should create a clean growth dataset whose rows for Australia from 1960–1969 look like the right table below:</li>

| Country   | Year | Debt (% of GDP) | Growth (% of GDP) | Year | Debt (% of GDP) | Growth (% of GDP) |
|-----------|------|-----------------|-------------------|------|-----------------|-------------------|
| Australia | 1960 | 31.47           | NA                | 1965 | NA              | 5.980893          |
| Australia | 1961 | 30.31           | 2.483271          | 1966 | 41.23           | 2.381966          |
| Australia | 1962 | 30.42           | 1.294468          | 1967 | 39.25           | 6.303650          |
| Australia | 1963 | 29.32           | 6.214949          | 1968 | 38.21           | 5.095103          |
| Australia | 1964 | 27.65           | 6.978540          | 1969 | 35.73           | 7.043526          |
<li>Write a pre-processing function that takes as its input the two clean datasets and then joins them together by country and year. The rows for Australia from 1960–1969 in your single combined dataset should look like the following table. Note that the country names are not identical across each dataset, so you will need to decide how to handle country names that do not match (e.g., you might want to manually modify those that are spelled differently and remove those that are not actual countries).</li>

| Country   | Year | Debt (% of GDP) | Growth (% of GDP) |
|-----------|------|-----------------|-------------------|
| Australia | 1960 | 31.47           | NA                |
| Australia | 1961 | 30.31           | 2.483271          |
| Australia | 1962 | 30.42           | 1.294468          |
| Australia | 1963 | 29.32           | 6.214949          |
| Australia | 1964 | 27.65           | 6.978540          |
| Australia | 1965 | NA              | 5.980893          |
| Australia | 1966 | 41.23           | 2.381966          |
| Australia | 1967 | 39.25           | 6.303650          |
| Australia | 1968 | 38.21           | 5.095103          |
| Australia | 1969 | 35.73           | 7.043526          |

</ul>
