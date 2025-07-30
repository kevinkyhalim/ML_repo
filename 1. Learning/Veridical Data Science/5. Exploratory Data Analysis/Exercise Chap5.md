<h1> True or False Exercises </h1>
For each question, specify whether the answer is true or false (briefly justify your answers).
<ul>
<li>You should evaluate the predictability and stability of every explanatory finding that you present.</li>
Answer: TRUE.
<br>

<li>Not every exploratory finding needs to be turned into an explanatory figure.</li>
Answer: TRUE, we do not always need to present a polished graph for every exploratory step we do.
<br>

<li>Exploratory and explanatory data analyses can be numeric summaries (i.e., are not only data visualizations).</li>
Answer: TRUE, there is no hard rule that says exploratory and explanatory data analysis must be presented as a graph.
<br>

<li>It is a good idea to add color to a plot, even when the color does not convey information.</li>
Answer: FALSE, adding color when it does not convey information can confuse the audience of the intended message.
<br>

<li>The correlation between two numeric variables describes the angle of the linear relationship in a scatterplot.</li>
Answer: FALSE, correlation describes the tightness of the points around a line that can be at any angle.
<br>

<li>Correlation only quantifies the linear component of the relationship between two variables.</li>
Answer: TRUE
<br>

<li>The mean is a better summary of a typical value than the median when there are outliers present.</li>
Answer: FALSE, whether to use the mean or the median is context dependent. The mean will be more influenced by the outliers, which may be better in some scenarios, but not others.<br>

<li>When deciding between multiple possible versions of an explanatory figure, you should always choose the one that conveys more information (i.e., is more detailed).</li>
Answer: FALSE, not necessarily, the graph that conveys the information that we want should be picked
<br>

<li>If one variable causes changes in the other, then the two variables will be correlated.</li>
Answer: TRUE
<br>

<li>If one variable is correlated with another, then changes in one will cause changes in the other.</li>
Answer: FALSE, remember that correlation does not necessarily imply causation. There may be another variable that simultaneously and independently affects both variables.
<br>

</ul>
<h1>Conceptual Exercises</h1>
<ul>
<li>Explain how you could use a scatterplot to visualize two numeric variables and a categorical variable simultaneously.</li>
Answer: We can create a dot on the X,Y plan by assigning one of the numeric variables under the X axis, and the other numeric variable under the Y axis, and then color code the points based on the categorical variable.
<br>

<li>What is “overplotting”? Describe two techniques for reducing overplotting in a figure.</li>
Answer: The act of plotting all information in our data which results in no specific results / conclusion to be seen in the plot. Some techniques to reduce overplotting is to either increase the transparency of the data, or color code some of the data, especially those that we want the audience to focus their attention to or we can also filter out some information to only focus on specific values.
<br>

<li>Explain the difference between an exploratory and an explanatory plot.</li>
Answer: Exploratory refers to the act of further understanding the trends / patterns that exists in the dataset. While explanatory refers to the act of producing a polished figure to communicate the exploratory findings to an external audience.
<br>

<li>What is the relationship between correlation and covariance? Explain why the correlation provides a more comparable summary of the linear relationship between two numeric variables than the covariance.</li>
Answer: Correlation is basically a normalized value of the covariance, as it is calculated by dividing the covariance between 2 variables with the standard deviation of each of the variable.
<br>

<li>Describe two ways that you could demonstrate the predictability of a conclusion drawn from a data visualization.</li>
Answer:
1. Find external data that can validate our finding;
2. Conduct a literature search if it can validate our findings;
<br>

<li>This question asks you to identify an appropriate visualization technique for visualizing various relationships in the organ donation data. You may want to use the data visualization flowchart in Figure 5.1 to identify possible visualization types, but don’t feel constrained by it (keep in mind that there may be several appropriate visualization choices). To help recall what the data looks like, we have printed the relevant variables for Australia, Italy, and Germany from 2014–2017 in the following table (but your visualizations should be based on the entire dataset).</li>
<ul>
<li>What visualization technique would you use to visualize the number of countries in each region? Draw a brief sketch of what your visualization would look like or create it in R/Python.</li>
Answer: I will use a bar graph to compare the number of countries under each region
<br>
<li>What visualization technique would you use to visualize the total number of organ donations in 2017 for each region? Draw a brief sketch of what your visualization would look like or create it in R/Python.</li>
Answer: I would try and use a bar graph to compare between regions
<br>

<li>What visualization technique would you use to visualize the number of organ donations over time in Europe? Draw a brief sketch of what your visualization would look like or create it in R/Python.</li>
Answer: Line graph, with color depicting the different countries in Europ
<br>
| Country    | Region          | Year | Population | Total Deceased Donors (Imputed) |
|------------|-----------------|------|------------|---------------------------------|
| Australia  | Western Pacific | 2014 | 23,600,000 | 378                             |
| Australia  | Western Pacific | 2015 | 24,000,000 | 435                             |
| Australia  | Western Pacific | 2016 | 24,300,000 | 503                             |
| Australia  | Western Pacific | 2017 | 24,500,000 | 510                             |
| Germany    | Europe          | 2014 | 82,700,000 | 864                             |
| Germany    | Europe          | 2015 | 80,700,000 | 877                             |
| Germany    | Europe          | 2016 | 80,700,000 | 857                             |
| Germany    | Europe          | 2017 | 82,100,000 | 797                             |
| Italy      | Europe          | 2014 | 61,100,000 | 1384                            |
| Italy      | Europe          | 2015 | 59,800,000 | 1369                            |
| Italy      | Europe          | 2016 | 59,800,000 | 1478                            |
| Italy      | Europe          | 2017 | 59,400,000 | 1714                            |

</ul>

<li>The following set of two boxplots aim to compare the distribution of the 2017 organ donation rates between the American and European countries.</li>
<ul>
<li>It is not specified whether these boxplots are representing the imputed or unimputed donor counts. Which variable do you think makes more sense for this comparison? Why?</li>
Answer: I believe this is the data without imputation as seen in the very median value for the American region (near 0)
<br>
<li>Do you think that these two boxplots are comparable? If not, how would you modify them so that they are?</li>
Answer: I think a better way to compare these two boxplots is to compare the donor rate rather than the absolute numbers as bigger population countries will naturally have higher population, and therefore donation number. 
<br>
</ul>
<img src="https://vdsbook.com/05-data_viz_files/figure-html/unnamed-chunk-25-1.png", width = 500>

<li>The following plot shows a stability analysis of an EDA finding related to the increase in the total number of organ donations worldwide over time. This analysis shows how this result changes based on the preprocessing imputation judgment call options that we introduced in Chapter 4 (no imputation, average imputation, and previous imputation).</li>
<ul>
<li>Based on this plot, do you feel that the global organ donation trend result is stable to the choice of imputation judgment call?</li>
Answer: In general, I believe that all imputation method shows a an ever increasing trend from early 2000 to 2015. However, the average imputation method seems to be less variable than those done through "None" and "Previous"
<br>
<li>Do you think that all three of these choices are reasonable alternative judgment calls? If not, which do you think is the best choice?</li>
Answer: I believe the "Average" imputation method is more reasonable in this case as it seems improbable that organ donations can jump at a very high rate within less than 2-3 years, seen in the jump from early 2000 to 2004 from the imputation method of "None" and "Previous"
<br>
</ul>
<img src="https://vdsbook.com/05-data_viz_files/figure-html/line-global-impute-1.png", width = 500>

<li>The plot below is an exploratory figure that demonstrates that the proportion of organ donations that were utilized (i.e., were actually transplanted) is higher in the US than in Spain. Describe how you would create a polished explanatory figure that conveys this message as clearly and effectively as possible. Explicitly state your target audience. Get creative: your explanatory figure does not have to simply be a polished version of this plot (but it can be if you’d like). For a challenge, create your explanatory figure in R or Python.</li>
Answer: I would use a line graph to compare the proportion of utilized organ donations between US and Spain. Using a line graph will ensure that it will be easy and immediate to see that US numbers are always higher than Spain's proportions. I would also annotate the line graph to have both US and Spain at the right end of each line graph.
<br>
<img src="https://vdsbook.com/05-data_viz_files/figure-html/unnamed-chunk-27-1.png", width = 500>

<li>Here, we present two potential visualizations of organ donation rates by country in 2017 using (a) a map (a visualization type that does not feature in the flowchart), and (b) a bar chart.</li>
<ul>
<li>List at least two pros and cons of each visualization option for conveying this information.</li>
Pro:
1. Map
- The audience can easily see regions where there are high donations based on how dark the region are
- A visual representation may also imply certain geographical correlations where it is easy to see whether nearby locations are more correlated than those who are located far away (aka maybe similar geographies used the same policy and measure)
2. Bar
- It is easy to see the top countries amount of organ donations / country 
- It is also easy to compare the numbers between each country
<li>Which chart would you choose for presenting in a pamphlet that will be distributed to the public? Justify your choice.</li>
Answer: Map. As the public doesn't really need to know the specific details, a quick summary of total donor distribution around the world should suffice.
<br>
<li>Which chart would you choose for presenting in a technical report that will be reviewed by domain experts? Justify your choice.</li>
Answer: Bar. As domain experts can understand better why certain countries may have high / low donor number (or what even constitues as a high or low number), I believe that a visualization with more details will be helpful for domain experts.
<br>
<li>For the map in panel (a), identify one data visualization judgment call that we made when producing this plot that may have an impact on its takeaway message.</li>
Answer: I would say that the fact that we are using a Mercator map, where countries located nearer to the equator are more emphasized, it now seems to indicate that African countries seem to have very low donor number, compared to the focus of "Spain" being the country with the highest donor numbers.x
<br>
</ul>
<img src="https://vdsbook.com/05-data_viz_files/figure-html/donor-map-1.png", width = 500>


</ul>
<h1>Mathematical Exercises</h1>
Suppose that two variables,  and , have an approximately linear relationship that can be summarized mathematically as .

Draw the line  on a blank plot with axes  and  (either manually using pen and paper or using R or Python).

Based on this linear relationship, how much does  increase when ’s value is increased by ? (Hint: try replacing  with  in the summary linear relationship given in this question.)

If we log-transform both  and  so that our relationship is now , how much (in terms of a percentage of its original value) does  increase when  is increased by 1 percent (i.e., when  becomes )?

You are given the following 12 numbers:

Compute the mean and the median of  (manually).

Add the number 25 to the collection of numbers (so there are now 13 numbers). Recompute the mean and median. Comment on how much each value changed.

Compute the variance and standard deviation of  (manually). Note that most programmed functions will use a version of the formula for the variance that has a denominator of  instead of .

You are given another set of 12 numbers:

Compute the mean and the median of  (manually).

Compute the variance and standard deviation of  (manually). Note that most programmed functions will use a version of the formula for the variance that has a denominator of  instead of .

Create a scatterplot of  (the 12 numbers from the previous question) against  (the set of 12 numbers from this question) either manually using pen and paper or using R or Python.

Based on the scatterplot you created, what kind of correlation do you expect to find between  and ?

Compute the correlation and the covariance between  and  (we recommend doing this computation manually). Comment on how you interpret your results.

Prove the following relationship between the correlation and covariance:
 

<h1>Coding Exercises</h1>
<ul>
<li>A dot plot is a visualization technique for comparing the values of comparable numeric variables (such as the donation rates for each organ) across a grouping variable (such as country).</li>

<ul>
<li>Although there is no inbuilt function in the ggplot2 R library or in Python for creating dot plots, a dot plot can be produced with the creative use of the geom_point() and geom_line() ggplot2 R functions (or equivalent scatterplot functions in Python, such as the px.scatter() plotly function). Use these functions to create your own version of the following dot plot that summarizes the donation rates of kidneys, livers, lungs, and hearts for Spain and the US in 2017. The code that creates the data that underlies this plot can be found at the end of the 02_eda.qmd (or .ipynb) file in the organ_donations/dslc_documentation/ subfolder of the supplementary GitHub repository.</li>

</li>Summarize a take-away message for the dot plot.</li>

</ul>

<li>For this question, you will conduct your own EDA in R or Python for the organ donor project based on the exploratory prompt: “Is there a difference in deceased donor type (i.e., whether the organs come from brain death or circulatory death donors) across different countries?” The relevant variables in the preprocessed organ donation data will be deceased_donors_brain_death and deceased_donors_circulatory_death. You can conduct your analysis in the relevant section of the online EDA code can be found in the 02_eda.qmd (or .ipynb) file in the organ_donations/dslc_documentation/ subfolder of the supplementary GitHub repository.</li>

<ul>
<li>Run the organ donations EDA code in 02_eda.qmd (or .ipynb).</li>

<li>To conduct your EDA to answer the question (“Is there a difference in deceased donor type (i.e., whether the organs come from brain death or circulatory death donors) across different countries?”), you may want to just focus on a few countries that are of interest to you (easier) or you may want to look at all countries (harder). You may also want to focus on the data from one particular year (easier) or include all the years in your analysis (harder). Since these variables contain many missing values, you may want to impute them or restrict your analysis to just a subset of countries and years for which the data is not missing. Be sure to document any judgment calls that you make and justify your choices.</li>

<li>Conduct a PCS evaluation of at least one of your findings.</li>

<li>Create at least one polished explanatory plot based on your exploratory findings. Be sure to choose a clear take-away message, and ensure that your explanatory plot highlights this message.</li> 
</ul>

<li>For the organ donation project, add at least one additional exploration section to the 02_eda.qmd (or .ipynb) file in the organ_donations/dslc_documentation/ subfolder of the supplementary GitHub repository based on a question that you come up with on your own, conduct a PCS evaluation of your findings, and create at least one polished explanatory visualization based on what you find.</li>
</ul>

<h1>EDA Project</h1>
Re-evaluating growth in the time of debt This project continues with the “growth in the time of debt” project from Chapter 4 in which you cleaned the historical public debt data downloaded from the International Monetary Fund (IMF) and the gross domestic product (GDP) growth data from the World Bank. The data and template files can be found in the exercises/growth_debt/ folder in the supplementary GitHub repository.

Create a new DSLC code file in which to conduct an EDA based on a project goal similar to Reinhart and Rogoff’s original study, which aims to identify whether higher debt is associated with lower economic growth.

Conduct a PCS evaluation of your findings.

Produce at least one explanatory figure communicating an interesting finding from the data to an audience of your choosing.