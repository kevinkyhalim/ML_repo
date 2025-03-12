<h1> Exercises </h1>

<h2>True or False Exercises </h2>
For each question, specify whether the answer is true or false (briefly justify your answers).
<ul>
<li>There are many ways to formulate a data science problem.</li>
Answer: TRUE, depending on the goal of the analysis, the same context can be interpreted in many different ways.

<br>
<li>PCS evaluations are necessary only when you will be presenting your results to an external audience.</li>
Answer: FALSE, it will especially be important when applying the model in the real world as PCS evaluation ensures that the model is robust and stable.

<br>
<li>Once you have moved on to a new stage of the DSLC, it is OK to return to and update previous stages.</li>
Answer: TRUE, the data science life cycle should be seen as an iterative method to test out different hypothesis.

<br>
<li>There is only one correct way to conduct every data analysis.</li>
Answer: FALSE, there are many to conduct a data analysis as different data scientist / analyst might have different hypothesis that they want to test.

<li>Every data science project must progress through every stage of the DSLC.</li>
Answer: FALSE, not necessarily, some can skip the steps of data prediction & / exploration of intrinsic data structures, depending on the scope and need of the project.

<br>
<li>The judgment calls that you make during data cleaning can have an impact on your downstream results.</li>
Answer:TRUE, as assumptions during exploratory data analysis can have impact in terms of the type of answers / models that will be used.

<br>
<li>PCS evaluations can help prevent data snooping.</li>
Answer: TRUE, as PCS framework ensures that the relationship found during the analysis has to be replicable and stable in future data set, avoiding the issue of data snooping.

<br>
<li>PCS uncertainty quantification is unrelated to traditional statistical inference.</li>
Answer: TRUE, it will be different compared to the traditional hypothesis testing / confidence interval.

<br>
<li>Communication with domain experts is only important during the problem formulation stage.</li>
Answer: FALSE, communication with domain experts should be constant during the whole data science life cycle to ensure that the analysis makes sense to the domain experts.
</ul>

<h2>Conceptual Exercises</h2>
<ul>
<li>Describe the phenomenon of “data snooping” and how to avoid it.</li>
Answer: Data snooping is the phenomenon of taking the relationship seen in the data as valid and proven conclusive, while it may not necessarily be true. To avoid it, the principles of PCS (predictability, computability and stability) framework can help in ensuring that data snooping is minimized as the frameworks tries to ensure that the relationship that is found in the current data set can be found in future data sets.

<br>
<li>List two differences between Figure 2.2(a) and Figure 2.2(b). Discuss whether you think these changes make the takeaway message of Figure 2.2(b) clearer. Are there any other changes that you would make that might help make the takeaway message clearer?</li>
Answer:<br>
1. 2.2 highlights the background based on the political party that was in office.<br>
2. 2.2 has proper annotation on the earliest and latest spending on energy and climate
I believe that these changes make the points that the author wanted to make easier to digest as we can clearly see that there is barely any increase in Climate spending from 2000 to 2019 (irregardless of the political party in office), while the upward trend of energy spending is up irregardless of the political party in office.<br>
Highlighting the specific bills that was passed during certain years may give more context on the reason for the increase in energy spending, as it might imply that energy spending are based more on macroeconomic trends rather than political trends.

<br>
<li>What does it mean for a result or analysis to be “put into production”?</li>
Answer: To actually implement the result / analysis that was conducted in the real world. For example, if the data science project was about predicting the probability of machine breakdown, then "putting the model into production" will mean that the model will be used in the actual future prediction of machine breakdown, using similar input data that was used to train the model.

<br>
<li>Write a short summary of each stage of the DSLC.</li>
Answer:

1. Problem Formation & Data Collection
    - Domain Problem
        - NARROW DOWN the overall goal of the domain experts, as often the initial question that is being asked is not the right question to achieve their intended goal
        - Identify the data that is already available OR can be collected to answer it
    - Data Collection
        - Important to understand how the data was collected and what values within it mean
        - Every dataset that you use SHOULD BE accompanied by  detailed data documentation (README files & codebooks) that describes how the data was collected & the measurements it contains.
    - Plan for Evaluating Predictability
        
        Check if additional data needs to be collected or planned to be collected. 
        
        Also understand the method to validate the model by using splitting techniques that best reflect the relationship between the current data & future data.
2. Data Cleaning & Exploratory Data Analysis
    - Data Cleaning
        
        CLEAN → TIDY & APPROPRIATELY FORMATTED, with UNAMBIGUOUS ENTRIES
        
        Create a version of the data that is MAXIMALLY reflective of reality and CORRECTLY INTERPRETED by the computer
        
    - Preprocessing
        
        Modifying cleaned data to fit the requirements of a specific algorithm that will be applied
        
    - Exploratory Data Analysis
        
        Deeper look at the data by creating informative tables, such as calculating informative summary statistics (avg, median) & informative visualizations
        
        Exploratory & Explanatory
        
    - Data Snooping & PCS
        
        Data Snooping → presenting relationships & patterns found during EDA as proven conclusions, when in reality it was just shown that these patterns exists for one specific set of data points
        
        Ideally, we want to be free in searching for patterns in the data, but it MUST be able to be demonstrated & re-emerge in relevant future data & stable to the data & cleaning, preprocessing & analytical judgements that were made
        
3. Uncovering Intrinsic Data Structures
    
    Can the data be projected to lower dimensional subspace to summarise the most informative parts of the original data? (DIMENSIONALITY REDUCTION ANALYSIS)
    
    Identify whether there are any natural underlying groupings through CLUSTER ANALYSIS
    
4. Predictive &/ Inferential Analysis
    - Prediction
        
        Ensure that training, validation and test sets resemble relationship between current and future data, while ensuring that predictions that are produced are stable.
        
    - Data-driven Inference
        
        Traditionally involves Hypothesis testing & CI
        
        Check chapter 13 for underlying ideas
        
5. Evaluation of Results
    
    Try to use “negative control”, e.g. showing “fake” versions of the results to domain experts alongside actual results to confirm that the domain experts identify that your actual results are the ones that make the most sense
    
6. Communication of Results & Updating of Domain Knowledge
    
    Create mobile app, write research paper, create an infographic to inform the public, etc
    
    The communication of result should also be tailored to the audience, and take time to explain the analysis & figures.
    
    Build the software to what practitioners are already using / would at least be accessible via a point-and-click website / GUI which they could upload their data.
</ul>

<h2>Reading Exercises</h2>
Read “Science and Statistics” Box (1976), available in the exercises/reading folder on the supplementary GitHub repository. Comment on how it relates to the DSLC and the PCS framework.

<h2>Case Study Exercises</h2>
Imagine that you live in an area regularly affected by wildfires. You decide that you want to develop an algorithm that will predict the next day’s Air Quality Index (AQI) in your town. In this project, you will mentally walk through each stage of the DSLC for this project (you don’t need to implement them unless you’re feeling ambitious). Your answers to the following questions don’t need to be too specific; just provide some general ideas of what you might do at each stage.

- Problem formulation and data collection:<br>
Formulate a project question and discuss what real data sources you will need to collect to answer it. Your data should be publicly available, and we recommend explicitly searching the web for some relevant datasets. You may want to collect multiple sources of data. What data would you use for conducting some predictability evaluations?

- Data cleaning and EDA: <br>How might you need to clean or preprocess your data for analysis? Can you anticipate any issues that the data you found might have? What explorations might you conduct to learn about the data?

- Predictive analysis: <br> What real-world quantity are you trying to predict?

- Evaluation of results: <br> How would you evaluate the trustworthiness of your algorithm?

- Communication of results: <br> Who is your intended audience? What kind of final product would you create so that your algorithm can be used by your intended audience?