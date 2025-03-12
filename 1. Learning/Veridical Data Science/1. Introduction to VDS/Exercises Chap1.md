<h1> Exercises </h1>

<h2>True or False Exercises </h2>

For each question, specify whether the answer is true or false (briefly justify your answers).
<ul>
<li>Data is always a good approximation of reality. </li>
Answer: FALSE

<br>
<li> With training, it is possible to eliminate confirmation bias. </li>
Answer: FALSE, maybe not completely eliminate it, but be more aware of it

<br>
<li> Results computed from a dataset that is collected today may not apply to data that will be collected in the future, even if both datasets are collected using the same mechanism. </li>
Answer: TRUE

<br>
<li> As soon as your analysis provides an answer to your domain question, you have conclusively answered the question. </li>
Answer: FALSE, not necessarily, we also need to understand how the analysis arrives at the answer

<br>
<li>There are multiple ways to demonstrate the predictability of your results.</li>
Answer:  TRUE, splitting the dataset can be done in many different ways and will depend on the domain of the data
</ul>

<h2> Conceptual Exercises </h2>
<ul>
<li> Explain why a finding derived from data should be considered evidence rather than proof of a real-world phenomenon. </li>
Answer:<br>
Because absolute proof of real-world phenomenon will need a complete and exhaustive assessment (all other alternative data and scenarios) of whether the finding is indeed true.

<br>
<li> Describe at least two sources of uncertainty that are associated with every data-driven result. </li>
Answer:<br>
1. Method of collection (at specific time, using specific instruments / method); <br>
2. Method of cleaning &/ preprocessing, e.g. do we use the median, mean, 0 value to fill in missing values?

<br>
<li>Describe two techniques (quantitative or qualitative) that you can use to strengthen the evidence for the trustworthiness of a data-driven finding. </li>
Answer:<br>
1. Recompute results based on slightly altered versions of the data.<br>
2. Use alternative metrics during cleaning and preprocessing to investigate whether these metrics will dramatically change the result of the analysis.

<br>
<li>List at least two reasons why Professor Dickman’s calculations might not perfectly reflect the true number of animals that perished in the Australian bushfires. </li>
Answer:<br>
1. The numbers that he used was very old (2007) and may not necessarily reflect the actual density of animals now; <br>
2. The density of animals across space / terrain might not necessarily be constant.

<br>
<li>Imagine that you had infinite resources (people, time, money, etc.) at your disposal. For the Australian bushfire example, describe the hypothetical data that you would collect and how you would collect it to answer the question of how many animals perished in the fires as accurately as possible. Be specific. What values would you measure, and how would you physically measure them? (Assume that you have access to a time machine and can travel back in time to the time period immediately following the fires to collect your data if you choose.) </li>
Answer:<br>
Specifically, I would count the number of dead animal carcasses that can be seen from the bushfire. Since this might not necessarily reflect the actual number of dead animals as their carcasses can be completely burn, I would compare the number of dead animal bodies in certain terrains with the actual animal found in the same terrain but unaffected by the fire. With it, I can validate whether the initial count was correct and then extrapolate / use the number obtained from counting the number of live animals from terrains unaffected by the fire as the estimate of actual dead animals.

<br>
<li>In your own words, briefly summarize the predictability and stability elements of PCS to a family member who completely lacks any technical expertise.</li>
Answer:<br>
1. Predictability essentially implies that the model that was trained for a specific use case will be able to accurately predict the same phenomenon when fed in the same data in the future. e.g. if a model accurately predicts the chance of rain with 90% accuracy, then it should be able to contintue to accurately predict the chance of rain with 90% accuracy in the future
2. Stability implies that the data can perform at the same level as initially predicted when obtaining values that may not necessarily be 100% accurate / uses the same collection method as the data that was used to train the model. Using the same example above, the model must be able to predict the chance of rain around 90% regardless of the measurement error that may be present in the instruments that were used to collect the data.

<br>
<li>What is the role of the validation dataset in assessing the predictability of a data-driven result? </li>
Answer:<br>
It ensures that the model can accurately predict other sets of data that was withheld from it, as proxy for future data.

<br>
<li>Describe one technique that you could use for assessing the stability of a data-driven result. </li>
Answer: Comparing the results that are computed using alternative versions of the data that have been cleaned and preprocessed based on alternative judgment calls that we could have made (e.g., imputing missing values using the mean versus the median). If the alternative versions of the results are fairly similar to one another, this indicates that the uncertainty is not too extreme and that the results are fairly stable, which can be used as evidence of trustworthiness.

<br>
<li>Which splitting technique (group-based, time-based, or random) would be most appropriate for creating a training, validation, and test dataset split in each of the following scenarios:</li>
<ul>
<li>Using historical weather data to develop a weather forecast algorithm</li>
Answer: time-based<br>

<li> Using data from children across 10 Californian schools to evaluate the relationship between class size and test scores in other schools in the state of California </li>
Answer: group-based<br>

<li>Using data from previous elections to develop an algorithm for predicting the results of an upcoming election </li>
Answer: time-based<br>

<li>Using data from a random subset of an online store’s visitors to identify which characteristics are associated with the visitor making a purchase. </li>
Answer: random-based<br>

</ul>

</ul>

<h2> Reading Exercises </h2>
PDF files containing each of the papers listed here can be found in the exercises/reading folder on the supplementary GitHub repository.
<ul>
<li>Read “Statistical Modeling: The Two Cultures” by Breiman (2001), and write a short summary of how it relates to the PCS framework. Don’t worry if this paper is a little bit too technical at this stage; feel free just skim it to get the big picture.</li>

<br>
<li>Read “Veridical Data Science” by Yu and Kumbier (2020), our original paper presenting a high-level outline of the ideas underlying the veridical data science framework.</li>

<br>
<li>Read “50 Years of Data Science” by Donoho (2017), and comment on the similarities and differences between Donoho’s perspectives and the perspectives we have presented in this book so far.</li>
</ul>
<h2> Case Study Exercises </h2>
Read the 2022 NBC News article “Chart: Remote Work is Disappearing as More People Return to the Office” by Joe Murphy (a PDF copy can be found in the exercises/reading folder on the supplementary GitHub repository, and the original article can be found here). Then answer the following critical thinking questions:
<ul>
<li>What is the preliminary conclusion of the article?</li>

<br>
<li>On what data is the conclusion based? How was this data collected?</li>

<br>
<li>Is the data representative of the population of interest?</li>

<br>
<li>List at least two assumptions that underlie the conclusion.</li>

<br>
<li>Does the title accurately represent the findings of the study?</li>

<br>
<li>Do you think that these results are (qualitatively) trustworthy? What additional evidence might you seek to determine whether the results are trustworthy?</li>
</ul>