<h1>Exercises</h1>

<h2>True or False Exercises </h2>
For each question, specify whether the answer is true or false (briefly justify your answers).
<ul>
<li>Your dslc_documentation files can contain code.</li>
Answer: TRUE

<br>
<li>It is OK to modify your original raw data file if you document the edits that you made.</li>
Answer: FALSE, if needed modify a COPIED version of the raw data file.

<br>
<li>It is bad practice to rerun the code that cleans your data.</li>
Answer: FALSE, rather it is good practice to rerun the code that cleans the data to ensure that the analysis is done on the most up to date version of the cleaned dataset.

<br>
<li>R is better than Python for data science.</li>
Answer: FALSE, both R and Python has its own strengths and weaknesses for data science analysis.

<br>
<li>The term “reproducibility” is defined as being able to reproduce the same results on alternative data.</li>
Answer: FALSE, that is one definition for reproducibility, other definition such as being able to obtain the same results from running the same line of code in the same computer can also satisfy the term "reproducibility"

<br>
<li>The “reproducibility crisis” is a problem unique to data science, and is not an issue in traditional scientific fields.</li>
Answer: FALSE, it also is an issue in other scientific fields such as psychology.

<br>
<li>A good technique for increasing the reproducibility of your results (in the context of ensuring that you get the same results every time you run it) is to rerun your code from scratch often.</li>
Answer: TRUE, this ensures that the code is free of bugs.

<br>
<li>If someone else cannot reproduce your results (based on any definition of reproducibility), then your results must be wrong.</li>
Answer: FALSE, not necessarily, it may be that the documentation was bad. But it is of course better to have the results to be reproducible. Being unreproducible rather, impacts the trustworthiness of your analysis.

<br>
<li>If someone else manages to reproduce your results (using their own independent data and by conducting their own analyses), then this proves that your results are correct.</li>
Answer: FALSE, it is more of a testament of how trustworthy the analysis and results that were conducted, rather that seeing the results were correct. Both of us may have conducted the analysis with code that is filled with bugs.
</ul>

<h2>Conceptual Exercises</h2>
<ul>
<li>What is the purpose of the dslc_documentation/ folder in our recommended project structure?</li>
Answer: To clearly document the process that was undertaken during the analysis as part of the data science life cycle.

<br>
<li>Describe two techniques that help you to write clear, concise, and reusable code.</li>
Answer: Create functions to avoid copy and pasting code that are smilar, and follow the best practice guidelines that are available in the programming language of your choice.

<br>
<li>How would you frame the reproducibility analysis that Herndon and his collaborators conducted in their reevaluation of Reinhart and Rogoff’s results as a stability analysis?</li>
Asnwer: Herndon attempted to reproduce the results using the same data but new code (written by a different person). This is an example of reproducibility that corresponds to the stability of the results to the particular code that was written and to the particular person who conducted the analysis. This type of reproducibility provides stronger evidence of the trustworthiness of the results than trying to reproduce the same result using the same code.

<br>
<li>One form of reproducibility evaluation involves rewriting your own code from scratch.</li>
<ul>
<li>How would you frame this as a stability analysis?</li>
Answer: From a stability standpoint, it is stronger than the stability analysis of running the same code using the same source data, aka tests how robust the findings are when another person analyzes the same data set.

<br>
<li>Does this approach provide stronger or weaker evidence of trustworthiness than the approach used by Herndon?</li>
Answer: This approach provides the same level of trustworthiness as Herndon's attempt, assuming that we are using the same raw data. If a different raw data is used, then it will provide an even stronger evidence of trustworthiness.
</ul>
</ul>