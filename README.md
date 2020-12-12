# CS410 Course Project - CPLSA

## Background ##
This project attempts to duplicate the *Temporal-Author-Topic* analysis in section 4.1 of the paper [*A Mixture Model for Contextual Text Mining (KDD 2006)*](https://github.com/joel515/CourseProject/blob/main/kdd06-mix.pdf) by Qiaozhu Mei and ChengXiang Zhai.  This model replicates the **fixed-coverage contextual mixture model (FC-CPLSA)** covered in section 3.2 of the research paper.  The FC-CPLSA model is a specialized version of the CPLSA mixture model where the coverage over different contexts remains fixed.

The experiment attempts to perform an author-topic comparitive analysis on reasearch paper abstracts between two different authors over three different time periods.  The context features here are the author of the paper and the time which it was written.  In this case, each context featre, and combinations thereof, are evaluated as different "views".  The combination of all context features together form a global topic view.  The mixture model is constructed such that each subview is then related to the global view, but specific to one of the two authors, one of the three timeframes, or one of the six combinations of author and timeframe.  The following table from the reasearch paper illustrates the 12 applicable views more clearly:

| # Context Features | Views |
| --------------------- | ------------------------------------------ |
| 0 | Global View |
| 1 | Author A; Author B; < 1992; 1993 to 1999; 2000 to 2005 |
| 2 | A and < 1992; A and 1993 to 1999; A and 2000 to 2005; B and < 1992; B and 1993 to 1999; B and 2000 to 2005 |

## Data ##
An attempt was made to replicate the data used in the *Temporal-Author-Topic* experiment of section 4.1, which originally consisted of the abstracts from "two famous Data Mining researchers" from the ACM Digital Library prior to the papers publication in 2006.  Since the names of the two authors was not provided, an guess was made using the listing found [here](https://www.kdnuggets.com/2014/08/top-research-leaders-data-mining-data-science.html).  Abstracts were then scraped for [Jiawei Han (UIUC)](https://dl.acm.org/profile/81351593425/publications?AfterYear=1989&BeforeYear=2005&startPage=1&Role=author&pageSize=200) and [Philip S. Yu (UIC)](https://dl.acm.org/profile/81350576309/publications?AfterYear=1977&BeforeYear=2005&Role=author&startPage=0&pageSize=350) published prior to 2006.  The processed CSV file containing the appropriate metadata can be found [here](https://github.com/joel515/CourseProject/blob/main/data/all_abstracts.csv).

## Setup ##
This setup assumes that the user has `git` and Python >= 3.7 installed.  Clone this repository on a local workstation:

`git clone https://github.com/joel515/CourseProject.git`

One the package is cloned, `cd` into the `CourseProject` directory and run the `setup.py` installation script:

`python setup.py install`

This should pull the necessary dependencies to run the mixture model.  If this fails, or if you prefer to install libraries manually, the list of dependencies is as follows:

`numpy`
`pandas`
`nltk`

The latest versions of each should suffice.

## Test Usage ##
To use, you will either need to `cd` into the `CourseProject/cplsa` folder, or use the relative or absolute path to the `CourseProject/cplsa/cplsa.py` script.

### Quick and Dirty ###
To run the script using the data provided and achieve the optimal coverage results, at least the best results that I had achieved, run the following command:

`python cplsa.py ../data/all_abstracts.csv "author==1:author==2" "year<=1992:year>=1993 and year<=1999:year>=2000" -t 20 --noSTEM`

Note that this will take some time to run on a normal workstation (roughly an hour).  It will find an optimal solution for the 12 views with 20 different topics.  Word stemming is omitted from the vocabulary preprocessing.  By default, it will run 20 different "warmup" runs to find the optimal starting point (up to 25 iterations each) with an artificially large prior set on the global view to "ensure a strong signal from global themes", as per the original research paper.  Each warmup run will iterate until the mean global view probability reaches 0.1 (again as per the paper), or until it hits 25 iterations.  The optimal starting point (the one with the largest MLE) is "pickled" and then restarted for the full, 1000 iteration analysis.  Convergence is reached when the difference between the previous log-likelihood and the current one is less than 0.001 (typically around 190 iterations for this dataset).

### Quicker and Dirtier ###
You can also run a less optimal set of iterations to simply check that the package is running correctly:

`python cplsa.py ../data/all_abstracts.csv "author==1:author==2" "year<=1992:year>=1993 and year<=1999:year>=2000" -t 5 -w 2 -th 0.15 -e 0.1 --noSTEM`

This should run in much less time, but will give less than optimal, but decent, results.  This time we are only asking to evaluate 5 topics with only 2 warm up runs, iterating until the mean global view probability reaches 0.15.  The convergence criterion (i.e., the difference between the previous and current log-likelihood) is now only 0.1, so it should converge much quicker.

## General Usage ##
What follows is a description of the inputs and arguments of the CPLSA package.

### Inputs ###
There are 3 required inputs to run this context mixture model: a CSV file containing the documents and associated metadata, and two strings containing Boolean operations to categorize two columns of the metadata, separated by colons.

#### CSV File ####
The CSV file can contain any amount of information, as long as there is one column labeled `text` containing the documents to be evaluated, and two additional labeled columns containing metadata to use as context.  For instance, the data associated with this project looks like the following:

| id | author | year | text |
| --- | -- | -------- | -------------------- |
| 1 | 1 | 2005 | Graphs have become ... |
| ... | ... | ... | ... |
| 363 | 2 | 1985 | The performance of ... |

The `text` column contains the abstracts from either author 1 or 2.  It is important to note the spelling and case of the column titles for the contextual metadata (`author` and `year` in this case), as they will be used in the following inputs to generate the views.  Note that the `id` column is ancillary in this case and hence ignored.

#### View Specification ####
The next two inputs are used to generate the various contextual views to use in the mixture model.  The format for the inputs should be strings enclosed in double quotes.  Each input will refer to only one of the metadata columns and contain multiple Python-formatted Boolean operations to perform on that column's metadata, with each operation separated by a colon.  Each Boolean operation in the string is used to extract a one-feature view.  The code will then combine the different combinations of Boolean operations from the two inputs to extract two-feature views.

In our example, the second input is `"author==1:author==2"`, which consists of two valid Python Boolean operations to perform on the `author` column of metadata.  The input will create two views, one consisting of the `text` from the `author` labeled `1`, and the other consisting of the `text` from the `author` labeled `2`.

Likewise, the third input performs similar Boolean operations, this time on the `year` column (`"year<=1992:year>=1993 and year<=1999:year>=2000"`).  This input contains three valid Python Boolean operations which will result in three one-feature views.  Specifically, one view containing `text` from years prior to 1993 (`year<=1992`), one view containing `text` from the period between 1993 and 1999 (`year>=1993 and year<=1999`), and one view containing `text` after 1999 (`year>=2000`).

The code will automatically create two-feature views through merging each combination of Boolean operations with a logical "AND".  So in our example, the code will create the following six additional operations:

`author==1 and year<=1992`

`author==1 and year>=1993 and year<=1999`

`author==1 and year>=2000`

`author==2 and year<=1992`

`author==2 and year>=1993 and year<=1999`

`author==2 and year>=2000`

In total, if there are `n` operations specified in the first input string, and `m` operations specified in the second string, we will end up with `n + m + nm + 1` views.  In our example, we will have one global view, five one-feature views, and six two-feature views, for a total of 12 views.

### Arguments ###
