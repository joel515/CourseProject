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

## Usage ##
To use, you will either need to `cd` into the `CourseProject/cplsa` folder, or use the relative or absolute path to the `CourseProject/cplsa/cplsa/py` script.

### Quick and Dirty ###
To run the script using the data provided and achieve the optimal coverage results, at least the best results that I found, run the following command:

`python cplsa.py ../data/all_abstracts.csv "author==1:author==2" "year<=1992:year>=1993 and year<=1999:year>=2000" -t 20 --noSTEM`

Note that this will take some time to run on a normal workstation (roughly an hour).  It will find an optimal solution for the 12 views with 20 different topics.  By default, it will run 20 different "warmup" runs to find the optimal starting point (up to 25 iterations each) with an artificially large prior set on the global view to "ensure a strong signal from global themes", as per the original research paper.  Each warmup run will iterate until the mean global view probability reaches 0.1 (again as per the paper), or until it hits 25 iterations.  The optimal starting point (the one with the largest MLE) is "pickled" and then restarted for the full, 1000 iteration analysis.  Convergence is reached when the difference between the previous log-likelihood and the current one is less than 0.001 (typically around 190 iterations for this dataset).

### Quicker and Dirtier ###
You can also run a less optimal set of iterations to simply check that the package is running correctly:

`python cplsa.py ../data/all_abstracts.csv "author==1:author==2" "year<=1992:year>=1993 and year<=1999:year>=2000" -t 5 -w 2 -th 0.15 -e 0.1 --noSTEM`

This should run in much less time, but will give less than optimal, but decent, results.  This time we are only asking to evaluate 5 topics with only 2 warm up runs, iterating until the mean global view probability reaches 0.15.  The convergence criterion (i.e., the difference between the previous and current log-likelihood) is now only 0.1, so it should converge much quicker.