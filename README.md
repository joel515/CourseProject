# CS410 Course Project - CPLSA

## Background ##
This project attempts to duplicate the *Temporal-Author-Topic* analysis in section 4.1 of the paper [*A Mixture Model for Contextual Text Mining*](https://github.com/joel515/CourseProject/blob/main/kdd06-mix.pdf) by Qiaozhu Mei and ChengXiang Zhai.  This model replicates the **fixed-coverage contextual mixture model (FC-CPLSA)** covered in section 3.2 of the research paper.  The FC-CPLSA model is a specialized version of the CPLSA mixture model where the coverage over different contexts remains fixed.

The experiment attempts to perform an author-topic comparitive analysis on reasearch paper abstracts between two different authors over three different time periods.  The context features here are the author of the paper and the time which it was written.  In this case, each context featre, and combinations thereof, are evaluated as different "views".  The combination of all context features together form a global topic view.  The mixture model is constructed such that each subview is then related to the global view, but specific to one of the two authors, one of the three timeframes, or one of the six combinations of author and timeframe.  The following table from the reasearch paper illustrates this more clearly:

| # Context Features | Views |
| --------------------- | ------------------------------------------ |
| 0 | Global View |
| 1 | Author A; Author B; < 1992 |
| | 1993 to 1999; 2000 to 2005 |
| 2 | A and < 1992; A and 1993 to 1999 |
| | A and 2000 to 2005; B and < 1992 |
| | B and 1993 to 1999; B and 2000 to 2005 |

## Data ##
An attempt was made to replicate the data used in the *Temporal-Author-Topic* experiment of section 4.1, which originally consisted of the abstracts from 