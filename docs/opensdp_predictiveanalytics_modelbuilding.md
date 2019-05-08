---
title: "Predictive Analytics in Education - Building Models in R"
author: "Dashiell Young-Saver, Jared Knowles"
date: "May 30, 2019"
output: 
  html_document:
    theme: simplex
    css: styles.css
    highlight: NULL
    keep_md: true
    toc: true
    toc_depth: 3
    toc_float: true
    number_sections: false
    code_folding: show
    includes:
      in_header: zz-sdp_ga.html
---

# Predictive Analytics in Education

<div class="navbar navbar-default navbar-fixed-top" id="logo">
<div class="container">
<img src="https://opensdp.github.io/assets/images/OpenSDP-Banner_crimson.jpg" style="display: block; margin: 0 auto; height: 115px;">
</div>
</div>




## Getting Started

If you're using the `Rmd` file version of these materials, start by saving a new
version of the file, so you can edit it without worrying about overwriting the
original. Then work through the file in RStudio by highlighting one or a few
command lines at a time, clicking the "execute" icon (or pressing
control-enter), and then looking at the results in the R console. Edit or add
commands as you wish.  If you're using a paper or PDF version of these
materials, just read on--the R output appears below each section of commands.

This guide is built using the data from the fictional state education agency of
Montucky. This data set includes simulated data for multiple cohorts of 7th
graders along with their corresponding high school outcomes. Each observation
(row) is a student, and includes associated information about the student's
demographics, academic performance in 7th grade, associated school and district
(in grade 7), last high school attended, and high school completion. For
additional details on the data elements that makeup the synthetic data, there is
a codebook available in the data sub-directory.

## Setup

To prepare for this project you will need to ensure that your R installation
has the necessary add-on packages and that you can read in the training data.


```r
# Install add-on packages needed
install.packages("dplyr") # this will update your installed version to align with
install.packages("pROC") # those in the tutorial
install.packages("devtools")
install.packages("caret") # for machine learning
install.packages("future") # for multicore processing on Windows/Mac/Linux
```


```r
# Load the packages you need
library(dplyr)
library(pROC)
library(devtools)
library(future)
library(klaR)

# Load the helper functions not in packages
source("../R/functions.R")

# Read in the data
# This command assumes that the data is in a folder called data, below your
# current working directory. You can check your working directory with the
# getwd() command, and you can set your working directory using the RStudio
# environment, or the setwd() command.

load("../data/montucky.rda")
```



## Beyond Logistic Regression


Logistic regression is where you should start. It is fast to compute, easier to
interpret, and usually does a great job. However, there are a number of alternative
algorithms available to you and R provides a common interface to them through the
`caret` package. The `train` function (as in, training the models) is the
workhorse of the `caret` package. It has an extensive set of user-controlled
options.

When switching away from logistic regression, it can be advantageous to transform
predictors to be centered at 0 with a standard deviation of 1. This helps
put binary and continuous indicators on a similar scale and helps avoid
problems associated with rounding, large numbers, and the optimization algorithms
used to evaluate model fit. Here is an example of how you can do this in R:


```r
# RESUME PREPPING DATA FROM GUIDE 1
sea_data$sid <- paste(sea_data$sid, sea_data$sch_g7_lea_id, sep = "-")
sea_data$scale_score_7_math[sea_data$scale_score_7_math < 0] <- NA
```




```r
library(caret) # machine learning workhorse in R
sea_data <- as.data.frame(sea_data)

# To use caret we have to manually expand our categorical variables. caret
# can use formulas like lm() or glm() but becomes very slow and unreliable
# for some model/algorithm methods when doing so. Better to do it ourselves.

# Expand categorical variables
# First declare any variable we want to be treated as a category as a factor
sea_data$frpl_7 <- factor(sea_data$frpl_7)
sea_data$male <- factor(sea_data$male)
# Use the dummyVars function to build an expansion function to expand our data
expand_factors <- caret::dummyVars(~ race_ethnicity + frpl_7 + male,
                            data = sea_data)
# Create a matrix of indicator variables by using the predict method for the
# expand_factors object we just created. There are other ways you can do this, but
# the advantage is we can use the `expand_factors` object on future/different
# datasets to preserve factor levels that may not be present in those (like our
# test set perhaps)
fact_vars <- predict(expand_factors, sea_data)
# Get the column names of our categorical dummy variables
categ_vars <- colnames(fact_vars)
# Combine the new dummy variables with our original dataset
sea_data <- cbind(sea_data, fact_vars)
# Drop the matrix of dummy variables
rm(fact_vars)

# Define the numeric/continuous predictors we want to use
continuous_x_vars <- c('scale_score_7_math', 'scale_score_7_read',
                          'pct_days_absent_7', 'sch_g7_frpl_per')

# caret does not gracefully handle missing data, so we have to do some additional
# work when we define our training/test data split. You can choose additional
# ways to address missing data (imputation, substituting mean values, etc.) -
# here we opt for the simple but aggressive strategy of excluding any row
# with missing values for any predictor from being in the training data

train_idx <- row.names( # get the row.names to define our index
  na.omit(sea_data[sea_data$year %in% c(2003, 2004, 2005),
                   # reduce the data to rows of specific years and not missing data
                   c(continuous_x_vars, categ_vars)])
  # only select the predictors we are interested in
)
test_idx <- !row.names(sea_data[sea_data$year > 2005,]) %in% train_idx

# All of our algorithms will run much faster and more smoothly if we center
# our continuous measures at 0 with a standard deviation of 1. We have
# skipped the important step of identifying and removing outliers, which we
# should make sure to do, but is left up to you!
pre_proc <- caret::preProcess(sea_data[train_idx, continuous_x_vars],
                       method = c("scale", "center"))

# We now have a pre-processing object which will scale and center our variables
# for us. It will ignore any variables that are not defined within it, so we
# can pass all of our continuous and dummy variables to it to produce our
# final data frame of predictors.
preds <- predict(pre_proc,
                 sea_data[train_idx, # keep only the training rows
                          c(continuous_x_vars, categ_vars)]
                 ) # keep only the columns of dummy and continuous variables
```


Once we have defined our predictor variables, we need to tell `train` how we want
to test our models. Most of the algorithms offered through `caret` have "tuning
parameters", user-controlled values, that are not estimated from the data. Our
goal is to experiment with these values and find the values that fit the data
the best. To do this, we must tell `train` which values to try, and how to
evaluate their performance.

Luckily, `train()` has a number of sensible defaults that largely automate this
process for us. For the purpose of this exercise, a good set of defaults is to
use the `twoClassSummary()` model evaluation function (which tells us the area
under the curve as well as the sensitivity, specificity, and accuracy) and
to use repeated cross-fold validation.


```r
# Take advantage of all the processing power on your machine
library(doFuture)
plan(multiprocess(workers = 4)) # define the number of cpus to use
registerDoFuture() # register them with R

# Caret really really really likes if you do binary classification that you
# code the variables as factors with alphabetical labels. In this case, we
# recode 0/1 to be nongrad, grad.

yvar <- sea_data[train_idx, "ontime_grad"] # save only training observations
yvar <- ifelse(yvar == 1, "grad", "nongrad")
yvar <- factor(yvar)

# On a standard desktop/laptop it can be necessary to decrease the sample size
# to train in a reasonable amount of time. For the prototype and getting feedback
# it's a good idea to stick with a reasonable sample size of under 20,000 rows.
# Let's do that here:

train_idx_small <- sample(1:nrow(preds), 2e4)

# Caret has a number of complex options, you can read about under ?trainControl
# Here we set some sensible defaults
example_control <- trainControl(
  method = "cv", # we cross-validate our model to avoid overfitting
  classProbs = TRUE,  # we want to be able to predict probabilities, not just binary outcomes
  returnData = TRUE, # we want to store the model data to allow for postestimation
  summaryFunction = twoClassSummary, # we want to use the prSummary for better two-class accuracy measures
  trim = TRUE, # we want to reduce the size of the final model object to save RAM
  savePredictions = "final", # we want to store the predictions from the final model
  returnResamp = "final", # we want to store the resampling code to directly compare other methods
  allowParallel = TRUE # we want to use all the processors on our computer if possible
  )

# This will take quite some time:
set.seed(2532) # set seed so models are comparable
rpart_model <- train(
  y = yvar[train_idx_small], # specify the outcome, here subset
  # to the smaller number of observations
  x = preds[train_idx_small, ], # specify the matrix of predictors, again subset
  method = "rpart",  # choose the algorithm - here a regression tree
  tuneLength = 24, # choose how many values of tuning parameters to test
  trControl = example_control, # set up the conditions for the test (defined above)
   metric = "ROC" # select the metric to choose the best model based on
  )


set.seed(2532)
# Repeat above but just change the `method` argument to GBM for gradient boosted machines
gbm_model <- train(y = yvar[train_idx_small],
             x = preds[train_idx_small,],
             method = "gbm", tuneLength = 4,
             trControl = example_control,
             metric = "ROC")
```

```
Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2709             nan     0.1000    0.0081
     2        1.2572             nan     0.1000    0.0066
     3        1.2463             nan     0.1000    0.0052
     4        1.2372             nan     0.1000    0.0044
     5        1.2294             nan     0.1000    0.0038
     6        1.2230             nan     0.1000    0.0029
     7        1.2172             nan     0.1000    0.0030
     8        1.2113             nan     0.1000    0.0027
     9        1.2067             nan     0.1000    0.0024
    10        1.2021             nan     0.1000    0.0020
    20        1.1708             nan     0.1000    0.0011
    40        1.1402             nan     0.1000    0.0003
    60        1.1249             nan     0.1000    0.0003
    80        1.1169             nan     0.1000    0.0000
   100        1.1129             nan     0.1000    0.0000
   120        1.1104             nan     0.1000    0.0000
   140        1.1090             nan     0.1000   -0.0000
   160        1.1079             nan     0.1000   -0.0000
   180        1.1071             nan     0.1000   -0.0001
   200        1.1063             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2695             nan     0.1000    0.0086
     2        1.2513             nan     0.1000    0.0090
     3        1.2360             nan     0.1000    0.0074
     4        1.2253             nan     0.1000    0.0049
     5        1.2153             nan     0.1000    0.0051
     6        1.2058             nan     0.1000    0.0042
     7        1.1984             nan     0.1000    0.0037
     8        1.1915             nan     0.1000    0.0031
     9        1.1860             nan     0.1000    0.0026
    10        1.1807             nan     0.1000    0.0025
    20        1.1463             nan     0.1000    0.0010
    40        1.1183             nan     0.1000    0.0003
    60        1.1095             nan     0.1000   -0.0000
    80        1.1049             nan     0.1000    0.0000
   100        1.1021             nan     0.1000   -0.0001
   120        1.0996             nan     0.1000    0.0000
   140        1.0972             nan     0.1000   -0.0000
   160        1.0950             nan     0.1000   -0.0001
   180        1.0925             nan     0.1000   -0.0001
   200        1.0909             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2647             nan     0.1000    0.0114
     2        1.2468             nan     0.1000    0.0090
     3        1.2309             nan     0.1000    0.0076
     4        1.2187             nan     0.1000    0.0063
     5        1.2077             nan     0.1000    0.0054
     6        1.1981             nan     0.1000    0.0044
     7        1.1899             nan     0.1000    0.0040
     8        1.1823             nan     0.1000    0.0035
     9        1.1759             nan     0.1000    0.0031
    10        1.1696             nan     0.1000    0.0030
    20        1.1338             nan     0.1000    0.0008
    40        1.1104             nan     0.1000    0.0001
    60        1.1028             nan     0.1000    0.0000
    80        1.0981             nan     0.1000   -0.0000
   100        1.0949             nan     0.1000   -0.0000
   120        1.0909             nan     0.1000   -0.0000
   140        1.0876             nan     0.1000   -0.0001
   160        1.0845             nan     0.1000   -0.0001
   180        1.0816             nan     0.1000   -0.0000
   200        1.0781             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2631             nan     0.1000    0.0122
     2        1.2434             nan     0.1000    0.0097
     3        1.2276             nan     0.1000    0.0080
     4        1.2134             nan     0.1000    0.0069
     5        1.2016             nan     0.1000    0.0056
     6        1.1919             nan     0.1000    0.0047
     7        1.1832             nan     0.1000    0.0041
     8        1.1758             nan     0.1000    0.0037
     9        1.1686             nan     0.1000    0.0034
    10        1.1625             nan     0.1000    0.0029
    20        1.1280             nan     0.1000    0.0007
    40        1.1072             nan     0.1000    0.0000
    60        1.0988             nan     0.1000    0.0001
    80        1.0925             nan     0.1000    0.0000
   100        1.0870             nan     0.1000    0.0001
   120        1.0819             nan     0.1000   -0.0002
   140        1.0775             nan     0.1000   -0.0001
   160        1.0730             nan     0.1000    0.0000
   180        1.0675             nan     0.1000    0.0001
   200        1.0641             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2710             nan     0.1000    0.0083
     2        1.2571             nan     0.1000    0.0068
     3        1.2463             nan     0.1000    0.0055
     4        1.2367             nan     0.1000    0.0047
     5        1.2288             nan     0.1000    0.0039
     6        1.2227             nan     0.1000    0.0032
     7        1.2169             nan     0.1000    0.0028
     8        1.2119             nan     0.1000    0.0023
     9        1.2062             nan     0.1000    0.0027
    10        1.2017             nan     0.1000    0.0022
    20        1.1718             nan     0.1000    0.0007
    40        1.1417             nan     0.1000    0.0003
    60        1.1276             nan     0.1000    0.0001
    80        1.1196             nan     0.1000    0.0001
   100        1.1156             nan     0.1000    0.0000
   120        1.1133             nan     0.1000   -0.0001
   140        1.1120             nan     0.1000   -0.0000
   160        1.1109             nan     0.1000   -0.0000
   180        1.1101             nan     0.1000   -0.0000
   200        1.1094             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2668             nan     0.1000    0.0103
     2        1.2494             nan     0.1000    0.0083
     3        1.2363             nan     0.1000    0.0065
     4        1.2251             nan     0.1000    0.0055
     5        1.2152             nan     0.1000    0.0049
     6        1.2068             nan     0.1000    0.0043
     7        1.2002             nan     0.1000    0.0032
     8        1.1933             nan     0.1000    0.0034
     9        1.1874             nan     0.1000    0.0029
    10        1.1825             nan     0.1000    0.0023
    20        1.1489             nan     0.1000    0.0010
    40        1.1211             nan     0.1000    0.0002
    60        1.1119             nan     0.1000    0.0001
    80        1.1082             nan     0.1000   -0.0000
   100        1.1046             nan     0.1000   -0.0000
   120        1.1026             nan     0.1000    0.0000
   140        1.0997             nan     0.1000   -0.0000
   160        1.0974             nan     0.1000   -0.0001
   180        1.0951             nan     0.1000   -0.0000
   200        1.0936             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2648             nan     0.1000    0.0112
     2        1.2471             nan     0.1000    0.0088
     3        1.2319             nan     0.1000    0.0073
     4        1.2184             nan     0.1000    0.0067
     5        1.2072             nan     0.1000    0.0053
     6        1.1977             nan     0.1000    0.0046
     7        1.1894             nan     0.1000    0.0039
     8        1.1823             nan     0.1000    0.0032
     9        1.1754             nan     0.1000    0.0033
    10        1.1695             nan     0.1000    0.0026
    20        1.1357             nan     0.1000    0.0007
    40        1.1135             nan     0.1000    0.0002
    60        1.1060             nan     0.1000   -0.0000
    80        1.1010             nan     0.1000    0.0000
   100        1.0968             nan     0.1000    0.0000
   120        1.0937             nan     0.1000   -0.0000
   140        1.0894             nan     0.1000   -0.0000
   160        1.0862             nan     0.1000   -0.0001
   180        1.0833             nan     0.1000   -0.0001
   200        1.0802             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2629             nan     0.1000    0.0118
     2        1.2430             nan     0.1000    0.0098
     3        1.2270             nan     0.1000    0.0079
     4        1.2130             nan     0.1000    0.0067
     5        1.2020             nan     0.1000    0.0051
     6        1.1927             nan     0.1000    0.0047
     7        1.1836             nan     0.1000    0.0041
     8        1.1762             nan     0.1000    0.0032
     9        1.1700             nan     0.1000    0.0033
    10        1.1633             nan     0.1000    0.0031
    20        1.1286             nan     0.1000    0.0010
    40        1.1081             nan     0.1000    0.0000
    60        1.0995             nan     0.1000   -0.0000
    80        1.0943             nan     0.1000   -0.0001
   100        1.0889             nan     0.1000   -0.0001
   120        1.0841             nan     0.1000   -0.0001
   140        1.0797             nan     0.1000   -0.0001
   160        1.0756             nan     0.1000   -0.0001
   180        1.0717             nan     0.1000   -0.0000
   200        1.0684             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2715             nan     0.1000    0.0082
     2        1.2582             nan     0.1000    0.0066
     3        1.2473             nan     0.1000    0.0053
     4        1.2382             nan     0.1000    0.0046
     5        1.2300             nan     0.1000    0.0041
     6        1.2232             nan     0.1000    0.0034
     7        1.2177             nan     0.1000    0.0026
     8        1.2121             nan     0.1000    0.0026
     9        1.2073             nan     0.1000    0.0024
    10        1.2029             nan     0.1000    0.0020
    20        1.1727             nan     0.1000    0.0009
    40        1.1426             nan     0.1000    0.0005
    60        1.1275             nan     0.1000    0.0002
    80        1.1194             nan     0.1000    0.0001
   100        1.1153             nan     0.1000    0.0000
   120        1.1129             nan     0.1000    0.0000
   140        1.1115             nan     0.1000   -0.0000
   160        1.1103             nan     0.1000   -0.0000
   180        1.1093             nan     0.1000   -0.0000
   200        1.1087             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2668             nan     0.1000    0.0103
     2        1.2508             nan     0.1000    0.0082
     3        1.2376             nan     0.1000    0.0066
     4        1.2262             nan     0.1000    0.0054
     5        1.2164             nan     0.1000    0.0050
     6        1.2077             nan     0.1000    0.0042
     7        1.2001             nan     0.1000    0.0037
     8        1.1931             nan     0.1000    0.0034
     9        1.1872             nan     0.1000    0.0028
    10        1.1825             nan     0.1000    0.0022
    20        1.1493             nan     0.1000    0.0011
    40        1.1211             nan     0.1000    0.0002
    60        1.1126             nan     0.1000    0.0001
    80        1.1082             nan     0.1000    0.0000
   100        1.1058             nan     0.1000   -0.0000
   120        1.1025             nan     0.1000   -0.0001
   140        1.1007             nan     0.1000   -0.0000
   160        1.0986             nan     0.1000   -0.0000
   180        1.0967             nan     0.1000   -0.0001
   200        1.0946             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2657             nan     0.1000    0.0105
     2        1.2465             nan     0.1000    0.0097
     3        1.2311             nan     0.1000    0.0077
     4        1.2186             nan     0.1000    0.0058
     5        1.2076             nan     0.1000    0.0052
     6        1.1985             nan     0.1000    0.0046
     7        1.1896             nan     0.1000    0.0042
     8        1.1828             nan     0.1000    0.0032
     9        1.1763             nan     0.1000    0.0033
    10        1.1704             nan     0.1000    0.0028
    20        1.1357             nan     0.1000    0.0010
    40        1.1136             nan     0.1000    0.0001
    60        1.1065             nan     0.1000   -0.0001
    80        1.1021             nan     0.1000   -0.0001
   100        1.0980             nan     0.1000   -0.0001
   120        1.0945             nan     0.1000   -0.0000
   140        1.0913             nan     0.1000   -0.0001
   160        1.0879             nan     0.1000   -0.0001
   180        1.0846             nan     0.1000   -0.0001
   200        1.0812             nan     0.1000    0.0002

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2633             nan     0.1000    0.0121
     2        1.2436             nan     0.1000    0.0095
     3        1.2272             nan     0.1000    0.0078
     4        1.2141             nan     0.1000    0.0064
     5        1.2014             nan     0.1000    0.0058
     6        1.1922             nan     0.1000    0.0047
     7        1.1834             nan     0.1000    0.0043
     8        1.1760             nan     0.1000    0.0036
     9        1.1694             nan     0.1000    0.0033
    10        1.1635             nan     0.1000    0.0028
    20        1.1295             nan     0.1000    0.0009
    40        1.1076             nan     0.1000    0.0001
    60        1.1002             nan     0.1000    0.0000
    80        1.0941             nan     0.1000   -0.0001
   100        1.0892             nan     0.1000   -0.0000
   120        1.0846             nan     0.1000   -0.0001
   140        1.0796             nan     0.1000   -0.0000
   160        1.0751             nan     0.1000    0.0003
   180        1.0705             nan     0.1000   -0.0001
   200        1.0656             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2706             nan     0.1000    0.0083
     2        1.2572             nan     0.1000    0.0068
     3        1.2456             nan     0.1000    0.0055
     4        1.2362             nan     0.1000    0.0044
     5        1.2283             nan     0.1000    0.0037
     6        1.2214             nan     0.1000    0.0033
     7        1.2155             nan     0.1000    0.0029
     8        1.2104             nan     0.1000    0.0023
     9        1.2050             nan     0.1000    0.0027
    10        1.2001             nan     0.1000    0.0024
    20        1.1692             nan     0.1000    0.0011
    40        1.1397             nan     0.1000    0.0004
    60        1.1249             nan     0.1000    0.0003
    80        1.1172             nan     0.1000    0.0000
   100        1.1133             nan     0.1000    0.0001
   120        1.1107             nan     0.1000    0.0000
   140        1.1093             nan     0.1000   -0.0000
   160        1.1082             nan     0.1000   -0.0000
   180        1.1072             nan     0.1000   -0.0000
   200        1.1065             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2656             nan     0.1000    0.0105
     2        1.2478             nan     0.1000    0.0085
     3        1.2347             nan     0.1000    0.0066
     4        1.2236             nan     0.1000    0.0052
     5        1.2138             nan     0.1000    0.0046
     6        1.2051             nan     0.1000    0.0043
     7        1.1973             nan     0.1000    0.0038
     8        1.1905             nan     0.1000    0.0034
     9        1.1845             nan     0.1000    0.0029
    10        1.1792             nan     0.1000    0.0025
    20        1.1455             nan     0.1000    0.0009
    40        1.1186             nan     0.1000    0.0001
    60        1.1097             nan     0.1000    0.0000
    80        1.1053             nan     0.1000   -0.0000
   100        1.1020             nan     0.1000   -0.0001
   120        1.0993             nan     0.1000   -0.0001
   140        1.0970             nan     0.1000   -0.0001
   160        1.0942             nan     0.1000    0.0001
   180        1.0918             nan     0.1000   -0.0000
   200        1.0906             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2641             nan     0.1000    0.0117
     2        1.2455             nan     0.1000    0.0094
     3        1.2302             nan     0.1000    0.0075
     4        1.2162             nan     0.1000    0.0066
     5        1.2049             nan     0.1000    0.0055
     6        1.1956             nan     0.1000    0.0045
     7        1.1880             nan     0.1000    0.0038
     8        1.1799             nan     0.1000    0.0040
     9        1.1735             nan     0.1000    0.0031
    10        1.1677             nan     0.1000    0.0028
    20        1.1331             nan     0.1000    0.0010
    40        1.1115             nan     0.1000    0.0001
    60        1.1045             nan     0.1000    0.0000
    80        1.0993             nan     0.1000   -0.0001
   100        1.0956             nan     0.1000   -0.0000
   120        1.0926             nan     0.1000   -0.0001
   140        1.0896             nan     0.1000    0.0000
   160        1.0864             nan     0.1000   -0.0001
   180        1.0827             nan     0.1000   -0.0001
   200        1.0806             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2636             nan     0.1000    0.0119
     2        1.2426             nan     0.1000    0.0100
     3        1.2263             nan     0.1000    0.0080
     4        1.2126             nan     0.1000    0.0068
     5        1.2012             nan     0.1000    0.0057
     6        1.1908             nan     0.1000    0.0049
     7        1.1814             nan     0.1000    0.0043
     8        1.1734             nan     0.1000    0.0038
     9        1.1664             nan     0.1000    0.0033
    10        1.1603             nan     0.1000    0.0027
    20        1.1270             nan     0.1000    0.0010
    40        1.1058             nan     0.1000    0.0003
    60        1.0981             nan     0.1000    0.0000
    80        1.0922             nan     0.1000   -0.0001
   100        1.0879             nan     0.1000    0.0000
   120        1.0833             nan     0.1000   -0.0002
   140        1.0790             nan     0.1000   -0.0000
   160        1.0739             nan     0.1000   -0.0001
   180        1.0698             nan     0.1000   -0.0002
   200        1.0653             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2716             nan     0.1000    0.0076
     2        1.2582             nan     0.1000    0.0068
     3        1.2469             nan     0.1000    0.0055
     4        1.2380             nan     0.1000    0.0044
     5        1.2306             nan     0.1000    0.0035
     6        1.2231             nan     0.1000    0.0033
     7        1.2176             nan     0.1000    0.0027
     8        1.2119             nan     0.1000    0.0027
     9        1.2077             nan     0.1000    0.0020
    10        1.2032             nan     0.1000    0.0022
    20        1.1730             nan     0.1000    0.0010
    40        1.1444             nan     0.1000    0.0003
    60        1.1296             nan     0.1000    0.0002
    80        1.1222             nan     0.1000    0.0000
   100        1.1184             nan     0.1000   -0.0000
   120        1.1159             nan     0.1000   -0.0000
   140        1.1145             nan     0.1000   -0.0000
   160        1.1134             nan     0.1000   -0.0000
   180        1.1127             nan     0.1000   -0.0000
   200        1.1120             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2657             nan     0.1000    0.0100
     2        1.2511             nan     0.1000    0.0072
     3        1.2373             nan     0.1000    0.0071
     4        1.2255             nan     0.1000    0.0058
     5        1.2160             nan     0.1000    0.0047
     6        1.2078             nan     0.1000    0.0040
     7        1.2006             nan     0.1000    0.0034
     8        1.1946             nan     0.1000    0.0029
     9        1.1887             nan     0.1000    0.0028
    10        1.1840             nan     0.1000    0.0021
    20        1.1490             nan     0.1000    0.0009
    40        1.1240             nan     0.1000    0.0002
    60        1.1156             nan     0.1000   -0.0000
    80        1.1118             nan     0.1000    0.0000
   100        1.1088             nan     0.1000   -0.0001
   120        1.1066             nan     0.1000   -0.0000
   140        1.1041             nan     0.1000   -0.0001
   160        1.1015             nan     0.1000   -0.0000
   180        1.0995             nan     0.1000   -0.0001
   200        1.0972             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2646             nan     0.1000    0.0112
     2        1.2463             nan     0.1000    0.0089
     3        1.2318             nan     0.1000    0.0070
     4        1.2200             nan     0.1000    0.0060
     5        1.2096             nan     0.1000    0.0049
     6        1.2006             nan     0.1000    0.0045
     7        1.1920             nan     0.1000    0.0039
     8        1.1847             nan     0.1000    0.0036
     9        1.1781             nan     0.1000    0.0030
    10        1.1727             nan     0.1000    0.0025
    20        1.1380             nan     0.1000    0.0009
    40        1.1160             nan     0.1000    0.0002
    60        1.1081             nan     0.1000    0.0003
    80        1.1032             nan     0.1000   -0.0000
   100        1.0989             nan     0.1000   -0.0000
   120        1.0957             nan     0.1000   -0.0001
   140        1.0916             nan     0.1000   -0.0000
   160        1.0883             nan     0.1000   -0.0000
   180        1.0856             nan     0.1000   -0.0001
   200        1.0827             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2634             nan     0.1000    0.0119
     2        1.2439             nan     0.1000    0.0095
     3        1.2281             nan     0.1000    0.0079
     4        1.2149             nan     0.1000    0.0065
     5        1.2035             nan     0.1000    0.0054
     6        1.1937             nan     0.1000    0.0046
     7        1.1852             nan     0.1000    0.0040
     8        1.1779             nan     0.1000    0.0034
     9        1.1710             nan     0.1000    0.0033
    10        1.1655             nan     0.1000    0.0027
    20        1.1322             nan     0.1000    0.0007
    40        1.1114             nan     0.1000    0.0003
    60        1.1031             nan     0.1000    0.0002
    80        1.0979             nan     0.1000    0.0000
   100        1.0925             nan     0.1000   -0.0001
   120        1.0882             nan     0.1000   -0.0001
   140        1.0838             nan     0.1000   -0.0001
   160        1.0800             nan     0.1000   -0.0001
   180        1.0764             nan     0.1000   -0.0002
   200        1.0726             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2716             nan     0.1000    0.0077
     2        1.2581             nan     0.1000    0.0067
     3        1.2469             nan     0.1000    0.0054
     4        1.2380             nan     0.1000    0.0044
     5        1.2300             nan     0.1000    0.0038
     6        1.2237             nan     0.1000    0.0030
     7        1.2180             nan     0.1000    0.0027
     8        1.2135             nan     0.1000    0.0021
     9        1.2078             nan     0.1000    0.0027
    10        1.2034             nan     0.1000    0.0022
    20        1.1741             nan     0.1000    0.0007
    40        1.1446             nan     0.1000    0.0004
    60        1.1299             nan     0.1000    0.0002
    80        1.1222             nan     0.1000    0.0001
   100        1.1181             nan     0.1000    0.0000
   120        1.1159             nan     0.1000   -0.0000
   140        1.1146             nan     0.1000    0.0000
   160        1.1136             nan     0.1000   -0.0000
   180        1.1127             nan     0.1000   -0.0001
   200        1.1120             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2677             nan     0.1000    0.0100
     2        1.2512             nan     0.1000    0.0081
     3        1.2377             nan     0.1000    0.0064
     4        1.2262             nan     0.1000    0.0056
     5        1.2164             nan     0.1000    0.0046
     6        1.2084             nan     0.1000    0.0039
     7        1.2009             nan     0.1000    0.0036
     8        1.1946             nan     0.1000    0.0030
     9        1.1888             nan     0.1000    0.0026
    10        1.1839             nan     0.1000    0.0021
    20        1.1498             nan     0.1000    0.0010
    40        1.1235             nan     0.1000    0.0003
    60        1.1150             nan     0.1000    0.0001
    80        1.1108             nan     0.1000    0.0001
   100        1.1075             nan     0.1000   -0.0000
   120        1.1049             nan     0.1000   -0.0001
   140        1.1027             nan     0.1000   -0.0000
   160        1.1004             nan     0.1000   -0.0000
   180        1.0980             nan     0.1000    0.0001
   200        1.0961             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2647             nan     0.1000    0.0110
     2        1.2471             nan     0.1000    0.0088
     3        1.2323             nan     0.1000    0.0073
     4        1.2196             nan     0.1000    0.0059
     5        1.2091             nan     0.1000    0.0052
     6        1.2001             nan     0.1000    0.0046
     7        1.1922             nan     0.1000    0.0035
     8        1.1843             nan     0.1000    0.0039
     9        1.1779             nan     0.1000    0.0030
    10        1.1722             nan     0.1000    0.0026
    20        1.1378             nan     0.1000    0.0010
    40        1.1176             nan     0.1000   -0.0000
    60        1.1100             nan     0.1000    0.0001
    80        1.1045             nan     0.1000    0.0002
   100        1.1003             nan     0.1000   -0.0000
   120        1.0976             nan     0.1000   -0.0000
   140        1.0944             nan     0.1000   -0.0001
   160        1.0910             nan     0.1000   -0.0001
   180        1.0880             nan     0.1000   -0.0000
   200        1.0848             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2635             nan     0.1000    0.0117
     2        1.2447             nan     0.1000    0.0093
     3        1.2290             nan     0.1000    0.0078
     4        1.2156             nan     0.1000    0.0064
     5        1.2047             nan     0.1000    0.0053
     6        1.1950             nan     0.1000    0.0045
     7        1.1867             nan     0.1000    0.0042
     8        1.1795             nan     0.1000    0.0035
     9        1.1731             nan     0.1000    0.0032
    10        1.1673             nan     0.1000    0.0027
    20        1.1325             nan     0.1000    0.0010
    40        1.1126             nan     0.1000    0.0003
    60        1.1047             nan     0.1000   -0.0001
    80        1.0978             nan     0.1000   -0.0001
   100        1.0932             nan     0.1000   -0.0001
   120        1.0892             nan     0.1000    0.0000
   140        1.0852             nan     0.1000   -0.0000
   160        1.0809             nan     0.1000    0.0000
   180        1.0770             nan     0.1000   -0.0001
   200        1.0737             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2705             nan     0.1000    0.0079
     2        1.2576             nan     0.1000    0.0064
     3        1.2468             nan     0.1000    0.0052
     4        1.2383             nan     0.1000    0.0042
     5        1.2305             nan     0.1000    0.0038
     6        1.2244             nan     0.1000    0.0031
     7        1.2187             nan     0.1000    0.0028
     8        1.2137             nan     0.1000    0.0025
     9        1.2086             nan     0.1000    0.0024
    10        1.2042             nan     0.1000    0.0023
    20        1.1731             nan     0.1000    0.0010
    40        1.1423             nan     0.1000    0.0005
    60        1.1277             nan     0.1000    0.0002
    80        1.1201             nan     0.1000    0.0001
   100        1.1160             nan     0.1000    0.0000
   120        1.1136             nan     0.1000    0.0000
   140        1.1119             nan     0.1000   -0.0001
   160        1.1106             nan     0.1000   -0.0000
   180        1.1098             nan     0.1000    0.0000
   200        1.1092             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2667             nan     0.1000    0.0103
     2        1.2511             nan     0.1000    0.0073
     3        1.2374             nan     0.1000    0.0070
     4        1.2262             nan     0.1000    0.0053
     5        1.2164             nan     0.1000    0.0049
     6        1.2086             nan     0.1000    0.0037
     7        1.2014             nan     0.1000    0.0036
     8        1.1942             nan     0.1000    0.0035
     9        1.1886             nan     0.1000    0.0027
    10        1.1836             nan     0.1000    0.0026
    20        1.1469             nan     0.1000    0.0015
    40        1.1208             nan     0.1000    0.0003
    60        1.1125             nan     0.1000    0.0001
    80        1.1085             nan     0.1000   -0.0000
   100        1.1056             nan     0.1000   -0.0001
   120        1.1027             nan     0.1000   -0.0000
   140        1.1000             nan     0.1000    0.0000
   160        1.0985             nan     0.1000   -0.0000
   180        1.0962             nan     0.1000   -0.0000
   200        1.0945             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2643             nan     0.1000    0.0114
     2        1.2473             nan     0.1000    0.0084
     3        1.2313             nan     0.1000    0.0078
     4        1.2189             nan     0.1000    0.0061
     5        1.2085             nan     0.1000    0.0052
     6        1.1997             nan     0.1000    0.0039
     7        1.1915             nan     0.1000    0.0040
     8        1.1836             nan     0.1000    0.0036
     9        1.1771             nan     0.1000    0.0031
    10        1.1714             nan     0.1000    0.0023
    20        1.1355             nan     0.1000    0.0010
    40        1.1136             nan     0.1000    0.0002
    60        1.1052             nan     0.1000   -0.0000
    80        1.0997             nan     0.1000   -0.0001
   100        1.0963             nan     0.1000   -0.0001
   120        1.0933             nan     0.1000   -0.0001
   140        1.0900             nan     0.1000   -0.0000
   160        1.0870             nan     0.1000   -0.0001
   180        1.0837             nan     0.1000   -0.0000
   200        1.0803             nan     0.1000    0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2628             nan     0.1000    0.0118
     2        1.2433             nan     0.1000    0.0095
     3        1.2276             nan     0.1000    0.0075
     4        1.2144             nan     0.1000    0.0063
     5        1.2024             nan     0.1000    0.0057
     6        1.1920             nan     0.1000    0.0051
     7        1.1839             nan     0.1000    0.0040
     8        1.1766             nan     0.1000    0.0034
     9        1.1690             nan     0.1000    0.0034
    10        1.1626             nan     0.1000    0.0030
    20        1.1292             nan     0.1000    0.0008
    40        1.1090             nan     0.1000    0.0002
    60        1.1014             nan     0.1000    0.0000
    80        1.0951             nan     0.1000   -0.0001
   100        1.0904             nan     0.1000   -0.0000
   120        1.0856             nan     0.1000   -0.0001
   140        1.0804             nan     0.1000   -0.0001
   160        1.0761             nan     0.1000   -0.0000
   180        1.0722             nan     0.1000   -0.0001
   200        1.0681             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2703             nan     0.1000    0.0083
     2        1.2566             nan     0.1000    0.0068
     3        1.2456             nan     0.1000    0.0055
     4        1.2358             nan     0.1000    0.0049
     5        1.2275             nan     0.1000    0.0040
     6        1.2207             nan     0.1000    0.0032
     7        1.2149             nan     0.1000    0.0028
     8        1.2091             nan     0.1000    0.0027
     9        1.2043             nan     0.1000    0.0023
    10        1.1995             nan     0.1000    0.0022
    20        1.1683             nan     0.1000    0.0011
    40        1.1383             nan     0.1000    0.0004
    60        1.1232             nan     0.1000    0.0002
    80        1.1153             nan     0.1000    0.0001
   100        1.1115             nan     0.1000    0.0000
   120        1.1089             nan     0.1000   -0.0000
   140        1.1076             nan     0.1000   -0.0001
   160        1.1066             nan     0.1000   -0.0000
   180        1.1058             nan     0.1000   -0.0000
   200        1.1051             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2678             nan     0.1000    0.0100
     2        1.2508             nan     0.1000    0.0086
     3        1.2378             nan     0.1000    0.0064
     4        1.2260             nan     0.1000    0.0060
     5        1.2160             nan     0.1000    0.0050
     6        1.2071             nan     0.1000    0.0044
     7        1.1993             nan     0.1000    0.0036
     8        1.1923             nan     0.1000    0.0033
     9        1.1857             nan     0.1000    0.0031
    10        1.1804             nan     0.1000    0.0025
    20        1.1466             nan     0.1000    0.0010
    40        1.1177             nan     0.1000    0.0003
    60        1.1085             nan     0.1000    0.0001
    80        1.1042             nan     0.1000    0.0002
   100        1.1016             nan     0.1000   -0.0000
   120        1.0990             nan     0.1000   -0.0001
   140        1.0966             nan     0.1000   -0.0001
   160        1.0949             nan     0.1000   -0.0000
   180        1.0925             nan     0.1000   -0.0001
   200        1.0904             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2643             nan     0.1000    0.0108
     2        1.2459             nan     0.1000    0.0092
     3        1.2310             nan     0.1000    0.0074
     4        1.2178             nan     0.1000    0.0061
     5        1.2070             nan     0.1000    0.0057
     6        1.1971             nan     0.1000    0.0047
     7        1.1890             nan     0.1000    0.0038
     8        1.1812             nan     0.1000    0.0037
     9        1.1742             nan     0.1000    0.0034
    10        1.1681             nan     0.1000    0.0029
    20        1.1332             nan     0.1000    0.0009
    40        1.1116             nan     0.1000    0.0002
    60        1.1031             nan     0.1000   -0.0001
    80        1.0977             nan     0.1000    0.0002
   100        1.0940             nan     0.1000   -0.0001
   120        1.0904             nan     0.1000   -0.0001
   140        1.0865             nan     0.1000   -0.0001
   160        1.0831             nan     0.1000   -0.0000
   180        1.0801             nan     0.1000   -0.0000
   200        1.0774             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2634             nan     0.1000    0.0122
     2        1.2442             nan     0.1000    0.0097
     3        1.2273             nan     0.1000    0.0080
     4        1.2132             nan     0.1000    0.0068
     5        1.2017             nan     0.1000    0.0056
     6        1.1914             nan     0.1000    0.0050
     7        1.1819             nan     0.1000    0.0046
     8        1.1742             nan     0.1000    0.0035
     9        1.1675             nan     0.1000    0.0030
    10        1.1607             nan     0.1000    0.0033
    20        1.1260             nan     0.1000    0.0010
    40        1.1045             nan     0.1000    0.0002
    60        1.0957             nan     0.1000   -0.0001
    80        1.0904             nan     0.1000    0.0000
   100        1.0858             nan     0.1000   -0.0000
   120        1.0795             nan     0.1000   -0.0001
   140        1.0738             nan     0.1000   -0.0000
   160        1.0696             nan     0.1000   -0.0001
   180        1.0658             nan     0.1000   -0.0000
   200        1.0626             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2711             nan     0.1000    0.0083
     2        1.2576             nan     0.1000    0.0067
     3        1.2465             nan     0.1000    0.0055
     4        1.2369             nan     0.1000    0.0046
     5        1.2288             nan     0.1000    0.0038
     6        1.2224             nan     0.1000    0.0032
     7        1.2172             nan     0.1000    0.0026
     8        1.2115             nan     0.1000    0.0026
     9        1.2064             nan     0.1000    0.0025
    10        1.2022             nan     0.1000    0.0021
    20        1.1722             nan     0.1000    0.0007
    40        1.1418             nan     0.1000    0.0005
    60        1.1267             nan     0.1000    0.0002
    80        1.1193             nan     0.1000    0.0000
   100        1.1151             nan     0.1000   -0.0000
   120        1.1129             nan     0.1000   -0.0000
   140        1.1115             nan     0.1000   -0.0000
   160        1.1105             nan     0.1000   -0.0000
   180        1.1097             nan     0.1000   -0.0000
   200        1.1090             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2666             nan     0.1000    0.0103
     2        1.2505             nan     0.1000    0.0080
     3        1.2371             nan     0.1000    0.0065
     4        1.2264             nan     0.1000    0.0052
     5        1.2161             nan     0.1000    0.0049
     6        1.2075             nan     0.1000    0.0041
     7        1.1999             nan     0.1000    0.0037
     8        1.1932             nan     0.1000    0.0031
     9        1.1877             nan     0.1000    0.0026
    10        1.1822             nan     0.1000    0.0025
    20        1.1493             nan     0.1000    0.0010
    40        1.1212             nan     0.1000    0.0002
    60        1.1119             nan     0.1000   -0.0001
    80        1.1076             nan     0.1000    0.0000
   100        1.1052             nan     0.1000   -0.0000
   120        1.1031             nan     0.1000   -0.0000
   140        1.1006             nan     0.1000    0.0000
   160        1.0993             nan     0.1000   -0.0000
   180        1.0971             nan     0.1000   -0.0001
   200        1.0953             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2651             nan     0.1000    0.0114
     2        1.2468             nan     0.1000    0.0091
     3        1.2314             nan     0.1000    0.0071
     4        1.2187             nan     0.1000    0.0064
     5        1.2074             nan     0.1000    0.0053
     6        1.1984             nan     0.1000    0.0044
     7        1.1902             nan     0.1000    0.0038
     8        1.1832             nan     0.1000    0.0033
     9        1.1766             nan     0.1000    0.0030
    10        1.1704             nan     0.1000    0.0030
    20        1.1353             nan     0.1000    0.0007
    40        1.1134             nan     0.1000    0.0001
    60        1.1061             nan     0.1000   -0.0000
    80        1.1014             nan     0.1000   -0.0000
   100        1.0966             nan     0.1000   -0.0000
   120        1.0929             nan     0.1000   -0.0001
   140        1.0899             nan     0.1000   -0.0001
   160        1.0869             nan     0.1000   -0.0001
   180        1.0840             nan     0.1000   -0.0001
   200        1.0808             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2636             nan     0.1000    0.0119
     2        1.2436             nan     0.1000    0.0096
     3        1.2277             nan     0.1000    0.0078
     4        1.2146             nan     0.1000    0.0066
     5        1.2026             nan     0.1000    0.0059
     6        1.1923             nan     0.1000    0.0047
     7        1.1837             nan     0.1000    0.0043
     8        1.1759             nan     0.1000    0.0037
     9        1.1691             nan     0.1000    0.0031
    10        1.1631             nan     0.1000    0.0026
    20        1.1294             nan     0.1000    0.0008
    40        1.1087             nan     0.1000    0.0000
    60        1.0996             nan     0.1000   -0.0001
    80        1.0943             nan     0.1000   -0.0001
   100        1.0896             nan     0.1000    0.0000
   120        1.0852             nan     0.1000   -0.0001
   140        1.0815             nan     0.1000   -0.0001
   160        1.0770             nan     0.1000   -0.0001
   180        1.0724             nan     0.1000   -0.0001
   200        1.0681             nan     0.1000    0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2706             nan     0.1000    0.0082
     2        1.2573             nan     0.1000    0.0066
     3        1.2461             nan     0.1000    0.0053
     4        1.2376             nan     0.1000    0.0042
     5        1.2294             nan     0.1000    0.0041
     6        1.2227             nan     0.1000    0.0033
     7        1.2169             nan     0.1000    0.0029
     8        1.2117             nan     0.1000    0.0027
     9        1.2070             nan     0.1000    0.0022
    10        1.2026             nan     0.1000    0.0021
    20        1.1719             nan     0.1000    0.0009
    40        1.1436             nan     0.1000    0.0005
    60        1.1293             nan     0.1000    0.0002
    80        1.1220             nan     0.1000    0.0001
   100        1.1181             nan     0.1000    0.0000
   120        1.1160             nan     0.1000    0.0000
   140        1.1145             nan     0.1000   -0.0000
   160        1.1135             nan     0.1000    0.0000
   180        1.1125             nan     0.1000   -0.0000
   200        1.1118             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2668             nan     0.1000    0.0102
     2        1.2505             nan     0.1000    0.0081
     3        1.2372             nan     0.1000    0.0063
     4        1.2268             nan     0.1000    0.0049
     5        1.2178             nan     0.1000    0.0045
     6        1.2095             nan     0.1000    0.0040
     7        1.2022             nan     0.1000    0.0037
     8        1.1950             nan     0.1000    0.0035
     9        1.1886             nan     0.1000    0.0031
    10        1.1835             nan     0.1000    0.0025
    20        1.1514             nan     0.1000    0.0009
    40        1.1241             nan     0.1000    0.0002
    60        1.1158             nan     0.1000   -0.0000
    80        1.1120             nan     0.1000   -0.0001
   100        1.1092             nan     0.1000    0.0000
   120        1.1063             nan     0.1000   -0.0001
   140        1.1042             nan     0.1000   -0.0000
   160        1.1022             nan     0.1000   -0.0000
   180        1.1001             nan     0.1000   -0.0001
   200        1.0985             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2643             nan     0.1000    0.0114
     2        1.2465             nan     0.1000    0.0089
     3        1.2317             nan     0.1000    0.0076
     4        1.2188             nan     0.1000    0.0060
     5        1.2080             nan     0.1000    0.0051
     6        1.1991             nan     0.1000    0.0043
     7        1.1906             nan     0.1000    0.0040
     8        1.1840             nan     0.1000    0.0031
     9        1.1774             nan     0.1000    0.0030
    10        1.1715             nan     0.1000    0.0028
    20        1.1382             nan     0.1000    0.0012
    40        1.1171             nan     0.1000    0.0002
    60        1.1085             nan     0.1000   -0.0000
    80        1.1032             nan     0.1000   -0.0000
   100        1.0990             nan     0.1000    0.0002
   120        1.0954             nan     0.1000   -0.0000
   140        1.0918             nan     0.1000   -0.0001
   160        1.0887             nan     0.1000   -0.0001
   180        1.0859             nan     0.1000   -0.0001
   200        1.0817             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2639             nan     0.1000    0.0117
     2        1.2440             nan     0.1000    0.0099
     3        1.2272             nan     0.1000    0.0078
     4        1.2135             nan     0.1000    0.0066
     5        1.2022             nan     0.1000    0.0051
     6        1.1924             nan     0.1000    0.0046
     7        1.1837             nan     0.1000    0.0042
     8        1.1766             nan     0.1000    0.0034
     9        1.1696             nan     0.1000    0.0033
    10        1.1639             nan     0.1000    0.0027
    20        1.1309             nan     0.1000    0.0007
    40        1.1105             nan     0.1000    0.0001
    60        1.1015             nan     0.1000   -0.0001
    80        1.0952             nan     0.1000   -0.0001
   100        1.0904             nan     0.1000    0.0000
   120        1.0859             nan     0.1000   -0.0001
   140        1.0819             nan     0.1000   -0.0001
   160        1.0777             nan     0.1000   -0.0001
   180        1.0735             nan     0.1000   -0.0001
   200        1.0693             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2624             nan     0.1000    0.0120
     2        1.2430             nan     0.1000    0.0095
     3        1.2271             nan     0.1000    0.0078
     4        1.2132             nan     0.1000    0.0066
     5        1.2019             nan     0.1000    0.0054
     6        1.1923             nan     0.1000    0.0048
     7        1.1832             nan     0.1000    0.0042
     8        1.1755             nan     0.1000    0.0036
     9        1.1684             nan     0.1000    0.0031
    10        1.1628             nan     0.1000    0.0024
    20        1.1284             nan     0.1000    0.0008
    40        1.1103             nan     0.1000    0.0000
    60        1.1028             nan     0.1000   -0.0001
    80        1.0966             nan     0.1000   -0.0000
   100        1.0907             nan     0.1000   -0.0001
   120        1.0866             nan     0.1000   -0.0001
   140        1.0819             nan     0.1000    0.0002
   160        1.0780             nan     0.1000   -0.0001
   180        1.0731             nan     0.1000   -0.0000
   200        1.0695             nan     0.1000   -0.0001
```



```r
# Construct a list of model performance comparing the two models directly
resamps <- caret::resamples(list(RPART = rpart_model,
              GBM = gbm_model))
# Summarize it
summary(resamps)
```

```

Call:
summary.resamples(object = resamps)

Models: RPART, GBM 
Number of resamples: 10 

ROC 
           Min.  1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.6979645 0.703748 0.7127550 0.7127632 0.7210586 0.7317533    0
GBM   0.7255110 0.738801 0.7460459 0.7473505 0.7609042 0.7676751    0

Sens 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.8285061 0.8393674 0.8464177 0.8472780 0.8556463 0.8666159    0
GBM   0.8689024 0.8715944 0.8750476 0.8784491 0.8799543 0.8963415    0

Spec 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.3968023 0.4106914 0.4396802 0.4354376 0.4531250 0.4738372    0
GBM   0.3915575 0.4081478 0.4236919 0.4166859 0.4273256 0.4316860    0
```

```r
# plot it
# see ?resamples for more options
dotplot(resamps, metric = "ROC")
```

<img src="../figure/pa_2caretmodeleval-1.png" style="display: block; margin: auto;" />

This plot only tells us about our results on the data we have used in training. 
We are estimating our accuracy on the cross-validated resamples on the training 
data. 

## Test Your Results

Now we need to evaluate our performance on our hold-out set of data.


```r
# We use the pre-processing we defined on the training data, but this time
# we create test_data with these values:

test_data <- predict(pre_proc,
                     sea_data[test_idx, # specify the rows in the test set
                        c(continuous_x_vars, categ_vars)]) # keep the same vars

# Create a vector of outcomes for the test set
test_y <- sea_data[test_idx, "ontime_grad"]
test_y <- ifelse(test_y == 1, "grad", "nongrad")
test_y <- factor(test_y)

confusionMatrix(reference = test_y, data = predict(rpart_model, test_data))
```

```
Confusion Matrix and Statistics

          Reference
Prediction  grad nongrad
   grad    70025   24150
   nongrad 11062   18551
                                         
               Accuracy : 0.7155         
                 95% CI : (0.713, 0.7181)
    No Information Rate : 0.655          
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.3213         
                                         
 Mcnemar's Test P-Value : < 2.2e-16      
                                         
            Sensitivity : 0.8636         
            Specificity : 0.4344         
         Pos Pred Value : 0.7436         
         Neg Pred Value : 0.6264         
             Prevalence : 0.6550         
         Detection Rate : 0.5657         
   Detection Prevalence : 0.7608         
      Balanced Accuracy : 0.6490         
                                         
       'Positive' Class : grad           
                                         
```

```r
# Note that making predictions from the nb classifier can take a long time
# consider alternative models or making fewer predictions if this is a bottleneck
confusionMatrix(reference = test_y, data = predict(gbm_model, test_data))
```

```
Confusion Matrix and Statistics

          Reference
Prediction  grad nongrad
   grad    71111   24228
   nongrad  9976   18473
                                          
               Accuracy : 0.7237          
                 95% CI : (0.7212, 0.7262)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3361          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8770          
            Specificity : 0.4326          
         Pos Pred Value : 0.7459          
         Neg Pred Value : 0.6493          
             Prevalence : 0.6550          
         Detection Rate : 0.5745          
   Detection Prevalence : 0.7702          
      Balanced Accuracy : 0.6548          
                                          
       'Positive' Class : grad            
                                          
```

We can also make ROC plots of our test-set, out of sample, prediction accuracy.


```r
# ROC plots
yhat <- predict(rpart_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_2outofsampleROC-1.png" style="display: block; margin: auto;" />

```r
# NB ROC plot
# Note that making predictions from the nb classifier can take a long time
# consider alternative models or making fewer predictions if this is a bottleneck
yhat <- predict(gbm_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_2outofsampleROC-2.png" style="display: block; margin: auto;" />


## References and More

The `caret` package has extensive documentation which you can use to find
new models, learn about algorithms, and explore many other important aspects
of building predictive analytics. You can learn more here:

https://topepo.github.io/caret/

## Predicting on New Data

Here is a block of code that will allow you to make predictions on new data. To
do this we need to identify our model object and read in our new data for
current students. Then we simply make predictions. This is called model scoring.


```r
load("../data/montucky_current.rda")

current_data$frpl_7 <- factor(current_data$frpl_7)
current_data$male <- factor(current_data$male)
fact_vars <- predict(expand_factors, current_data)
# Get the column names of our categorical dummy variables
categ_vars <- colnames(fact_vars)
# Combine the new dummy variables with our original dataset
pred_data <- cbind(current_data, fact_vars)


pred_data <- predict(pre_proc,
                     pred_data[ ,
                        c(continuous_x_vars, categ_vars)]) # keep the same vars


preds_gbm <- predict(gbm_model, newdata = pred_data, type = "prob")$grad
preds_rpart <- predict(rpart_model, newdata = pred_data, type = "prob")$grad

current_data <- bind_cols(current_data,
                          data.frame(preds_gbm),
                          data.frame(preds_rpart))
str(current_data)
```

```
'data.frame':	34321 obs. of  36 variables:
 $ sid                 : chr  "0001" "0004" "0005" "0019" ...
 $ sch_g7_code         : chr  "01-01" "01-07" "01-07" "01-08" ...
 $ year                : num  2018 2018 2018 2018 2018 ...
 $ grade               : chr  "7" "7" "7" "7" ...
 $ scale_score_7_math  : num  30.9 35.4 40.8 40.3 42.8 ...
 $ scale_score_7_read  : num  32.4 40.8 33.6 46.9 25.6 ...
 $ age                 : num  12 12 12 12 12 12 12 12 12 12 ...
 $ frpl_7              : Factor w/ 4 levels "0","1","2","9": 2 2 1 1 2 3 1 1 1 1 ...
 $ ell_7               : num  0 0 0 0 0 0 0 0 0 0 ...
 $ iep_7               : num  0 0 0 0 0 1 0 1 0 0 ...
 $ gifted_7            : num  0 0 0 0 1 1 0 0 0 0 ...
 $ cohort_year         : num  2008 2008 2008 2008 2008 ...
 $ pct_days_absent_7   : num  0 0 0 0 8.89 ...
 $ male                : Factor w/ 2 levels "0","1": 2 1 1 1 NA 2 2 2 1 1 ...
 $ race_ethnicity      : chr  "White" "Hispan..." "Hispan..." "Hispan..." ...
 $ any_grad            : logi  NA NA NA NA NA NA ...
 $ disappeared         : logi  NA NA NA NA NA NA ...
 $ dropout             : logi  NA NA NA NA NA NA ...
 $ early_grad          : logi  NA NA NA NA NA NA ...
 $ late_grad           : logi  NA NA NA NA NA NA ...
 $ ontime_grad         : logi  NA NA NA NA NA NA ...
 $ still_enrolled      : logi  NA NA NA NA NA NA ...
 $ transferout         : logi  NA NA NA NA NA NA ...
 $ sch_g7_name         : chr  "Pike" "London Lane" "London Lane" "Highland" ...
 $ sch_g7_enroll       : num  17 241 241 16 2331 ...
 $ sch_g7_male_per     : num  0.585 0.468 0.468 0.504 0.605 ...
 $ sch_g7_frpl_per     : num  0.458 0.586 0.586 0.765 0.911 ...
 $ sch_g7_lep_per      : num  0 0.0192 0.0192 0.3213 0 ...
 $ sch_g7_gifted_per   : num  0.067 0.136 0.136 0.223 0 ...
 $ sch_g7_lea_id       : chr  "01" "01" "01" "01" ...
 $ sch_g7_poverty_desig: chr  "LowQuartile" "HighQuartile" "HighQuartile" "LowQuartile" ...
 $ vendor_ews_score    : logi  NA NA NA NA NA NA ...
 $ sch_g7_lea_name     : Factor w/ 45 levels "Allen","Beebe",..: 22 22 22 22 22 22 22 22 22 22 ...
 $ coop_name_g7        : chr  "Township" "Township" "Township" "Township" ...
 $ preds_gbm           : num  0.462 0.42 0.612 0.729 0.777 ...
 $ preds_rpart         : num  0.697 0.529 0.741 0.806 0.777 ...
```

## Advanced Techniques

Predictive analytic techniques are constantly being refined and evolving. This 
section will review a few techniques that I have found helpful when doing 
predictive analytic work with education data. 

### Regression Trees

Many of the most popular machine learning algorithms implemented on tabular data, 
data structured with rows and columns, are tree algorithms. Regression tree 
algorithms come in many different flavors (forests, boosted trees, pruned 
trees, etc.), but the basic strategy is to partition the data by splits in the 
available predictors so each branch of the tree uniquely identifies a specific 
combination of values of predictors that divide the outcome well between the 
two groups. 


```r
########################################################################
# Build and plot a regression tree
########################################################################
library(rpart.plot)
library(rpart)

# Basic partition tree model to plot
binary.model <- rpart(ontime_grad ~ scale_score_7_math +  
                       pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male +
                       race_ethnicity +
                       sch_g7_lep_per + sch_g7_gifted_per,
                     data = sea_data,
                     control = rpart.control(cp = .005))

# Pass the model object to the plot function
# You can look at ?rpart.plot to identify customizations you wish to use
rpart.plot(binary.model)
```

<img src="../figure/pa_2unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

```r
# To improve the labels, you can rename the variables in the data
sea_data$short_race_label <- NA
sea_data$short_race_label[sea_data$race_ethnicity == "Hispan..."] <- "H"
sea_data$short_race_label[sea_data$race_ethnicity == "White"] <- "W"
sea_data$short_race_label[sea_data$race_ethnicity == "Asian"] <- "A"
sea_data$short_race_label[sea_data$race_ethnicity == "Americ..."] <- "AI"
sea_data$short_race_label[sea_data$race_ethnicity == "Demogg..."] <- "M"
sea_data$short_race_label[sea_data$race_ethnicity == "Native..."] <- "HA"
sea_data$short_race_label[sea_data$race_ethnicity == "Black ..."] <- "B"

# Fit a classification model, just like a logistic regression formula interface
binary.model <- rpart(ontime_grad ~ scale_score_7_math +  
                        pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male +
                        short_race_label +
                        sch_g7_lep_per + sch_g7_gifted_per,
                      data = sea_data,
                      control = rpart.control(cp = .005))

# Make a plot showing the tree
rpart.plot(binary.model)
```

<img src="../figure/pa_2unnamed-chunk-3-2.png" style="display: block; margin: auto;" />

### Variable Importance

Another important diagnostic tool is understanding what variables/predictors are 
providing most of the differentiation between the two classes. In regression 
analysis, we can use regression coefficients. For many of the models that we can 
access using `caret`, there are no coefficients produced. Instead we have to 
use alternative methods to measure the influence that predictors have on the 
outcome. For many of the algorithms we can analyze using `caret` we can ask 
it to compute the "variable importance", a measure of the relative influence, 
for the predictors in the model. 

We can also plot these values in a graph and inspect it visually. 



```r
########################################################################
## Variable importance from a caret/train model object
########################################################################

## Fit a simple model just to show the syntax, you should be fitting a model
## split on train and test data, substitute your own model here.

# Unfortunately, we have to remove NAs first, because
# caret won't do it for us
train_data <- na.omit(sea_data[, c("ontime_grad", "scale_score_7_math",
                                   "pct_days_absent_7", "ell_7", "iep_7",
                                   "male", "frpl_7", "race_ethnicity",
                                   "sch_g7_lep_per", "sch_g7_gifted_per")])

# Outcome must be a factor in R to work with caret, so we recode
# Avoid 0/1 factor labels - this causes problems in fitting many model types.
train_data$ontime_grad <- ifelse(train_data$ontime_grad == 1, "Grad", "Nongrad")
train_data$ontime_grad <- factor(train_data$ontime_grad)

# Fit the model
caret_mod <- train(ontime_grad ~ .,
                   method = "rpart",
                   data = train_data,
                   metric = "ROC",
                   trControl = trainControl(summaryFunction = twoClassSummary,
                                            classProbs = TRUE,
                                            method = "cv"))

# Get variable importance, not available for all methods
caret::varImp(caret_mod)
```

```
rpart variable importance

                           Overall
sch_g7_gifted_per         100.0000
sch_g7_lep_per             80.4034
race_ethnicityWhite        75.3547
scale_score_7_math         54.6781
race_ethnicityHispan...    45.0056
frpl_71                    41.8795
race_ethnicityAsian        33.2363
iep_7                       8.0732
race_ethnicityNative...     0.6189
frpl_72                     0.3054
male1                       0.0000
`race_ethnicityBlack ...`   0.0000
frpl_79                     0.0000
race_ethnicityDemogr...     0.0000
ell_7                       0.0000
pct_days_absent_7           0.0000
```

```r
# Plot variable importance
plot(caret::varImp(caret_mod))
```

<img src="../figure/pa_2unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

### Probability Trade offs

Another 


```r
########################################################################
## Probablity accuracy
## Plot the relationship between predicted probabilities and observed
## graduation rates to explore thresholds.
########################################################################
library(yardstick)
library(dplyr)

# Get the predicted classifications from the caret model above
class_vec <- predict(caret_mod, newdata = train_data)
# Get the predicted probabilities from the model above
## Note that caret insists on giving probabilities for both classes, we
## need to store only one, in this case, the first one
prob_vec <- predict(caret_mod, newdata = train_data, type = "prob")[[1]] # only need first column

# Combine the true values, the estimated class, and the class probability
# into one dataframe for plotting
estimates_tbl <- data.frame(
  truth = as.factor(train_data$ontime_grad),
  estimate = as.factor(class_vec),
  class_prob = prob_vec
)

# Using this struture we can use the `yardstick` package to nicely
# compute a variety of performanc emetrics
## Confusion matrix
estimates_tbl %>% yardstick::conf_mat(truth, estimate)
```

```
          Truth
Prediction  Grad Nongrad
   Grad    72661   30429
   Nongrad  5600   10751
```

```r
# Accuracy
estimates_tbl %>% yardstick::metrics(truth, estimate)
```

```
# A tibble: 2 x 3
  .metric  .estimator .estimate
  <chr>    <chr>          <dbl>
1 accuracy binary         0.698
2 kap      binary         0.221
```

```r
# AUC
estimates_tbl %>% yardstick::roc_auc(truth, class_prob)
```

```
# A tibble: 1 x 3
  .metric .estimator .estimate
  <chr>   <chr>          <dbl>
1 roc_auc binary         0.638
```

```r
# ROC graph
# Plots the ROC curve (the ROC at all possible threshold values for the
# probability cutoff)

estimates_tbl %>% 
  roc_curve(truth, class_prob) %>% 
  autoplot()
```

<img src="../figure/pa_2tidyrocplots-1.png" style="display: block; margin: auto;" />

Another way to look at an individual estimated threshold is to graphically 
depict the confusion matrix using a mosaic plot (a visual crosstab). This code 
will help you do that:


```r
# Save the confusion matrix as an R object
conf_mat <- estimates_tbl %>% yardstick::conf_mat(truth, estimate)

# Plot the confusion matrix if you like
library(vcd)
labs <- round(prop.table(conf_mat$table), 2)
# Can change the margin to change the labels
mosaic(conf_mat$table, pop=FALSE)
labeling_cells(text = labs, margin = 0)(conf_mat$table)
```

<img src="../figure/pa_2plotconfusionmatrix-1.png" style="display: block; margin: auto;" />

### Probability vs. Outcome Plot

Another important way to compare models is to compare how the probability 
prediction they generate relates to the outcome. In some cases an increase in 
probability will correspond very closely to an increase in observed graduation 
rates, but in other cases, boundaries exist in the probability space where a 
small change in the probability threshold can result in a large shift in the 
classification rates of true positives and true negatives. This is a feature that 
is dependent on the algorithm you use and how it was tuned, so investigating this 
behavior is a good post-predictive check. 


```r
################################################################################
## Probability vs. outcome plot for predicted probabilities
################################################################################

plotdf <- estimates_tbl %>%
  mutate(prob_cut = percent_rank(class_prob)) %>%
  # depending on how many unique probabilities your model produces you can try
  # mutate(prob_cut = ntile(class_prob, 50)) %>%
  group_by(prob_cut) %>%
  summarize(
    avg_prob = mean(class_prob),
    prob_grad = sum(truth == "Grad") / length(truth),
    count = n())


library(ggplot2)
library(ggalt) # for fun lollipop charts

ggplot(plotdf, aes(x = avg_prob, y = prob_grad)) +
  ggalt::geom_lollipop() + theme_classic() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand=FALSE) +
  geom_smooth(se=FALSE)
```

<img src="../figure/pa_2probvsoutcomeplot-1.png" style="display: block; margin: auto;" />

### Make a Bowers Plot

One way to communicate the performance of your model is to benchmark it against 
the research literature on dropout early warning models. In a 2013 paper, Bowers, 
Sprott, and Taff placed the predictive performance of over 100 dropout risk 
indicators onto the ROC scale (true positive rate and false negative rate). 
Using this framework, we can recreate a plot from that paper and annotate it with 
the performance of our model to benchmark against previous studies. TO do this 
we need the data from Bowers et. al. and a data frame of 



```r
################################################################################
## Advanced: Make a Bowers plot
#################################################################################
bowers_plot() + 
  # Add your annotation here:
  annotate(geom = "point",
           x = 1 - pull(estimates_tbl %>% yardstick::spec(truth, estimate)),
           y = pull(estimates_tbl %>% yardstick::sens(truth, estimate)),
           size = 5, color = "red")
```

<img src="../figure/pa_2unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

```r
# Use a custom probability threshold
```
