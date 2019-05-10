---
title: "Predictive Analytics in Education - Building Models in R"
author: "Jared Knowles"
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

# Predictive Analytics in Education: Building Models

<div class="navbar navbar-default navbar-fixed-top" id="logo">
<div class="container">
<img src="https://opensdp.github.io/assets/images/OpenSDP-Banner_crimson.jpg" style="display: block; margin: 0 auto; height: 115px;">
</div>
</div>





### Objective

After completing this guide, the user will be able to fit and test a range 
of statistical models for predicting education outcomes. You will also be 
able to make predictions on future data and communicate the model fit and 
tradeoffs of models with stakeholders. 

### Using this Guide

This guide uses synthetic data created by the OpenSDP synthetic data engine.
The data reflects student-level attainment data and is organized to be similar 
to the level of detail available at a state education agency - with a single 
row representing a student-grade-year. This guide does not cover the steps 
needed to clean raw student data. If you are interested in how to assemble 
data like this or the procedure used to generate the data, the code used 
is included in the data subdirectory. 


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
This guide also includes some custom functions you can use to make your work 
easier - these are read in from the `functions.R` file in the `R` directory. 
For details on these functions - look there. 


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
sea_data <- as.data.frame(sea_data) # make sure the data is a data frame
# RESUME PREPPING DATA FROM GUIDE 1
sea_data$sid <- paste(sea_data$sid, sea_data$sch_g7_lea_id, sep = "-")
sea_data$scale_score_7_math[sea_data$scale_score_7_math < 0] <- NA
```



## Beyond Logistic Regression

In the last guide we ended up with a logistic regression model with several predictors.
Logistic regression is where you should start. It is fast to compute, easier to
interpret, and usually does a great job. However, there are a number of alternative
algorithms available to you and R provides a common interface to them through the
`caret` package. The `train` function (as in, training the models) is the
workhorse of the `caret` package. It has an extensive set of user-controlled
options.

Moving away from logistic regression comes with some additional complexity. While 
logistic regression is robust and stable to predictors on a wide variety of 
scales and is inexpensive to compute - other algorithms may not have all of 
these advantages, even when they are more accurate. So there are additional 
data preparation steps we should take. 

1. Build the model matrix
2. Divide the data into training and test sets
3. Scale and center predictors

The order of these operations is important. We want to avoid information from 
our test set of data being incorporated into our training data. This is 
called "information leakage" and it will lead us to be overly confident 
in strength of our model and to ultimately select the wrong model. 

### Build the Model Matrix

When we fit the logistic regression model we used R's formula interface to specify our model and let
R handle the process of turning our data into a matrix suitable for model estimation. When fitting
models using caret, it is a good practie to create the model matrix on our own. We do this for three
reasons - first, not all of the algorithms accessible via the `caret::train()` function work with
the formula interface. Having a model matrix allows us to take full advantage of the suite of models
`caret()` can access (and models outside of that as well such as `keras` models). Second, the matrix
approach is more memory efficient and avoids costly data transformation operations during the
fitting of models - a particularly important concern when dealing with datasets with many rows
and/or predictors. Third, it provides us a good opportunity to check our data again and identify
potential model building issues such as extremely sparse factor categories, duplicated variables, or
perfectly collinear predictors.

One workflow for doing this is the annotated R code below, which uses the data preparation functions 
built into the `caret` package to help build model matrices that can be applied to training and 
test data together.



```r
library(caret) # machine learning workhorse in R

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
```

### Training / Test Data

Now we need to define our training and test data split. There are a number of ways you can split 
your data to estimate your out-of-sample model accuracy, but here we will opt for a temporal split 
where we fit the model to the earliest years of data in our model and hold out the most recent 
year in our data for "testing" the model fit. 

#### Missingness

The caret package does not automatically handle missing data, so we have to do some additional work
when we define our training/test data split. You can choose additional ways to address missing data
(imputation, substituting mean values, etc.) - here we opt for the simple but aggressive strategy of
excluding any row with missing values for any predictor from being in the training data. 

#### Outliers

This is not directly covered here, but you will also want to ensure outliers are dropped from your 
training data at this stage (if not earlier). 



```r
train_idx <- row.names( # get the row.names to define our index
  na.omit(sea_data[sea_data$year %in% c(2003, 2004, 2005),
                   # reduce the data to rows of specific years and not missing data
                   c(continuous_x_vars, categ_vars)])
  # only select the predictors we are interested in
)
test_idx <- !row.names(sea_data[sea_data$year > 2005,]) %in% train_idx
```

### Scale and Center Predictors

The reason we scale and center our continuouse predictors last is that scaling and centering the
data is - in a way - fitting a model to the data. We do not want information from the test set to
carry over to the training set - which it would if we scaled and centered the entire dataset
together. So we get the scaling and centering parameters on the training set, and we then apply 
those parameters to both the training and the test data. 

When switching away from logistic regression, it is important to transform predictors to be centered
at 0 with a standard deviation of 1. This helps put binary and continuous indicators on a similar
scale and helps avoid problems associated with rounding, large numbers, and the optimization
algorithms used to evaluate model fit. Here is an example of how you can do this in R:



```r
# Fit the data transformation model to the continuous variables in our data
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

Once we have defined our predictor variables, we need to tell `train` how we want to test our
models. Most of the algorithms offered through `caret` have "tuning parameters", user-controlled
values, that are not estimated from the data. Our goal is to experiment with these values and find
the values that fit the data the best. To do this, we must tell `train` which values to try, and how
to evaluate their performance.

Luckily, `train()` has a number of sensible defaults that largely automate this process for us. For
the purpose of this exercise, a good set of defaults is to use the `twoClassSummary()` model
evaluation function (which tells us the area under the curve as well as the sensitivity,
specificity, and accuracy) and to use cross-fold validation. To set up our model training run 
we need to make three final steps:

1. Set up R to use the computing resources on our machine



```r
# Take advantage of all the processing power on your machine
library(doFuture)
plan(multiprocess(workers = 4)) # define the number of cpus to use
registerDoFuture() # register them with R
```

2. Prep our outcome variable

Caret really really really likes if you do binary classification that you
code the variables as factors with alphabetical labels. In this case, we
recode 0/1 to be nongrad, grad.



```r
yvar <- sea_data[train_idx, "ontime_grad"] # save only training observations
yvar <- ifelse(yvar == 1, "grad", "nongrad")
yvar <- factor(yvar)
```

3. Set up our model test parameters

Caret has a number of complex options, you can read about under `?trainControl`. 
Here we set some sensible defaults


```r
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

# On a standard desktop/laptop it can be necessary to decrease the sample size
# to train in a reasonable amount of time. For the prototype and getting feedback
# it's a good idea to stick with a reasonable sample size of under 20,000 rows.
# Let's do that here:

train_idx_small <- sample(1:nrow(preds), 2e4)
```

And then we are ready to fit our model to the data:

For the most part this is similar to fitting a model via `lm()` - we assign the model fit an object 
name and call `train()`. The distribution of parameters between `train()` and those we specified 
above in the `trControl()` object. The key parameters to `train()` are given below with comments 
explaining how each of them works. 


```r
# This will take quite some time:
set.seed(2532) # set seed so models are comparable

#
rpart_model <- train(
  y = yvar[train_idx_small], # specify the outcome, here subset
  # to the smaller number of observations
  x = preds[train_idx_small, ], # specify the matrix of predictors, again subset
  method = "rpart",  # choose the algorithm - here a regression tree
  tuneLength = 24, # choose how many values of tuning parameters to test
  trControl = example_control, # set up the conditions for the test (defined above)
   metric = "ROC" # select the metric to choose the best model based on
  )
```

To fit a second model it's as easy as changing the `method` argument to caret and specifying a 
new method. 


```r
# Repeat above but just change the `method` argument to GBM for gradient boosted machines
gbm_model <- train(y = yvar[train_idx_small],
             x = preds[train_idx_small,],
             method = "gbm", tuneLength = 4,
             trControl = example_control,
             metric = "ROC")
```

```
Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2769             nan     0.1000    0.0076
     2        1.2641             nan     0.1000    0.0064
     3        1.2529             nan     0.1000    0.0051
     4        1.2440             nan     0.1000    0.0044
     5        1.2367             nan     0.1000    0.0038
     6        1.2303             nan     0.1000    0.0031
     7        1.2251             nan     0.1000    0.0025
     8        1.2197             nan     0.1000    0.0024
     9        1.2143             nan     0.1000    0.0024
    10        1.2103             nan     0.1000    0.0018
    20        1.1786             nan     0.1000    0.0011
    40        1.1476             nan     0.1000    0.0005
    60        1.1320             nan     0.1000    0.0002
    80        1.1244             nan     0.1000    0.0001
   100        1.1199             nan     0.1000    0.0000
   120        1.1176             nan     0.1000    0.0000
   140        1.1162             nan     0.1000   -0.0000
   160        1.1151             nan     0.1000   -0.0000
   180        1.1143             nan     0.1000   -0.0000
   200        1.1135             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2719             nan     0.1000    0.0096
     2        1.2568             nan     0.1000    0.0074
     3        1.2436             nan     0.1000    0.0065
     4        1.2333             nan     0.1000    0.0051
     5        1.2234             nan     0.1000    0.0051
     6        1.2149             nan     0.1000    0.0041
     7        1.2069             nan     0.1000    0.0038
     8        1.1994             nan     0.1000    0.0034
     9        1.1935             nan     0.1000    0.0027
    10        1.1890             nan     0.1000    0.0021
    20        1.1528             nan     0.1000    0.0011
    40        1.1246             nan     0.1000    0.0004
    60        1.1156             nan     0.1000    0.0001
    80        1.1117             nan     0.1000   -0.0000
   100        1.1094             nan     0.1000   -0.0000
   120        1.1070             nan     0.1000   -0.0000
   140        1.1048             nan     0.1000   -0.0000
   160        1.1023             nan     0.1000   -0.0001
   180        1.0997             nan     0.1000   -0.0000
   200        1.0980             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2708             nan     0.1000    0.0107
     2        1.2543             nan     0.1000    0.0080
     3        1.2396             nan     0.1000    0.0073
     4        1.2270             nan     0.1000    0.0060
     5        1.2160             nan     0.1000    0.0055
     6        1.2063             nan     0.1000    0.0046
     7        1.1982             nan     0.1000    0.0039
     8        1.1904             nan     0.1000    0.0038
     9        1.1837             nan     0.1000    0.0031
    10        1.1778             nan     0.1000    0.0027
    20        1.1416             nan     0.1000    0.0010
    40        1.1175             nan     0.1000    0.0002
    60        1.1098             nan     0.1000   -0.0001
    80        1.1050             nan     0.1000   -0.0000
   100        1.1001             nan     0.1000    0.0000
   120        1.0969             nan     0.1000   -0.0001
   140        1.0940             nan     0.1000   -0.0001
   160        1.0910             nan     0.1000    0.0000
   180        1.0859             nan     0.1000   -0.0001
   200        1.0825             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2684             nan     0.1000    0.0116
     2        1.2505             nan     0.1000    0.0090
     3        1.2352             nan     0.1000    0.0078
     4        1.2215             nan     0.1000    0.0067
     5        1.2105             nan     0.1000    0.0054
     6        1.2005             nan     0.1000    0.0049
     7        1.1914             nan     0.1000    0.0045
     8        1.1840             nan     0.1000    0.0038
     9        1.1776             nan     0.1000    0.0032
    10        1.1711             nan     0.1000    0.0030
    20        1.1361             nan     0.1000    0.0010
    40        1.1133             nan     0.1000    0.0001
    60        1.1048             nan     0.1000    0.0000
    80        1.0983             nan     0.1000   -0.0001
   100        1.0938             nan     0.1000   -0.0001
   120        1.0883             nan     0.1000    0.0000
   140        1.0840             nan     0.1000   -0.0001
   160        1.0791             nan     0.1000   -0.0001
   180        1.0750             nan     0.1000    0.0001
   200        1.0703             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2764             nan     0.1000    0.0077
     2        1.2635             nan     0.1000    0.0062
     3        1.2527             nan     0.1000    0.0052
     4        1.2439             nan     0.1000    0.0045
     5        1.2362             nan     0.1000    0.0037
     6        1.2300             nan     0.1000    0.0031
     7        1.2240             nan     0.1000    0.0027
     8        1.2193             nan     0.1000    0.0022
     9        1.2145             nan     0.1000    0.0026
    10        1.2099             nan     0.1000    0.0021
    20        1.1784             nan     0.1000    0.0009
    40        1.1476             nan     0.1000    0.0005
    60        1.1326             nan     0.1000    0.0003
    80        1.1248             nan     0.1000    0.0001
   100        1.1207             nan     0.1000    0.0000
   120        1.1182             nan     0.1000    0.0000
   140        1.1167             nan     0.1000   -0.0000
   160        1.1156             nan     0.1000   -0.0000
   180        1.1148             nan     0.1000   -0.0000
   200        1.1140             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2729             nan     0.1000    0.0091
     2        1.2571             nan     0.1000    0.0079
     3        1.2447             nan     0.1000    0.0065
     4        1.2333             nan     0.1000    0.0055
     5        1.2232             nan     0.1000    0.0050
     6        1.2147             nan     0.1000    0.0043
     7        1.2070             nan     0.1000    0.0038
     8        1.1999             nan     0.1000    0.0034
     9        1.1940             nan     0.1000    0.0026
    10        1.1887             nan     0.1000    0.0025
    20        1.1534             nan     0.1000    0.0010
    40        1.1257             nan     0.1000    0.0003
    60        1.1154             nan     0.1000    0.0001
    80        1.1116             nan     0.1000   -0.0000
   100        1.1087             nan     0.1000    0.0000
   120        1.1057             nan     0.1000   -0.0001
   140        1.1027             nan     0.1000   -0.0001
   160        1.1006             nan     0.1000   -0.0001
   180        1.0975             nan     0.1000    0.0001
   200        1.0950             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2713             nan     0.1000    0.0104
     2        1.2545             nan     0.1000    0.0085
     3        1.2401             nan     0.1000    0.0069
     4        1.2270             nan     0.1000    0.0065
     5        1.2161             nan     0.1000    0.0055
     6        1.2066             nan     0.1000    0.0047
     7        1.1984             nan     0.1000    0.0040
     8        1.1908             nan     0.1000    0.0035
     9        1.1838             nan     0.1000    0.0034
    10        1.1783             nan     0.1000    0.0026
    20        1.1417             nan     0.1000    0.0009
    40        1.1175             nan     0.1000    0.0003
    60        1.1095             nan     0.1000    0.0003
    80        1.1035             nan     0.1000   -0.0000
   100        1.0993             nan     0.1000    0.0000
   120        1.0954             nan     0.1000   -0.0000
   140        1.0914             nan     0.1000   -0.0001
   160        1.0880             nan     0.1000   -0.0001
   180        1.0847             nan     0.1000   -0.0001
   200        1.0816             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2701             nan     0.1000    0.0105
     2        1.2515             nan     0.1000    0.0094
     3        1.2354             nan     0.1000    0.0076
     4        1.2219             nan     0.1000    0.0067
     5        1.2099             nan     0.1000    0.0057
     6        1.2001             nan     0.1000    0.0047
     7        1.1914             nan     0.1000    0.0042
     8        1.1839             nan     0.1000    0.0035
     9        1.1768             nan     0.1000    0.0032
    10        1.1703             nan     0.1000    0.0030
    20        1.1355             nan     0.1000    0.0009
    40        1.1133             nan     0.1000    0.0000
    60        1.1051             nan     0.1000   -0.0001
    80        1.0992             nan     0.1000   -0.0001
   100        1.0928             nan     0.1000   -0.0001
   120        1.0875             nan     0.1000   -0.0001
   140        1.0828             nan     0.1000   -0.0000
   160        1.0784             nan     0.1000   -0.0001
   180        1.0742             nan     0.1000   -0.0001
   200        1.0688             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2758             nan     0.1000    0.0078
     2        1.2628             nan     0.1000    0.0066
     3        1.2525             nan     0.1000    0.0052
     4        1.2432             nan     0.1000    0.0046
     5        1.2359             nan     0.1000    0.0037
     6        1.2289             nan     0.1000    0.0033
     7        1.2231             nan     0.1000    0.0028
     8        1.2183             nan     0.1000    0.0024
     9        1.2134             nan     0.1000    0.0025
    10        1.2093             nan     0.1000    0.0020
    20        1.1767             nan     0.1000    0.0013
    40        1.1459             nan     0.1000    0.0003
    60        1.1306             nan     0.1000    0.0002
    80        1.1227             nan     0.1000    0.0002
   100        1.1181             nan     0.1000    0.0000
   120        1.1159             nan     0.1000    0.0000
   140        1.1142             nan     0.1000   -0.0000
   160        1.1130             nan     0.1000    0.0000
   180        1.1122             nan     0.1000   -0.0000
   200        1.1116             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2727             nan     0.1000    0.0095
     2        1.2562             nan     0.1000    0.0079
     3        1.2431             nan     0.1000    0.0065
     4        1.2323             nan     0.1000    0.0053
     5        1.2209             nan     0.1000    0.0053
     6        1.2126             nan     0.1000    0.0039
     7        1.2048             nan     0.1000    0.0037
     8        1.1984             nan     0.1000    0.0031
     9        1.1925             nan     0.1000    0.0029
    10        1.1869             nan     0.1000    0.0027
    20        1.1501             nan     0.1000    0.0016
    40        1.1239             nan     0.1000    0.0003
    60        1.1137             nan     0.1000   -0.0000
    80        1.1094             nan     0.1000   -0.0000
   100        1.1067             nan     0.1000   -0.0000
   120        1.1039             nan     0.1000    0.0000
   140        1.1012             nan     0.1000   -0.0000
   160        1.0988             nan     0.1000    0.0000
   180        1.0974             nan     0.1000   -0.0000
   200        1.0943             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2695             nan     0.1000    0.0106
     2        1.2530             nan     0.1000    0.0084
     3        1.2381             nan     0.1000    0.0072
     4        1.2244             nan     0.1000    0.0067
     5        1.2128             nan     0.1000    0.0054
     6        1.2038             nan     0.1000    0.0044
     7        1.1946             nan     0.1000    0.0044
     8        1.1871             nan     0.1000    0.0034
     9        1.1804             nan     0.1000    0.0032
    10        1.1742             nan     0.1000    0.0027
    20        1.1387             nan     0.1000    0.0012
    40        1.1150             nan     0.1000    0.0001
    60        1.1072             nan     0.1000    0.0000
    80        1.1013             nan     0.1000    0.0000
   100        1.0981             nan     0.1000   -0.0001
   120        1.0943             nan     0.1000    0.0000
   140        1.0907             nan     0.1000    0.0001
   160        1.0866             nan     0.1000   -0.0001
   180        1.0828             nan     0.1000   -0.0001
   200        1.0795             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2684             nan     0.1000    0.0115
     2        1.2500             nan     0.1000    0.0094
     3        1.2344             nan     0.1000    0.0079
     4        1.2207             nan     0.1000    0.0062
     5        1.2087             nan     0.1000    0.0057
     6        1.1985             nan     0.1000    0.0047
     7        1.1899             nan     0.1000    0.0041
     8        1.1816             nan     0.1000    0.0038
     9        1.1747             nan     0.1000    0.0033
    10        1.1687             nan     0.1000    0.0029
    20        1.1337             nan     0.1000    0.0008
    40        1.1113             nan     0.1000    0.0002
    60        1.1017             nan     0.1000    0.0000
    80        1.0942             nan     0.1000   -0.0000
   100        1.0885             nan     0.1000   -0.0001
   120        1.0833             nan     0.1000   -0.0000
   140        1.0784             nan     0.1000   -0.0001
   160        1.0735             nan     0.1000   -0.0001
   180        1.0691             nan     0.1000   -0.0001
   200        1.0653             nan     0.1000    0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2768             nan     0.1000    0.0076
     2        1.2637             nan     0.1000    0.0064
     3        1.2530             nan     0.1000    0.0051
     4        1.2438             nan     0.1000    0.0044
     5        1.2365             nan     0.1000    0.0037
     6        1.2300             nan     0.1000    0.0030
     7        1.2240             nan     0.1000    0.0029
     8        1.2189             nan     0.1000    0.0025
     9        1.2139             nan     0.1000    0.0023
    10        1.2097             nan     0.1000    0.0020
    20        1.1784             nan     0.1000    0.0012
    40        1.1468             nan     0.1000    0.0004
    60        1.1315             nan     0.1000    0.0003
    80        1.1234             nan     0.1000    0.0001
   100        1.1186             nan     0.1000    0.0000
   120        1.1161             nan     0.1000    0.0000
   140        1.1146             nan     0.1000   -0.0000
   160        1.1135             nan     0.1000   -0.0000
   180        1.1127             nan     0.1000   -0.0000
   200        1.1119             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2724             nan     0.1000    0.0096
     2        1.2569             nan     0.1000    0.0076
     3        1.2446             nan     0.1000    0.0059
     4        1.2333             nan     0.1000    0.0055
     5        1.2232             nan     0.1000    0.0049
     6        1.2147             nan     0.1000    0.0042
     7        1.2065             nan     0.1000    0.0041
     8        1.2002             nan     0.1000    0.0028
     9        1.1946             nan     0.1000    0.0027
    10        1.1895             nan     0.1000    0.0024
    20        1.1524             nan     0.1000    0.0012
    40        1.1243             nan     0.1000    0.0003
    60        1.1154             nan     0.1000    0.0000
    80        1.1110             nan     0.1000    0.0000
   100        1.1084             nan     0.1000    0.0000
   120        1.1059             nan     0.1000   -0.0001
   140        1.1035             nan     0.1000   -0.0000
   160        1.1006             nan     0.1000   -0.0000
   180        1.0984             nan     0.1000   -0.0000
   200        1.0963             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2709             nan     0.1000    0.0102
     2        1.2540             nan     0.1000    0.0084
     3        1.2391             nan     0.1000    0.0072
     4        1.2259             nan     0.1000    0.0062
     5        1.2149             nan     0.1000    0.0051
     6        1.2054             nan     0.1000    0.0046
     7        1.1975             nan     0.1000    0.0039
     8        1.1897             nan     0.1000    0.0039
     9        1.1834             nan     0.1000    0.0031
    10        1.1779             nan     0.1000    0.0029
    20        1.1408             nan     0.1000    0.0010
    40        1.1166             nan     0.1000    0.0002
    60        1.1086             nan     0.1000   -0.0000
    80        1.1036             nan     0.1000    0.0000
   100        1.0988             nan     0.1000    0.0002
   120        1.0950             nan     0.1000    0.0002
   140        1.0917             nan     0.1000   -0.0000
   160        1.0881             nan     0.1000    0.0000
   180        1.0848             nan     0.1000    0.0000
   200        1.0811             nan     0.1000    0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2687             nan     0.1000    0.0115
     2        1.2494             nan     0.1000    0.0090
     3        1.2324             nan     0.1000    0.0078
     4        1.2194             nan     0.1000    0.0063
     5        1.2075             nan     0.1000    0.0056
     6        1.1978             nan     0.1000    0.0049
     7        1.1887             nan     0.1000    0.0046
     8        1.1810             nan     0.1000    0.0034
     9        1.1740             nan     0.1000    0.0033
    10        1.1683             nan     0.1000    0.0029
    20        1.1325             nan     0.1000    0.0008
    40        1.1112             nan     0.1000    0.0001
    60        1.1029             nan     0.1000    0.0000
    80        1.0970             nan     0.1000   -0.0001
   100        1.0923             nan     0.1000   -0.0001
   120        1.0870             nan     0.1000    0.0000
   140        1.0822             nan     0.1000   -0.0000
   160        1.0767             nan     0.1000    0.0000
   180        1.0721             nan     0.1000    0.0000
   200        1.0681             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2760             nan     0.1000    0.0075
     2        1.2631             nan     0.1000    0.0060
     3        1.2523             nan     0.1000    0.0051
     4        1.2437             nan     0.1000    0.0043
     5        1.2358             nan     0.1000    0.0038
     6        1.2295             nan     0.1000    0.0030
     7        1.2240             nan     0.1000    0.0027
     8        1.2192             nan     0.1000    0.0025
     9        1.2148             nan     0.1000    0.0022
    10        1.2104             nan     0.1000    0.0020
    20        1.1790             nan     0.1000    0.0012
    40        1.1479             nan     0.1000    0.0003
    60        1.1332             nan     0.1000    0.0002
    80        1.1249             nan     0.1000    0.0001
   100        1.1204             nan     0.1000    0.0001
   120        1.1181             nan     0.1000   -0.0000
   140        1.1166             nan     0.1000   -0.0000
   160        1.1156             nan     0.1000    0.0000
   180        1.1147             nan     0.1000   -0.0000
   200        1.1139             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2729             nan     0.1000    0.0095
     2        1.2571             nan     0.1000    0.0077
     3        1.2439             nan     0.1000    0.0066
     4        1.2329             nan     0.1000    0.0055
     5        1.2231             nan     0.1000    0.0048
     6        1.2147             nan     0.1000    0.0041
     7        1.2076             nan     0.1000    0.0035
     8        1.2013             nan     0.1000    0.0031
     9        1.1954             nan     0.1000    0.0029
    10        1.1901             nan     0.1000    0.0026
    20        1.1563             nan     0.1000    0.0012
    40        1.1267             nan     0.1000    0.0003
    60        1.1173             nan     0.1000    0.0003
    80        1.1130             nan     0.1000   -0.0000
   100        1.1100             nan     0.1000   -0.0001
   120        1.1077             nan     0.1000   -0.0000
   140        1.1059             nan     0.1000   -0.0001
   160        1.1029             nan     0.1000   -0.0001
   180        1.1004             nan     0.1000   -0.0001
   200        1.0978             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2704             nan     0.1000    0.0105
     2        1.2541             nan     0.1000    0.0080
     3        1.2396             nan     0.1000    0.0072
     4        1.2272             nan     0.1000    0.0059
     5        1.2154             nan     0.1000    0.0058
     6        1.2058             nan     0.1000    0.0047
     7        1.1974             nan     0.1000    0.0042
     8        1.1905             nan     0.1000    0.0030
     9        1.1840             nan     0.1000    0.0032
    10        1.1787             nan     0.1000    0.0027
    20        1.1424             nan     0.1000    0.0010
    40        1.1187             nan     0.1000    0.0001
    60        1.1109             nan     0.1000    0.0000
    80        1.1063             nan     0.1000   -0.0001
   100        1.1019             nan     0.1000   -0.0001
   120        1.0980             nan     0.1000   -0.0001
   140        1.0941             nan     0.1000   -0.0001
   160        1.0911             nan     0.1000   -0.0001
   180        1.0885             nan     0.1000   -0.0001
   200        1.0852             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2693             nan     0.1000    0.0110
     2        1.2508             nan     0.1000    0.0092
     3        1.2352             nan     0.1000    0.0073
     4        1.2223             nan     0.1000    0.0061
     5        1.2104             nan     0.1000    0.0059
     6        1.2003             nan     0.1000    0.0049
     7        1.1917             nan     0.1000    0.0040
     8        1.1842             nan     0.1000    0.0033
     9        1.1771             nan     0.1000    0.0035
    10        1.1707             nan     0.1000    0.0030
    20        1.1359             nan     0.1000    0.0013
    40        1.1132             nan     0.1000    0.0002
    60        1.1054             nan     0.1000   -0.0001
    80        1.0979             nan     0.1000    0.0001
   100        1.0913             nan     0.1000   -0.0001
   120        1.0870             nan     0.1000    0.0000
   140        1.0825             nan     0.1000   -0.0000
   160        1.0786             nan     0.1000   -0.0001
   180        1.0729             nan     0.1000   -0.0002
   200        1.0685             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2762             nan     0.1000    0.0075
     2        1.2632             nan     0.1000    0.0063
     3        1.2530             nan     0.1000    0.0051
     4        1.2444             nan     0.1000    0.0045
     5        1.2368             nan     0.1000    0.0037
     6        1.2304             nan     0.1000    0.0031
     7        1.2251             nan     0.1000    0.0026
     8        1.2204             nan     0.1000    0.0024
     9        1.2159             nan     0.1000    0.0021
    10        1.2118             nan     0.1000    0.0020
    20        1.1810             nan     0.1000    0.0012
    40        1.1509             nan     0.1000    0.0004
    60        1.1354             nan     0.1000    0.0002
    80        1.1280             nan     0.1000    0.0002
   100        1.1240             nan     0.1000    0.0000
   120        1.1216             nan     0.1000    0.0000
   140        1.1201             nan     0.1000   -0.0000
   160        1.1189             nan     0.1000   -0.0001
   180        1.1182             nan     0.1000   -0.0000
   200        1.1175             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2727             nan     0.1000    0.0096
     2        1.2573             nan     0.1000    0.0073
     3        1.2444             nan     0.1000    0.0062
     4        1.2329             nan     0.1000    0.0054
     5        1.2233             nan     0.1000    0.0048
     6        1.2156             nan     0.1000    0.0038
     7        1.2084             nan     0.1000    0.0036
     8        1.2016             nan     0.1000    0.0035
     9        1.1956             nan     0.1000    0.0029
    10        1.1908             nan     0.1000    0.0022
    20        1.1551             nan     0.1000    0.0015
    40        1.1285             nan     0.1000    0.0004
    60        1.1204             nan     0.1000    0.0000
    80        1.1155             nan     0.1000   -0.0001
   100        1.1127             nan     0.1000   -0.0001
   120        1.1097             nan     0.1000    0.0000
   140        1.1067             nan     0.1000   -0.0000
   160        1.1045             nan     0.1000    0.0001
   180        1.1022             nan     0.1000    0.0001
   200        1.1004             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2703             nan     0.1000    0.0107
     2        1.2535             nan     0.1000    0.0082
     3        1.2394             nan     0.1000    0.0069
     4        1.2280             nan     0.1000    0.0058
     5        1.2169             nan     0.1000    0.0050
     6        1.2073             nan     0.1000    0.0045
     7        1.1991             nan     0.1000    0.0040
     8        1.1912             nan     0.1000    0.0036
     9        1.1843             nan     0.1000    0.0032
    10        1.1783             nan     0.1000    0.0028
    20        1.1439             nan     0.1000    0.0008
    40        1.1203             nan     0.1000    0.0000
    60        1.1127             nan     0.1000    0.0003
    80        1.1078             nan     0.1000    0.0002
   100        1.1025             nan     0.1000   -0.0000
   120        1.0990             nan     0.1000   -0.0001
   140        1.0963             nan     0.1000   -0.0000
   160        1.0924             nan     0.1000   -0.0000
   180        1.0888             nan     0.1000   -0.0001
   200        1.0858             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2695             nan     0.1000    0.0108
     2        1.2512             nan     0.1000    0.0087
     3        1.2356             nan     0.1000    0.0077
     4        1.2224             nan     0.1000    0.0062
     5        1.2104             nan     0.1000    0.0058
     6        1.2012             nan     0.1000    0.0046
     7        1.1926             nan     0.1000    0.0042
     8        1.1858             nan     0.1000    0.0034
     9        1.1786             nan     0.1000    0.0035
    10        1.1730             nan     0.1000    0.0030
    20        1.1374             nan     0.1000    0.0013
    40        1.1160             nan     0.1000    0.0001
    60        1.1066             nan     0.1000    0.0000
    80        1.1006             nan     0.1000   -0.0000
   100        1.0950             nan     0.1000   -0.0001
   120        1.0894             nan     0.1000   -0.0000
   140        1.0855             nan     0.1000   -0.0001
   160        1.0814             nan     0.1000   -0.0000
   180        1.0771             nan     0.1000   -0.0001
   200        1.0737             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2763             nan     0.1000    0.0078
     2        1.2628             nan     0.1000    0.0063
     3        1.2521             nan     0.1000    0.0052
     4        1.2432             nan     0.1000    0.0043
     5        1.2357             nan     0.1000    0.0037
     6        1.2295             nan     0.1000    0.0031
     7        1.2244             nan     0.1000    0.0024
     8        1.2186             nan     0.1000    0.0027
     9        1.2137             nan     0.1000    0.0022
    10        1.2100             nan     0.1000    0.0019
    20        1.1790             nan     0.1000    0.0011
    40        1.1483             nan     0.1000    0.0005
    60        1.1334             nan     0.1000    0.0002
    80        1.1259             nan     0.1000    0.0001
   100        1.1217             nan     0.1000   -0.0000
   120        1.1192             nan     0.1000   -0.0000
   140        1.1179             nan     0.1000   -0.0000
   160        1.1169             nan     0.1000   -0.0000
   180        1.1160             nan     0.1000    0.0000
   200        1.1153             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2717             nan     0.1000    0.0095
     2        1.2569             nan     0.1000    0.0074
     3        1.2443             nan     0.1000    0.0063
     4        1.2338             nan     0.1000    0.0050
     5        1.2246             nan     0.1000    0.0044
     6        1.2153             nan     0.1000    0.0045
     7        1.2080             nan     0.1000    0.0034
     8        1.2014             nan     0.1000    0.0032
     9        1.1960             nan     0.1000    0.0027
    10        1.1899             nan     0.1000    0.0029
    20        1.1548             nan     0.1000    0.0011
    40        1.1272             nan     0.1000    0.0001
    60        1.1175             nan     0.1000    0.0001
    80        1.1130             nan     0.1000    0.0002
   100        1.1100             nan     0.1000   -0.0000
   120        1.1074             nan     0.1000    0.0001
   140        1.1055             nan     0.1000   -0.0001
   160        1.1030             nan     0.1000   -0.0001
   180        1.1005             nan     0.1000   -0.0000
   200        1.0985             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2714             nan     0.1000    0.0100
     2        1.2542             nan     0.1000    0.0089
     3        1.2399             nan     0.1000    0.0068
     4        1.2272             nan     0.1000    0.0063
     5        1.2161             nan     0.1000    0.0054
     6        1.2069             nan     0.1000    0.0044
     7        1.1995             nan     0.1000    0.0038
     8        1.1920             nan     0.1000    0.0034
     9        1.1856             nan     0.1000    0.0032
    10        1.1793             nan     0.1000    0.0032
    20        1.1424             nan     0.1000    0.0013
    40        1.1195             nan     0.1000    0.0003
    60        1.1120             nan     0.1000    0.0003
    80        1.1072             nan     0.1000   -0.0001
   100        1.1030             nan     0.1000    0.0000
   120        1.0992             nan     0.1000    0.0001
   140        1.0953             nan     0.1000    0.0001
   160        1.0913             nan     0.1000   -0.0001
   180        1.0873             nan     0.1000   -0.0001
   200        1.0845             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2692             nan     0.1000    0.0109
     2        1.2497             nan     0.1000    0.0092
     3        1.2329             nan     0.1000    0.0078
     4        1.2197             nan     0.1000    0.0063
     5        1.2084             nan     0.1000    0.0055
     6        1.1982             nan     0.1000    0.0048
     7        1.1903             nan     0.1000    0.0039
     8        1.1827             nan     0.1000    0.0037
     9        1.1763             nan     0.1000    0.0030
    10        1.1707             nan     0.1000    0.0026
    20        1.1364             nan     0.1000    0.0011
    40        1.1155             nan     0.1000    0.0002
    60        1.1052             nan     0.1000    0.0001
    80        1.0998             nan     0.1000   -0.0001
   100        1.0946             nan     0.1000   -0.0000
   120        1.0893             nan     0.1000   -0.0000
   140        1.0843             nan     0.1000   -0.0001
   160        1.0788             nan     0.1000   -0.0001
   180        1.0741             nan     0.1000   -0.0001
   200        1.0703             nan     0.1000   -0.0002

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2765             nan     0.1000    0.0077
     2        1.2638             nan     0.1000    0.0064
     3        1.2538             nan     0.1000    0.0052
     4        1.2454             nan     0.1000    0.0043
     5        1.2375             nan     0.1000    0.0039
     6        1.2309             nan     0.1000    0.0034
     7        1.2251             nan     0.1000    0.0028
     8        1.2199             nan     0.1000    0.0025
     9        1.2151             nan     0.1000    0.0024
    10        1.2112             nan     0.1000    0.0020
    20        1.1795             nan     0.1000    0.0012
    40        1.1494             nan     0.1000    0.0004
    60        1.1342             nan     0.1000    0.0003
    80        1.1264             nan     0.1000    0.0001
   100        1.1221             nan     0.1000   -0.0000
   120        1.1200             nan     0.1000   -0.0000
   140        1.1184             nan     0.1000   -0.0000
   160        1.1175             nan     0.1000   -0.0000
   180        1.1167             nan     0.1000   -0.0000
   200        1.1159             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2733             nan     0.1000    0.0093
     2        1.2575             nan     0.1000    0.0075
     3        1.2453             nan     0.1000    0.0060
     4        1.2337             nan     0.1000    0.0057
     5        1.2242             nan     0.1000    0.0048
     6        1.2157             nan     0.1000    0.0042
     7        1.2078             nan     0.1000    0.0040
     8        1.2012             nan     0.1000    0.0034
     9        1.1954             nan     0.1000    0.0027
    10        1.1903             nan     0.1000    0.0025
    20        1.1562             nan     0.1000    0.0011
    40        1.1281             nan     0.1000    0.0003
    60        1.1188             nan     0.1000    0.0000
    80        1.1142             nan     0.1000    0.0000
   100        1.1114             nan     0.1000    0.0000
   120        1.1090             nan     0.1000    0.0001
   140        1.1067             nan     0.1000   -0.0001
   160        1.1040             nan     0.1000   -0.0001
   180        1.1015             nan     0.1000   -0.0000
   200        1.0996             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2706             nan     0.1000    0.0101
     2        1.2528             nan     0.1000    0.0087
     3        1.2385             nan     0.1000    0.0068
     4        1.2255             nan     0.1000    0.0063
     5        1.2146             nan     0.1000    0.0055
     6        1.2051             nan     0.1000    0.0047
     7        1.1969             nan     0.1000    0.0042
     8        1.1899             nan     0.1000    0.0036
     9        1.1836             nan     0.1000    0.0028
    10        1.1781             nan     0.1000    0.0026
    20        1.1437             nan     0.1000    0.0014
    40        1.1202             nan     0.1000    0.0002
    60        1.1126             nan     0.1000   -0.0000
    80        1.1080             nan     0.1000    0.0002
   100        1.1041             nan     0.1000   -0.0000
   120        1.1007             nan     0.1000    0.0001
   140        1.0973             nan     0.1000   -0.0000
   160        1.0940             nan     0.1000   -0.0001
   180        1.0911             nan     0.1000   -0.0000
   200        1.0867             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2691             nan     0.1000    0.0106
     2        1.2501             nan     0.1000    0.0090
     3        1.2341             nan     0.1000    0.0078
     4        1.2209             nan     0.1000    0.0062
     5        1.2095             nan     0.1000    0.0055
     6        1.1996             nan     0.1000    0.0049
     7        1.1907             nan     0.1000    0.0043
     8        1.1833             nan     0.1000    0.0037
     9        1.1759             nan     0.1000    0.0033
    10        1.1701             nan     0.1000    0.0027
    20        1.1366             nan     0.1000    0.0009
    40        1.1149             nan     0.1000    0.0001
    60        1.1064             nan     0.1000   -0.0001
    80        1.1001             nan     0.1000   -0.0000
   100        1.0925             nan     0.1000   -0.0001
   120        1.0874             nan     0.1000   -0.0001
   140        1.0831             nan     0.1000    0.0000
   160        1.0786             nan     0.1000    0.0001
   180        1.0739             nan     0.1000   -0.0001
   200        1.0697             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2760             nan     0.1000    0.0077
     2        1.2626             nan     0.1000    0.0063
     3        1.2525             nan     0.1000    0.0049
     4        1.2435             nan     0.1000    0.0047
     5        1.2359             nan     0.1000    0.0038
     6        1.2288             nan     0.1000    0.0032
     7        1.2236             nan     0.1000    0.0026
     8        1.2186             nan     0.1000    0.0024
     9        1.2136             nan     0.1000    0.0025
    10        1.2098             nan     0.1000    0.0019
    20        1.1785             nan     0.1000    0.0010
    40        1.1477             nan     0.1000    0.0006
    60        1.1331             nan     0.1000    0.0003
    80        1.1252             nan     0.1000    0.0000
   100        1.1213             nan     0.1000   -0.0000
   120        1.1189             nan     0.1000   -0.0000
   140        1.1175             nan     0.1000    0.0000
   160        1.1165             nan     0.1000   -0.0000
   180        1.1158             nan     0.1000   -0.0000
   200        1.1151             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2731             nan     0.1000    0.0096
     2        1.2572             nan     0.1000    0.0076
     3        1.2443             nan     0.1000    0.0062
     4        1.2332             nan     0.1000    0.0056
     5        1.2237             nan     0.1000    0.0047
     6        1.2145             nan     0.1000    0.0047
     7        1.2069             nan     0.1000    0.0038
     8        1.2002             nan     0.1000    0.0031
     9        1.1951             nan     0.1000    0.0022
    10        1.1898             nan     0.1000    0.0025
    20        1.1532             nan     0.1000    0.0014
    40        1.1269             nan     0.1000    0.0002
    60        1.1183             nan     0.1000    0.0000
    80        1.1137             nan     0.1000   -0.0000
   100        1.1110             nan     0.1000   -0.0001
   120        1.1076             nan     0.1000   -0.0000
   140        1.1040             nan     0.1000   -0.0001
   160        1.1017             nan     0.1000   -0.0001
   180        1.0994             nan     0.1000    0.0000
   200        1.0974             nan     0.1000    0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2713             nan     0.1000    0.0101
     2        1.2531             nan     0.1000    0.0088
     3        1.2383             nan     0.1000    0.0071
     4        1.2251             nan     0.1000    0.0063
     5        1.2139             nan     0.1000    0.0055
     6        1.2045             nan     0.1000    0.0045
     7        1.1963             nan     0.1000    0.0038
     8        1.1893             nan     0.1000    0.0034
     9        1.1824             nan     0.1000    0.0033
    10        1.1768             nan     0.1000    0.0027
    20        1.1418             nan     0.1000    0.0011
    40        1.1196             nan     0.1000    0.0001
    60        1.1109             nan     0.1000    0.0003
    80        1.1069             nan     0.1000   -0.0001
   100        1.1017             nan     0.1000   -0.0001
   120        1.0978             nan     0.1000   -0.0001
   140        1.0947             nan     0.1000   -0.0001
   160        1.0912             nan     0.1000   -0.0001
   180        1.0874             nan     0.1000    0.0001
   200        1.0834             nan     0.1000    0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2703             nan     0.1000    0.0111
     2        1.2519             nan     0.1000    0.0094
     3        1.2365             nan     0.1000    0.0076
     4        1.2237             nan     0.1000    0.0060
     5        1.2116             nan     0.1000    0.0059
     6        1.2008             nan     0.1000    0.0050
     7        1.1923             nan     0.1000    0.0042
     8        1.1844             nan     0.1000    0.0038
     9        1.1773             nan     0.1000    0.0034
    10        1.1710             nan     0.1000    0.0027
    20        1.1363             nan     0.1000    0.0009
    40        1.1154             nan     0.1000    0.0000
    60        1.1066             nan     0.1000    0.0001
    80        1.1008             nan     0.1000   -0.0001
   100        1.0943             nan     0.1000   -0.0001
   120        1.0886             nan     0.1000   -0.0001
   140        1.0835             nan     0.1000   -0.0001
   160        1.0786             nan     0.1000   -0.0001
   180        1.0742             nan     0.1000   -0.0000
   200        1.0702             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2765             nan     0.1000    0.0075
     2        1.2641             nan     0.1000    0.0064
     3        1.2530             nan     0.1000    0.0052
     4        1.2444             nan     0.1000    0.0043
     5        1.2371             nan     0.1000    0.0035
     6        1.2308             nan     0.1000    0.0031
     7        1.2254             nan     0.1000    0.0025
     8        1.2201             nan     0.1000    0.0025
     9        1.2153             nan     0.1000    0.0025
    10        1.2111             nan     0.1000    0.0020
    20        1.1805             nan     0.1000    0.0010
    40        1.1482             nan     0.1000    0.0006
    60        1.1327             nan     0.1000    0.0002
    80        1.1246             nan     0.1000    0.0000
   100        1.1204             nan     0.1000   -0.0000
   120        1.1181             nan     0.1000   -0.0000
   140        1.1165             nan     0.1000   -0.0000
   160        1.1153             nan     0.1000   -0.0000
   180        1.1144             nan     0.1000   -0.0000
   200        1.1138             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2727             nan     0.1000    0.0097
     2        1.2580             nan     0.1000    0.0076
     3        1.2447             nan     0.1000    0.0066
     4        1.2337             nan     0.1000    0.0056
     5        1.2237             nan     0.1000    0.0046
     6        1.2160             nan     0.1000    0.0038
     7        1.2085             nan     0.1000    0.0037
     8        1.2022             nan     0.1000    0.0033
     9        1.1967             nan     0.1000    0.0025
    10        1.1914             nan     0.1000    0.0025
    20        1.1545             nan     0.1000    0.0011
    40        1.1264             nan     0.1000    0.0003
    60        1.1166             nan     0.1000   -0.0000
    80        1.1129             nan     0.1000   -0.0000
   100        1.1096             nan     0.1000   -0.0001
   120        1.1072             nan     0.1000   -0.0001
   140        1.1039             nan     0.1000    0.0002
   160        1.1016             nan     0.1000   -0.0001
   180        1.0992             nan     0.1000   -0.0001
   200        1.0976             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2705             nan     0.1000    0.0105
     2        1.2534             nan     0.1000    0.0085
     3        1.2390             nan     0.1000    0.0071
     4        1.2265             nan     0.1000    0.0059
     5        1.2154             nan     0.1000    0.0055
     6        1.2060             nan     0.1000    0.0044
     7        1.1971             nan     0.1000    0.0041
     8        1.1895             nan     0.1000    0.0037
     9        1.1833             nan     0.1000    0.0030
    10        1.1775             nan     0.1000    0.0026
    20        1.1410             nan     0.1000    0.0008
    40        1.1179             nan     0.1000    0.0002
    60        1.1099             nan     0.1000    0.0001
    80        1.1048             nan     0.1000   -0.0001
   100        1.1000             nan     0.1000   -0.0001
   120        1.0965             nan     0.1000    0.0000
   140        1.0932             nan     0.1000   -0.0000
   160        1.0903             nan     0.1000    0.0000
   180        1.0866             nan     0.1000   -0.0001
   200        1.0830             nan     0.1000   -0.0000

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2687             nan     0.1000    0.0116
     2        1.2498             nan     0.1000    0.0090
     3        1.2340             nan     0.1000    0.0076
     4        1.2206             nan     0.1000    0.0062
     5        1.2087             nan     0.1000    0.0057
     6        1.1989             nan     0.1000    0.0046
     7        1.1908             nan     0.1000    0.0039
     8        1.1832             nan     0.1000    0.0036
     9        1.1768             nan     0.1000    0.0031
    10        1.1702             nan     0.1000    0.0028
    20        1.1342             nan     0.1000    0.0011
    40        1.1117             nan     0.1000    0.0002
    60        1.1032             nan     0.1000   -0.0001
    80        1.0972             nan     0.1000    0.0001
   100        1.0913             nan     0.1000   -0.0000
   120        1.0869             nan     0.1000   -0.0001
   140        1.0818             nan     0.1000   -0.0000
   160        1.0769             nan     0.1000   -0.0001
   180        1.0728             nan     0.1000   -0.0001
   200        1.0687             nan     0.1000   -0.0001

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.2686             nan     0.1000    0.0114
     2        1.2510             nan     0.1000    0.0089
     3        1.2343             nan     0.1000    0.0078
     4        1.2208             nan     0.1000    0.0065
     5        1.2088             nan     0.1000    0.0057
     6        1.1997             nan     0.1000    0.0044
     7        1.1909             nan     0.1000    0.0041
     8        1.1825             nan     0.1000    0.0040
     9        1.1754             nan     0.1000    0.0032
    10        1.1695             nan     0.1000    0.0029
    20        1.1352             nan     0.1000    0.0010
    40        1.1139             nan     0.1000    0.0002
    60        1.1054             nan     0.1000   -0.0000
    80        1.0990             nan     0.1000   -0.0001
   100        1.0937             nan     0.1000   -0.0000
   120        1.0888             nan     0.1000   -0.0001
   140        1.0846             nan     0.1000   -0.0001
   160        1.0810             nan     0.1000   -0.0001
   180        1.0764             nan     0.1000   -0.0001
   200        1.0732             nan     0.1000   -0.0001
```

#### Look at Our Models


```r
print(rpart_model)
```

```
CART 

20000 samples
   17 predictor
    2 classes: 'grad', 'nongrad' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 18000, 17999, 18000, 17999, 18001, 18000, ... 
Resampling results across tuning parameters:

  cp            ROC        Sens       Spec     
  0.0003957974  0.6991685  0.8561892  0.4182514
  0.0004317789  0.6950887  0.8593296  0.4181059
  0.0004893495  0.6999576  0.8574902  0.4265976
  0.0005037421  0.6976120  0.8587919  0.4240076
  0.0005469200  0.6947720  0.8614738  0.4195460
  0.0005757052  0.6942016  0.8658410  0.4168117
  0.0006716561  0.6962912  0.8678320  0.4160923
  0.0007196315  0.6965223  0.8737317  0.4070236
  0.0007676070  0.6966751  0.8722761  0.4070242
  0.0007915947  0.6966751  0.8722761  0.4070242
  0.0008155824  0.6983517  0.8751117  0.4032814
  0.0008635579  0.6984944  0.8725070  0.4088929
  0.0009115333  0.6989847  0.8704389  0.4133550
  0.0010074842  0.6987327  0.8752661  0.4071627
  0.0011514105  0.6977997  0.8778705  0.4001140
  0.0012233736  0.6979492  0.8775640  0.4005457
  0.0012953368  0.6977523  0.8781767  0.4005480
  0.0015112263  0.6983851  0.8780236  0.4037138
  0.0037420841  0.6922384  0.8787155  0.3940675
  0.0038860104  0.6915234  0.8805536  0.3884560
  0.0044617156  0.6869029  0.9010839  0.3432608
  0.0089953944  0.6800196  0.8977158  0.3373620
  0.0378526195  0.6740644  0.8443140  0.4070112
  0.0541882556  0.6542136  0.8433990  0.3881241

ROC was used to select the optimal model using the largest value.
The final value used for the model was cp = 0.0004893495.
```

```r
print(gbm_model)
```

```
Stochastic Gradient Boosting 

20000 samples
   17 predictor
    2 classes: 'grad', 'nongrad' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 17999, 18000, 18000, 18000, 18001, 18001, ... 
Resampling results across tuning parameters:

  interaction.depth  n.trees  ROC        Sens       Spec     
  1                   50      0.7345248  0.9053017  0.3380831
  1                  100      0.7396466  0.8857643  0.3917706
  1                  150      0.7403262  0.8785626  0.4040016
  1                  200      0.7403586  0.8775671  0.4080321
  2                   50      0.7398561  0.8842325  0.3870193
  2                  100      0.7416136  0.8716681  0.4173864
  2                  150      0.7427501  0.8713618  0.4227125
  2                  200      0.7435793  0.8731237  0.4255895
  3                   50      0.7420563  0.8761884  0.4096131
  3                  100      0.7448199  0.8725114  0.4270276
  3                  150      0.7459056  0.8722820  0.4270284
  3                  200      0.7472254  0.8721290  0.4312025
  4                   50      0.7423445  0.8759585  0.4096133
  4                  100      0.7447433  0.8708267  0.4264533
  4                  150      0.7465366  0.8732785  0.4280368
  4                  200      0.7484963  0.8723591  0.4343688

Tuning parameter 'shrinkage' was held constant at a value of 0.1

Tuning parameter 'n.minobsinnode' was held constant at a value of 10
ROC was used to select the optimal model using the largest value.
The final values used for the model were n.trees = 200, interaction.depth =
 4, shrinkage = 0.1 and n.minobsinnode = 10.
```


## Model Comparison

Our next step is to compare our two models. One quick way to do this is to compare the prediction 
performance of each of the models across all of the folds.


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
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.6631522 0.6799713 0.7046453 0.6999576 0.7178448 0.7302726    0
GBM   0.7283823 0.7457027 0.7490227 0.7484963 0.7538740 0.7646306    0

Sens 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.8252874 0.8524904 0.8540230 0.8574902 0.8675091 0.8851455    0
GBM   0.8468606 0.8643678 0.8682503 0.8723591 0.8854406 0.8896552    0

Spec 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.4028777 0.4109417 0.4187050 0.4265976 0.4346423 0.4791367    0
GBM   0.4057554 0.4198738 0.4352518 0.4343688 0.4465252 0.4618705    0
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
   grad    70641   24532
   nongrad 10446   18169
                                          
               Accuracy : 0.7174          
                 95% CI : (0.7149, 0.7199)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3218          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8712          
            Specificity : 0.4255          
         Pos Pred Value : 0.7422          
         Neg Pred Value : 0.6349          
             Prevalence : 0.6550          
         Detection Rate : 0.5707          
   Detection Prevalence : 0.7688          
      Balanced Accuracy : 0.6483          
                                          
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
   grad    70973   24072
   nongrad 10114   18629
                                          
               Accuracy : 0.7238          
                 95% CI : (0.7213, 0.7263)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3377          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8753          
            Specificity : 0.4363          
         Pos Pred Value : 0.7467          
         Neg Pred Value : 0.6481          
             Prevalence : 0.6550          
         Detection Rate : 0.5733          
   Detection Prevalence : 0.7678          
      Balanced Accuracy : 0.6558          
                                          
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
 $ preds_gbm           : num  0.441 0.375 0.543 0.654 0.773 ...
 $ preds_rpart         : num  0.333 0.418 0.657 0.835 0.695 ...
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
ell_7                       0.0000
frpl_79                     0.0000
race_ethnicityDemogr...     0.0000
pct_days_absent_7           0.0000
`race_ethnicityBlack ...`   0.0000
male1                       0.0000
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
