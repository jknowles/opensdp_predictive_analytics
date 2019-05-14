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





## Introduction

### Objective

After completing this guide, the user will be able to fit and test a range 
of statistical models for predicting education outcomes. You will also be 
able to make predictions on future data and communicate the model fit and 
trade-offs of models with stakeholders. 

### Using this Guide

This guide uses synthetic data created by the OpenSDP synthetic data engine.
The data reflects student-level attainment data and is organized to be similar 
to the level of detail available at a state education agency - with a single 
row representing a student-grade-year. This guide does not cover the steps 
needed to clean raw student data. If you are interested in how to assemble 
data like this or the procedure used to generate the data, the code used 
is included in the data subdirectory. 


### Getting Started

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

### Setup

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

### Outline

Here are the steps:

1. Prepare the Data for Training
2. Fit New Models
3. Evaluate Model Fit
4. Explore the Fitted Model Objects
5. Compare Our Models to Others

In this guide, we will start from a data set that has been prepared, but if you
are applying this to work in your organization, it's a good idea to review the 
Introduction to Predictive Analytics in Education guide and walk through the 
process of reviewing and exploring data for predictive analytics. 

We're using multiple cohorts of middle-school students for the predictive analytics task -- students
who were seventh graders between 2007 and in 2011. These are the cohorts for which we have access to
reliable information on their high school graduation status (late graduate, on time graduate,
dropout, transferred out, disappeared). For this guide you are being given the entire set of data so
that you can explore different ways of organizing the data across cohorts. This guide explores
models and data from the earlier cohorts, and evaluates their performance on more recent cohorts.

One last point -- even though the data is synthetic, we have simulated missing data
for you. In the real world, you'll need to make predictions for every
student, even if you're missing data for that student which your model needs in
order to run. Just making predictions using a logistic regression won't be
enough. You'll need to use decision rules based on good data exploration and
your best judgment to predict and fill in outcomes for students where you have
insufficient data.


## Prepare the Data

### Beyond Logistic Regression

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
models using caret, it is a good practice to create the model matrix on our own. We do this for three
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

The reason we scale and center our continuous predictors last is that scaling and centering the
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

## Fit Models

### Prepare the Model Fitting Parameters

Once we have defined our predictor variables, we need to tell `train()` how we want to test our
models. Most of the algorithms offered through `caret` have "tuning parameters", user-controlled
values, that are not estimated from the data. Our goal is to experiment with these values and find
the values that fit the data the best. To do this, we must tell `train()` which values to try, and how
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
set.seed(2532) # set seed so models are comparable

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

### Train the Models

And then we are ready to fit our model to the data:

For the most part this is similar to fitting a model via `lm()` - we assign the model fit an object 
name and call `train()`. The distribution of parameters between `train()` and those we specified 
above in the `trControl()` object. The key parameters to `train()` are given below with comments 
explaining how each of them works. 


```r
# This will take quite some time:

#
rpart_model <- train(
  y = yvar[train_idx_small], # specify the outcome, here subset
  # to the smaller number of observations
  x = preds[train_idx_small, ], # specify the matrix of predictors, again subset
  method = "rpart",  # choose the algorithm - here a regression tree
  tuneLength = 12, # choose how many values of tuning parameters to test
  trControl = example_control, # set up the conditions for the test (defined above)
   metric = "ROC" # select the metric to choose the best model based on
  )
```

The `train()` function trains our model by splitting the data using a resampling strategy - in this 
case crossvalidation - and applying the algorithm of our choice to the resampled data. A key 
difference for algorithms fit through `train()` and a traditional logistic regression model is the 
presence of tuning parameters - model parameters not tied to data. These are options that we have 
to specify as the user. The `train()` function searches possible values of these parameters and 
learns the value that performs best on our data - this process is often called tuning. 
(TODO: LINK TO RESAMPLING RESOURCE)

The `rpart_model` object produced by `train()` includes not only the final tuned model, but also 
the results of the training experiment including the model performance for each value of the tuning 
parameter. Much of the art of predictive analytics is in defining the tuning parameter search 
space, choosing the correct model performance metric, and specifying the correct resampling strategy 
to correctly estimate model performance on future data. The result of this process is a meta-model 
that is estimating the likely performance of the model at different tuning parameter values in 
predicting future data. 

To fit a second model it's as easy as changing the `method` argument to caret and specifying a 
new method. In this case we will use the same training process, but we will tune fewer values 
of the tuning parameters for this new algorithm. (TODO: Link to caret method list)


```r
# Repeat above but just change the `method` argument to GBM for gradient boosted machines
treebag_model <- train(y = yvar[train_idx_small],
             x = preds[train_idx_small,],
             method = "treebag", tuneLength = 4,
             trControl = example_control,
             metric = "ROC")
```

This object uses the `treebag` method. Note that `caret` tries to provide a unified interface to 
over 200 algorithm types available in R. Not all of these algorithms are suitable for modeling 
all types of data and many of these methods will require the installation of additional R packages. 
When exploring your algorithm options it can be helpful to consult a textbook and to review available 
methods by their "family" to identify which algorithmic families may be most efficient for your 
problem at hand. 

#### Look at Our Models

When we print out the model objects we do not get a list of coefficients and table of statistical 
significance as we would with a linear model - instead we get the results of our model training 
experiment: a table of model performance statistics for every value of the tuning parameter. Notice 
that the `rpart_model` object has multiple rows of results, but the `treebag_model`, which has no 
tuning parameters to be learned from the data, has just a single row. 


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
Summary of sample sizes: 18000, 17999, 18001, 18000, 18000, 17999, ... 
Resampling results across tuning parameters:

  cp            ROC        Sens       Spec     
  0.0009563043  0.6938150  0.8816933  0.3932617
  0.0009808249  0.6938294  0.8810115  0.3937035
  0.0011769898  0.6946771  0.8925235  0.3751713
  0.0012505517  0.6951995  0.8942658  0.3714918
  0.0015447992  0.6958360  0.8968410  0.3664894
  0.0016674023  0.6957982  0.8955531  0.3697282
  0.0027953509  0.6929403  0.9035054  0.3513244
  0.0028689128  0.6925184  0.9042624  0.3477898
  0.0036045314  0.6903971  0.9124425  0.3295545
  0.0091216713  0.6877186  0.8970734  0.3491215
  0.0426658820  0.6793263  0.8366946  0.4270950
  0.0508312491  0.6206484  0.8699651  0.3110788

ROC was used to select the optimal model using the largest value.
The final value used for the model was cp = 0.001544799.
```

```r
print(treebag_model)
```

```
Bagged CART 

20000 samples
   17 predictor
    2 classes: 'grad', 'nongrad' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 18001, 18000, 18000, 18001, 18000, 17999, ... 
Resampling results:

  ROC        Sens       Spec    
  0.6945625  0.8242807  0.429012
```

The best value of the tuning parameter is chosen by selecting the value that optimizes the performance 
metric we specified to `trainControl()` - in this case, the `ROC` that is maximized. This value is 
only measured on the holdout set in our cross-validation startegy. 

## Model Comparison

Our next step is to compare our two models. One quick way to do this is to compare the prediction 
performance of each of the models across all of the folded holdout sets. This results in 10 
estimates of out of sample performance. We can plot these estimates to see how the distribution 
of out of sample performance metrics compares between two models. The `resamples()` function in 
caret makes this easy: 


```r
# Construct a list of model performance comparing the two models directly
resamps <- caret::resamples(list(RPART = rpart_model,
              TREEBAG = treebag_model))
# Summarize it
summary(resamps)
```

```

Call:
summary.resamples(object = resamps)

Models: RPART, TREEBAG 
Number of resamples: 10 

ROC 
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART   0.6860428 0.6895883 0.6951502 0.6958360 0.6982414 0.7080971    0
TREEBAG 0.6818932 0.6863913 0.6905270 0.6945625 0.6972216 0.7317117    0

Sens 
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART   0.8651515 0.8933712 0.8977651 0.8968410 0.9054924 0.9136364    0
TREEBAG 0.8022727 0.8153749 0.8261364 0.8242807 0.8320397 0.8433005    0

Spec 
             Min. 1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART   0.3382353 0.35625 0.3627675 0.3664894 0.3832268 0.3926471    0
TREEBAG 0.3991163 0.41250 0.4253130 0.4290120 0.4430147 0.4750000    0
```

```r
# plot it
# see ?resamples for more options
dotplot(resamps, metric = "ROC")
```

<img src="../figure/pa_2caretmodeleval-1.png" style="display: block; margin: auto;" />

This shows us that while the best performance of the `treebag_model` exceeded that 
of the `rpart_model` - there is quite a lot of overlap when the experiment is repeated ten 
times. This will often be the case and the amount of variability in out of sample 
performance (variance) is an important factor to take into account when selecting models. 
It isn't just simply a matter of taking the model that maximizes the cross-validated ROC - 
we may need to factor in the variability in that outcome and the cost in terms of time 
and computing power necessary to get that result. 

Ultimately, this plot only tells us about our results on the data we have used in training. 
We are estimating our accuracy on the cross-validated resamples on the training 
data. In our case, that means each model has it's performance estimated 10 times on a 2,000 row 
subset of the data (nrows / K) that was not seen for that particular tuning parameter. What 
we really care about is how the model performs on future cohorts, and this is why we left 
some of our data out of the model building process altogether as a validation set. It's time 
now to assess the performance of these two models on that future cohort. 

## Predict and Out of Sample Validation

Now we need to evaluate our performance on our hold-out set of data. Remember that we split our 
data up by time so our model has been trained on cohorts from the earlier years in our data. The 
true test of our model comes in seeing how well it predicts future cohorts of students - cohorts 
the model has never seen. This is where we see how robust the model is to changes in the measures 
used and secular patterns in student attainment. 

This is a little more involved than running the predict command - we need to transform our test 
data using the exact same process and parameters we used to transform the training data. Luckily 
we stored the `pre_proc` object which we can now apply to the test dataset. Then we recode the 
outcome variable for the test data, which we will use to compare our model predictions to. 

We can use a number of different approaches in R to evaluate model performance, but since we 
are already using the `caret` package we can use its built in `confusionMatrix()` function to 
compare the observed graduation status of students with the predictions from our models above 
on the new data object `test_data`. 


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

confusionMatrix(reference = test_y, data = predict(rpart_model, test_data), 
                positive = "nongrad", mode = "everything")
```

```
Confusion Matrix and Statistics

          Reference
Prediction  grad nongrad
   grad    72734   26841
   nongrad  8353   15860
                                          
               Accuracy : 0.7157          
                 95% CI : (0.7132, 0.7182)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.2991          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.3714          
            Specificity : 0.8970          
         Pos Pred Value : 0.6550          
         Neg Pred Value : 0.7304          
              Precision : 0.6550          
                 Recall : 0.3714          
                     F1 : 0.4740          
             Prevalence : 0.3450          
         Detection Rate : 0.1281          
   Detection Prevalence : 0.1956          
      Balanced Accuracy : 0.6342          
                                          
       'Positive' Class : nongrad         
                                          
```

```r
# Note that making predictions can take a long time
# consider alternative models or making fewer predictions if this is a bottleneck
confusionMatrix(reference = test_y, data = predict(treebag_model, test_data), 
                positive = "nongrad",  mode = "everything")
```

```
Confusion Matrix and Statistics

          Reference
Prediction  grad nongrad
   grad    69585   20339
   nongrad 11502   22362
                                          
               Accuracy : 0.7428          
                 95% CI : (0.7403, 0.7452)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4015          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.5237          
            Specificity : 0.8582          
         Pos Pred Value : 0.6603          
         Neg Pred Value : 0.7738          
              Precision : 0.6603          
                 Recall : 0.5237          
                     F1 : 0.5841          
             Prevalence : 0.3450          
         Detection Rate : 0.1806          
   Detection Prevalence : 0.2736          
      Balanced Accuracy : 0.6909          
                                          
       'Positive' Class : nongrad         
                                          
```

This allows us to compare the models on a wide range of possible performance metrics to see what 
tradeoffs there are between them. It looks like, despite having fewer tuning parameters, the 
`treebag_model` is outperforming the `rpart_model` on the validation set. 

Let's look at this graphically as well: We can also make ROC plots of our test-set, out of sample, prediction accuracy.


```r
# ROC plots
yhat <- predict(rpart_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_2outofsampleROC-1.png" style="display: block; margin: auto;" />

```r
# TB ROC plot
yhat <- predict(treebag_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_2outofsampleROC-2.png" style="display: block; margin: auto;" />


## Predicting on Future Data

Now that you've picked your model you need to be able to put it into production and predict the 
future outcomes for current students. This guide includes simulated current student data (student 
data without an observed outcome) so you can walk through the process of applying your trained 
models to this new data. 

This process is called model scoring. The key step here is that we have to pre-process the current 
data the exact same way we pre-processed the training data. In this case we can apply the `preproc` 
object we used above to the current data - assuming we are using the same variable names and 
have done the same work converting indicator data to factors. 


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


preds_treebag <- predict(treebag_model, newdata = pred_data, type = "prob")$grad
preds_rpart <- predict(rpart_model, newdata = pred_data, type = "prob")$grad

current_data <- bind_cols(current_data,
                          data.frame(preds_treebag),
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
 $ preds_treebag       : num  0.52 0.48 0.32 0.2 0.88 0.28 0.96 0.96 0.6 0.4 ...
 $ preds_rpart         : num  0.683 0.374 0.611 0.797 0.797 ...
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

# Get variable importance, not available for all methods
caret::varImp(rpart_model)
```

```
rpart variable importance

                            Overall
sch_g7_frpl_per           100.00000
frpl_7.0                   56.69920
race_ethnicityWhite        42.22320
frpl_7.1                   24.24906
scale_score_7_math         23.23764
race_ethnicityHispan...    13.78622
race_ethnicityAsian         9.98040
race_ethnicityBlack ...     9.96223
scale_score_7_read          4.86640
frpl_7.2                    1.26058
male.1                      0.29758
male.0                      0.29672
race_ethnicityAmeric...     0.28157
pct_days_absent_7           0.24250
frpl_7.9                    0.06089
`race_ethnicityBlack ...`   0.00000
race_ethnicityDemogr...     0.00000
race_ethnicityNative...     0.00000
```

```r
# Plot variable importance
plot(caret::varImp(rpart_model))
```

<img src="../figure/pa_2unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

```r
plot(caret::varImp(treebag_model))
```

<img src="../figure/pa_2unnamed-chunk-4-2.png" style="display: block; margin: auto;" />

The difference between the importance of the predictors in each model helps us to understand the 
different ways each model is arriving at its predictions. This is a convenient way to get an at a 
glance look at what is driving the predictions of our model inside the black box. (TODO: Link to 
other resources on this)

### Probability Trade offs

Another key aspect to predicting binary outcomes is deciding, as we saw in the first guide, how 
strong the predicted probability of an outcome has to be before we classify a student into a 
particular category. When we are working with binary outcomes, it can be helpful to use the ROC 
plot approach to evaluate at which point in the probability distribution we are identifying enough 
students without raising too many false alarms. 

The `yardstick` package for R provides a useful interface for quickly comparing many models on a 
number of performance metrics, including making tradeoff plots. 


```r
########################################################################
## Probablity accuracy
## Plot the relationship between predicted probabilities and observed
## graduation rates to explore thresholds.
########################################################################
library(yardstick)
library(dplyr)

# Get the predicted classifications from the caret model above
class_vec <- predict(treebag_model, newdata = test_data)
# Get the predicted probabilities from the model above
## Note that caret insists on giving probabilities for both classes, we
## need to store only one, in this case, the first one
prob_vec <- predict(treebag_model, newdata = test_data, type = "prob")[[1]] # only need first column

# Combine the true values, the estimated class, and the class probability
# into one dataframe for plotting
estimates_tbl <- data.frame(
  truth = test_y,
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
Prediction  grad nongrad
   grad    69585   20339
   nongrad 11502   22362
```

```r
# Accuracy
estimates_tbl %>% yardstick::metrics(truth, estimate)
```

```
# A tibble: 2 x 3
  .metric  .estimator .estimate
  <chr>    <chr>          <dbl>
1 accuracy binary         0.743
2 kap      binary         0.402
```

```r
# AUC
estimates_tbl %>% yardstick::roc_auc(truth, class_prob)
```

```
# A tibble: 1 x 3
  .metric .estimator .estimate
  <chr>   <chr>          <dbl>
1 roc_auc binary         0.775
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
    prob_grad = sum(truth == "grad") / length(truth),
    count = n())


library(ggplot2)
library(ggalt) # for fun lollipop charts

ggplot(plotdf, aes(x = avg_prob, y = prob_grad)) +
  ggalt::geom_lollipop() + theme_classic() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand=FALSE) +
  geom_smooth(se=FALSE) + 
  labs(x = "Mean Probability for Percentile", 
       y = "Probability of Student Graduating", 
       title = "Relationship between Increased Predicted Probability and Graduation Rates", 
       subtitle = "On Holdout Dataset",
       caption = "On Holdout Dataset for Treebag Model")
```

<img src="../figure/pa_2probvsoutcomeplot-1.png" style="display: block; margin: auto;" />

### Make a Bowers Plot

One way to communicate the performance of your model is to benchmark it against 
the research literature on dropout early warning models. In a 2013 paper, Bowers, 
Sprott, and Taff placed the predictive performance of over 100 dropout risk 
indicators onto the ROC scale (true positive rate and false negative rate). 
Using this framework, we can recreate a plot from that paper and annotate it with 
the performance of our model to benchmark against previous studies. To do this 
we need the data from Bowers et. al. and a data frame of estimates `estimates_tbl`. 



```r
# Requires internet connection
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

<img src="../figure/pa_2bowers_plot-1.png" style="display: block; margin: auto;" />

```r
# Use a custom probability threshold
```

## References and More

The `caret` package has extensive documentation which you can use to find
new models, learn about algorithms, and explore many other important aspects
of building predictive analytics. You can learn more here:

https://topepo.github.io/caret/
