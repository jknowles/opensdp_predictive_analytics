---
title: "Introduction to Predictive Analytics in Education"
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

### Objective

After completing this guide, the user will be familiar with the most common 
principles and technique of predictive analytics and how they can be applied 
to education data analysis problems - particularly through the example of 
student "early warning indicators". 

### Using this Guide

This guide uses synthetic data created by the OpenSDP synthetic data engine.
The data reflects student-level attainment data and is organized to be similar 
to the level of detail available at a state education agency - with a single 
row representing a student-grade-year. This guide does not cover the steps 
needed to clean raw student data. If you are interested in how to assemble 
data like this or the procedure used to generate the data, the code used 
is included in the data subdirectory. 






## Introduction

For this guide, we will be building a model to predict on-time high school
graduation using 7th grade data. The problem definition phase is often the
most important when developing a predictive model, so when applying these
lessons to your own work consider the outcome to be predicted and the measures
available. You can learn more about the importance of data preparation
by reading this guide to EWS development.

This guide will provide you with a baseline model to introduce you to the main
concepts of predictive analytics. From there, you will learn techniques needed
to interpret, understand, and iteratively improve the model.

This guide demonstrates the techniques of predictive analytics using a synthetic
data set. The data in this guide were generated using the OpenSDP synthetic data
generator. The code to generate the data is available in the `data` directory.
The data are completely synthetic student-level records and are free to report
and distribute - no FERPA concerns here. The one caveat is that the relationships
between predictors and the outcome in the data may not be typical of the
relationships you would find in your local data set. In particular, there is
less room for model improvement in this synthetic data than is typical in observed
education data. But, for learning the ropes, it will suit us just fine!

## Outline

Here are the steps:

1. explore and validate the data
2. examine the relationship between predictors and outcomes
3. evaluate the predictive power of different variables and select predictors for your model
4. make predictions using logistic regression
5. convert the predicted probabilities into a 0/1 indicator
6. look at the effect of different probability cutoffs on prediction accuracy 
7. Survey advanced techniques for model comparison and model fitting

This guide will take you through these steps using a baseline model for
predicting graduation using student-level grade 7 records. While this guide
focuses on a predictive analytic approach, it is good when evaluating different
methods for predicting student outcomes to consider alternative approaches. For
example, you should consider the "checklist" approaches
created at the Chicago Consortium of School
Research (CCSR). With that "checklist" approach, you experiment with different
thresholds for your predictor variables, and combine them to directly predict
the outcome. This approach has the advantage of being easy to explain and
implement, but it might not yield the most accurate predictions. We won't
demonstrate that approach here, but we will show how to compare the predictive
power of different modeling approaches so that you can make an informed decision
about the trade offs between transparency and accuracy.

In this guide, we will start from a data set that has been prepared, but if you
are applying this to work in your organization, it's good to consider a few
questions about the prediction task before you pull together your data and start
fitting models. First off, take time to think about variables, time, and data
sets. The sooner in a student's academic trajectory you can make a prediction,
the sooner you can intervene -- but the less accurate your predictions will be.
What data, and specifically which variables, do you have available to make
predictions? What outcome are you trying to predict?

In the case of this guide, we are limiting our focus to using data available
at the end of grade 7 to predict outcomes at the end of high school. A critical
step in developing a predictive model is to identify the time points that
different measures are collected during the process you are predicting -- you
can't use data from the future to make predictions. If you're planning to use
your model to make predictions for students at the end of 11th grade, for
instance, and if most students take AP classes as seniors, you can't use data
about AP course-taking collected during senior year to predict the likelihood of
college enrollment, even if you have that data available for past groups of
students.

We're using multiple cohorts of middle-school students for the predictive
analytics task -- students who were seventh graders between 2007 and in 2011.
These are the cohorts for which we have access to reliable information on their
high school graduation status (late graduate, on time graduate, dropout,
transferred out, disappeared). For this guide you are being given the entire set
of data so that you can explore different ways of organizing the data across
cohorts. An example is that you may choose to explore your models and data using
the earlier cohort, and evaluate their performance on more recent cohorts.

One last point -- even though the data is synthetic, we have simulated missing data
for you. In the real world, you'll need to make predictions for every
student, even if you're missing data for that student which your model needs in
order to run. Just making predictions using a logistic regression won't be
enough. You'll need to use decision rules based on good data exploration and
your best judgment to predict and fill in outcomes for students where you have
insufficient data.

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

# Load the helper functions not in packages
source("../R/functions.R")

# Read in the data
# This command assumes that the data is in a folder called data, below your
# current working directory. You can check your working directory with the
# getwd() command, and you can set your working directory using the RStudio
# environment, or the setwd() command.

load("../data/montucky.rda")
```
## Explore the Data

### Uniqueness

Ensure that the data imported correctly.

First, check whether the the data is unique by student ID.


```r
nrow(sea_data) == n_distinct(sea_data$sid)
```

```
[1] FALSE
```

Wait, what is the issue here?


```r
table(sea_data$sid == sea_data$sid[[1]]) # test how many times the first sid appears
```

```

 FALSE   TRUE 
123743     45 
```

Why might our IDs be repeated 45 times? Let's look at how many LEAs we have in
our SEA data set:


```r
length(unique(sea_data$sch_g7_lea_id)) # test how many LEAs are in our data
```

```
[1] 45
```

We see that our student IDs are not unique by LEA. That's an easy enough
fix.


```r
nrow(sea_data) == n_distinct(sea_data$sid, sea_data$sch_g7_lea_id)
```

```
[1] TRUE
```

Let's append the LEA ID onto the student ID to make student IDs truly unique:


```r
sea_data$sid <- paste(sea_data$sid, sea_data$sch_g7_lea_id, sep = "-")
```


```r
nrow(sea_data) == n_distinct(sea_data$sid)
```

```
[1] TRUE
```

### Defining the Outcome

A key initial task in building an EWS is to identify the cohort membership of
students. When we build a predictive model need to identify two time points --
when we will be making the prediction, and when the outcome we are predicting
will be observed. In this case, we will be making the prediction upon receiving
the 7th grade data on students (at the end of 7th grade), and we will
be predicting their completion of high school.

Let's focus first on identifying the 7th grade year for each student. We have
three year variables, what is their relationship:


```r
table(sea_data$year)
```

```

 2001  2002  2003  2004  2005  2006  2007  2008  2009  2010  2011 
   10    91  5831 27466 34072 31210 23616  1267   215     9     1 
```

```r
table(sea_data$cohort_year)
```

```

 2003  2004  2005  2006  2007  2008  2009  2010  2011  2012  2013 
   10    91  5831 27466 34072 31210 23616  1267   215     9     1 
```

```r
table(sea_data$cohort_grad_year)
```

```

 2006  2007  2008  2009  2010  2011  2012  2013  2014  2015  2016 
   10    91  5831 27466 34072 31210 23616  1267   215     9     1 
```

From the data dictionary we know that the first year variable is the year
that corresponds with the student entering 7th grade (`year`). The `cohort_year`
variable defines the 9th grade cohort a student belongs to for measuring on-time
graduation. Finally, the `cohort_grad_year` is the year that the cohort a
student is a member of should graduate to be considered "on-time".

If a student graduates, their year of graduation is recorded as `year_of_graduation`.

The definition of graduation types is an important design decision for our
predictive analytic system. We have to set a time by which students are graduated
so we know whether to count them as graduating or not completing. The state of
Montucky uses a 4-year cohort graduation rate for most reporting, defining
on-time graduation as graduation within 4 years of entering high school. This
variable is defined as:


```r
table(sea_data$year_of_graduation == sea_data$cohort_grad_year)
```

```

FALSE  TRUE 
 1889 81087 
```

```r
table(sea_data$ontime_grad)
```

```

    0     1 
42701 81087 
```

This is an example of a business rule - it's a restriction on the definition
of the data we make so that we can consistently use the data. In this case, it
is necessary so we can definitively group students for the purposes of predicting
their outcomes. You could consider alternative graduation timelines, for example:


```r
table(sea_data$year_of_graduation <= sea_data$cohort_grad_year + 1)
```

```

FALSE  TRUE 
  211 82765 
```

What does this rule say? How is it different than the definition of on-time above?

### Structure and Geography

Now that we have a sense of the time structure of the data, let's look at
geography. How many high schools and how many districts are? What are those
regional education services coops?

We are going to be building a model for the an entire state, but stakeholders
may have questions about how the model works for particular schools, districts,
or regions. Let's practice exploring the data by these different geographies.



```r
length(unique(sea_data$first_hs_name))
```

```
[1] 297
```

```r
length(unique(sea_data$first_hs_lea_id))
```

```
[1] 46
```

```r
table(sea_data$coop_name_g7, useNA = "always")
```

```

     Angelea        Birch     Caldwell Cold Springs         Hope       Marvel 
       15725        13444        17385        15668        12629        14983 
     Monarch       Weston  Wintergreen         <NA> 
       12139        10131        11684            0 
```

For this exercise, districts in Montucky are organized into cooperative regions.
Cooperative regions are just groups of LEAs. It may be helpful to compare how
our model performs in a given school, LEA, or cooperative region to the rest of the
data in the state. As an example of this kind of analysis, select a specific
coop region and explore its data, drawing comparisons to the full data set. Which
districts are part of this coop region and how many students do they have?
Substitute different abbreviation codes for different coops and then replace the
`my_coop` variable below.


```r
my_coop <- sea_data$coop_name_g7[50] # select the coop for the 50th observation
# Which districts are in this coop and how many 7th graders do we have for each?
table(sea_data$sch_g7_lea_id[sea_data$coop_name_g7 == my_coop],
      useNA = "always")
```

```

  01  028  033  044   09 <NA> 
2499 2500 4996 2497 2491    0 
```

```r
# which schools?
table(sea_data$sch_g7_name[sea_data$coop_name_g7 == my_coop],
      useNA = "always")
```

```

       Adams        Adler        Allen        Baker       Bootes       Carmen 
         309          892           35           98          269           91 
     Chelsea     Chestnut      Coleman    Commander  Copper Cove  Cornerstone 
         196          365          235          170          389          654 
      Dalton     Danehill      Dogwood    Gail Hill   Greenfield      Hanover 
         859          150          146          198          146          444 
   Hawthorne     Hillside      Hoffman       Irving     Islander      Kennedy 
         140          285          227          115          305          380 
   Lakeshore        Lever     Majestic       Meadow     Meridian Milton South 
         184          155           60          305          156          903 
      Murphy     Oak Tree       Oriole         Park       Peyton      Prairie 
         195          236          159         1424          174           18 
     Rainbow  Reigh Count        Reyes      Ritchie      Sargent    Sea Glass 
         176          110          392          263          104          137 
    Sterling Stone Street       Tupelo   Valley Way    Van Dusen       Venice 
         302          171          179          253          180          628 
     Wallaby   Wellington  Whitebridge   Winchester   Woodpecker       Yawkey 
         143          110          151          124          126          367 
        <NA> 
           0 
```

### Student Subgroups

What student subgroups are we interested in? Let's start by looking at student
subgroups. Here's whether a student is male.


```r
table(sea_data$male, useNA="always")
```

```

    0     1  <NA> 
59696 62882  1210 
```

Here's a short for loop to look at one-way tabs of a lot of variables at once.


```r
for(i in c("male", "race_ethnicity", "frpl_7", "iep_7", "ell_7",
           "gifted_7")){
  print(i)
  print(table(sea_data[, i], useNA="always"))
}
```

Let's examine the distribution of student subgroups by geography. For this
command, we'll use the same looping syntax from above, which lets you avoid
repetition by applying commands to multiple variables at once. You can type
`?for` into the R console if you want to learn more about how to use loops in R.


```r
# TODO - how do you want to do these crosstabs in dplyr instead of in a table?
# sea_data %>% group_by(coop_name_g7) %>% 
#   count(male, name = "male_count") %>% 
#   mutate(male = ifelse(male == 1, "m", "f")) %>%
#   spread(male, male_count)
# 
# sea_data %>% group_by(coop_name_g7) %>% 
#   count(frpl_7, name = "frpl_count") %>% 
#   mutate(frpl_7 = paste0("frpl_", frpl_7)) %>%
#   spread(frpl_7, frpl_count)




for(var in c("male", "race_ethnicity", "frpl_7", "iep_7", "ell_7",
           "gifted_7")){
  print(var)
  print( # have to call print inside a loop
    round( # round the result
      prop.table( # convert table to percentages
        table(sea_data$coop_name_g7, sea_data[, var],  # build the table
                           useNA = "always"),
        margin = 1), # calculate percentages by column, change to 1 for row
      digits = 3) # round off at 3 digits
    *100 ) # put on percentage instead of proportion scale
}
```

Now, let's look at high school outcomes. We won't examine them all, but you should.
Here's the on-time high school graduation outcome variable we looked at above:


```r
table(sea_data$ontime_grad, useNA = "always")
```

```

    0     1  <NA> 
42701 81087     0 
```

Wait! What if the data includes students who transferred out of state? That
might bias the graduation rate and make it too low, because those 7th graders
might show up as having dropped out.


```r
table(sea_data$transferout, useNA = "always")
```

```

     0      1   <NA> 
111281  12507      0 
```

```r
table(transfer = sea_data$transferout, grad = sea_data$ontime_grad, useNA = "always")
```

```
        grad
transfer     0     1  <NA>
    0    30194 81087     0
    1    12507     0     0
    <NA>     0     0     0
```

This is another case where we may want to consider a business rule. How should
students who transfer out be treated? We don't know whether they graduated
or not. Should they be excluded from the analysis? Coded as not completing?
The decision is yours, but it is important to consider all the possible high
school outcomes when building a model and how the model will treat them.

Let's look at the distribution of another outcome variable, `any_grad`, which
includes late graduation and on-time graduation by both geography and by
subgroup.


```r
round(
  prop.table(
    table(sea_data$coop_name_g7, sea_data$any_grad, useNA="always"),
      margin = 1
    ),
  digits = 2)

for(var in c("male", "race_ethnicity", "frpl_7", "iep_7", "ell_7",
           "gifted_7")){
  print(var)
  print(
    prop.table(
      table(grad = sea_data$any_grad, var = sea_data[, var],
            useNA = "always"),
      margin = 1)
  )
}
```

### Review Existing Indicator

Model comparison is central to developing good predictive models. The data set
for this guide comes with predictions from a fictitious vendor, which as an
analyst your goal is to match or beat in terms of accuracy. This section of
the guide explains how you can review and evaluate the accuracy of a model
using only the predictions it provides.

First, let's check the format of the model predictions:


```r
summary(sea_data$vendor_ews_score)
```

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
0.05032 0.89332 0.95164 0.92004 0.97805 0.99977 
```

There are two things to notice about this vendor's prediction. First, instead of
classifying students, each student receives a predicted probability for
graduating. This is a common approach. Instead of saying a student will graduate
or not graduate, we can say what the probability is that each student will
graduate. A second thing to notice is that we have a prediction for every single 
student in the data - no missing values. We already know some students have 
missing data for some variables. Making predictions for all students is a really 
great feature, but it calls into question what data elements are being used to 
make those predictions and how accurate such predictions may be. 

#### Confusion Matrix

A standard way to interpret the accuracy of predictive models that are classifying 
cases into two categories is called a confusion matrix. This allows us to 
compare a binary prediction (will graduate, will not graduate) to the an 
observed binary outcome (did graduate, did not graduate). To use this technique 
we select a probability threshold, above which observations are classified in 
one category, and below which they fall into the other category. A good threshold 
to try at first is 0.5. We can create a prediction on the fly and build the 
confusion matrix using the code below:


```r
set_thresh <- 0.5

conf_count <- table(observed = sea_data$ontime_grad,
      pred = sea_data$vendor_ews_score > set_thresh)
conf_count
```

```
        pred
observed FALSE  TRUE
       0    31 42670
       1    14 81073
```

The diagonal of the matrix tells us which classifications we made correctly. In 
this case, we predicted and were correct that 31 students would not graduate 
on time. We also predicted correctly 81,073 students who graduated on-time. 
However, we incorrectly predicted that 42,670 students would graduate on-time, 
but they did not -- this is not a very good prediction. We also predicted 
14 students would not graduate, but they did wind up graduating. 

We'll look at several different ways of defining accuracy a little bit further 
below, but here we can see that we do a good job identifying most of the students 
who graduate on-time, but we do that by predicting almost everyone to graduate 
on-time, which is not a very useful analytic. 

So, using a threshold of 0.5 we can see the vendor model predicts almost 
everyone to graduate, which results in getting a lot of correct predictions 
of students who will graduate, but failing to identify very many students 
who are at risk. Let's look at another method for setting the threshold and 
review the resulting confusion matrix. 

A common approach is to set the threshold at the mean of the predicted 
probability. 


```r
set_thresh <- mean(sea_data$vendor_ews_score)

conf_count <- table(observed = sea_data$ontime_grad,
      pred = sea_data$vendor_ews_score > set_thresh)
conf_count
```

```
        pred
observed FALSE  TRUE
       0 20826 21875
       1 20867 60220
```

```r
# Create a proportion table and round for easier interpretation
round( 
  prop.table(conf_count), 
    digits = 3
)
```

```
        pred
observed FALSE  TRUE
       0 0.168 0.177
       1 0.169 0.486
```

Doing this, we see that we identify many fewer students who graduate on-time 
but we in exchange we identify many more students who do not graduate on-time. 
For example, we identify 20,826 students who do not graduate on-time successfully 
as non-graduates. We failed to identify 21,875 additional students who did not 
graduate on-time. We falsely identified 20,867 students as likely to not graduate 
on-time, but who in fact do graduate on time. Finally, we identify 60,220 students 
to graduate on-time who do graduate on-time. 

All we have done is changed the our threshold for action from 0.5 to the mean 
of the predicted probability 
0.92. So, we can already see 
that how to interpret the accuracy of our predictors depends both on what we 
value for accuracy, and how we set our probability cutoff for our predicted 
probability. 

### Explore Possible Predictors

Now that we understand how the vendor's prediction model is working, let's turn to
identifying the predictors available for building an alternative model. Let's
examine the performance and behavioral variables that you can use as predictors.
These are mostly numerical variables, so you should use `summary()`, `hist()`,
and `table()` functions explore them. Here's some syntax for examining 7th grade
math scores. You can replicate and edit it to examine other potential predictors
and their distributions by different subgroups.


```r
summary(sea_data$scale_score_7_math)
```

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
 -42.26   32.73   38.50   39.09   44.71  110.51    1237 
```

```r
hist(sea_data$scale_score_7_math)
```

<img src="../figure/pa_math_score_explore-1.png" style="display: block; margin: auto;" />

A quick way to explore variables by a category is to use the `by()` function. 


```r
# TODO: Replace this with a visualization
by(sea_data$scale_score_7_math, sea_data$coop_name_g7, FUN = mean,
   na.rm = TRUE)
```

```
sea_data$coop_name_g7: Angelea
[1] 40.26076
------------------------------------------------------------ 
sea_data$coop_name_g7: Birch
[1] 37.51467
------------------------------------------------------------ 
sea_data$coop_name_g7: Caldwell
[1] 37.73536
------------------------------------------------------------ 
sea_data$coop_name_g7: Cold Springs
[1] 40.29902
------------------------------------------------------------ 
sea_data$coop_name_g7: Hope
[1] 40.85334
------------------------------------------------------------ 
sea_data$coop_name_g7: Marvel
[1] 38.68512
------------------------------------------------------------ 
sea_data$coop_name_g7: Monarch
[1] 36.84879
------------------------------------------------------------ 
sea_data$coop_name_g7: Weston
[1] 38.87301
------------------------------------------------------------ 
sea_data$coop_name_g7: Wintergreen
[1] 40.84764
```

```r
by(sea_data$scale_score_7_math, sea_data$frpl_7, FUN = mean,
   na.rm = TRUE)
```

```
sea_data$frpl_7: 0
[1] 41.40213
------------------------------------------------------------ 
sea_data$frpl_7: 1
[1] 35.0435
------------------------------------------------------------ 
sea_data$frpl_7: 2
[1] 34.89737
------------------------------------------------------------ 
sea_data$frpl_7: 9
[1] 39.19554
```

After exploring the predictors themselves, a good next step is to explore their 
relationship to the outcome. Let's start with the relationship between on-time 
graduation and 7th grade math scores and the percent of days absent in 7th grade. 
A good way to do this is to compare the difference in the means between graduates 
and non-graduates. We can standardize this difference using the standard deviation, 
a measure that is known as an effect size. Comparing differences in terms of 
standard deviations gives us a good sense of how different graduates and 
non-graduates are from one another on key measures. To simplify this process, 
we can use the `effect_size_diff()` function from the `functions.R` script. 


```r
effect_size_diff(sea_data$scale_score_7_math, sea_data$ontime_grad, 
                 na.rm = TRUE)
```

```
[1] 0.2901296
```

```r
effect_size_diff(sea_data$pct_days_absent_7, sea_data$ontime_grad, 
                 na.rm = TRUE)
```

```
[1] -0.003474914
```

We can see that math scores differentiate graduates and non-graduates in these 
data, but the two groups are nearly identical in terms of their attendance rates. 

It can be helpful at this point to visually inspect these differences as well. 
But you can't make a meaningful scatterplot when the independent, or y value, is
a binary outcome variable (try it!). Let's look at a technique to identify
the relationship between a continuous variable and a binary outcome.

The idea behind this code is to show the mean of the outcome variable for each
value of the predictor, or for categories of the predictor variable if it has
too many values. First, define categories (in this case, round to the nearest
percentage) of the percent absent variable, and then truncate the variable so that
low-frequency values are grouped together.


```r
sea_data$pct_absent_cat <- round(sea_data$pct_days_absent_7, digits = 0)
table(sea_data$pct_absent_cat)
```

```

    0     1     2     3     4     5     6     7     8     9    10    11    12 
62131  7311  7082  6876  6201  3000  5596  4867  4173  3533  1473  2625  2043 
   13    14    15    16    17    18    19    20    21    22    23    24    25 
 1574  1199   496   748   533   372   244    77   124    66    56    30     7 
   26    27    28    30    32    80   140 
   11     4     4     2     1   602   120 
```

```r
sea_data$pct_absent_cat[sea_data$pct_absenct_cat >= 30] <- 30
```

Next, define a variable which is the average on-time graduation rate for each
absence category, and then make a scatter plot of average graduation rates by
absence percent.


```r
# TODO - Update this plot sequence to ggplot2
sea_data <- sea_data %>%
  group_by(pct_absent_cat) %>% # perform the operation for each value
  mutate(abs_ontime_grad = mean(ontime_grad, na.rm = TRUE)) %>% # add a new variable
  as.data.frame() 

plot(sea_data$pct_absent_cat, sea_data$abs_ontime_grad)
```

<img src="../figure/pa_plot_absence_categories-1.png" style="display: block; margin: auto;" />

You can do the same thing for 7th grade test scores. First look at the math
test score and notice that some scores appear to be outliers.


```r
hist(sea_data$scale_score_7_math)
```

<img src="../figure/pa_match_score_hists-1.png" style="display: block; margin: auto;" />

```r
sea_data$scale_score_7_math[sea_data$scale_score_7_math < 0] <- NA
hist(sea_data$scale_score_7_math)
```

<img src="../figure/pa_match_score_hists-2.png" style="display: block; margin: auto;" />

You can do the same plot as above now by modifying the `group_by()`
command.


```r
sea_data <- sea_data %>%
  mutate(math_7_cut = ntile(scale_score_7_math, n = 100)) %>%
  group_by(math_7_cut) %>% # perform the operation for each value
  mutate(math_7_ontime_grad = mean(ontime_grad, na.rm=TRUE)) %>% # add a new variable
  as.data.frame()
  

plot(sea_data$math_7_cut, sea_data$math_7_ontime_grad)
```

<img src="../figure/pa_math_grad_comparison-1.png" style="display: block; margin: auto;" />

This is a neat trick you can use to communicate your model predictions as well 
which we will use again below. 

### Missingness

Finally, here's some sample code you can use to look at missingness patterns in
the data. Note we use the `is.na()` function to test whether a value is missing.


```r
for(var in c("coop_name_g7", "male", "race_ethnicity")){
  print(var)
  print(
  prop.table(table(sea_data[, var],
                         "missing_math" = is.na(sea_data$pct_days_absent_7)), 1)
  )
}
```

```
[1] "coop_name_g7"
              missing_math
                     FALSE        TRUE
  Angelea      0.995930048 0.004069952
  Birch        0.995908956 0.004091044
  Caldwell     0.995225769 0.004774231
  Cold Springs 0.994574930 0.005425070
  Hope         0.995328213 0.004671787
  Marvel       0.994860842 0.005139158
  Monarch      0.994398221 0.005601779
  Weston       0.994274998 0.005725002
  Wintergreen  0.995035947 0.004964053
[1] "male"
   missing_math
          FALSE        TRUE
  0 0.994773519 0.005226481
  1 0.995372285 0.004627715
[1] "race_ethnicity"
           missing_math
                  FALSE        TRUE
  Americ... 0.994354839 0.005645161
  Asian     0.994390244 0.005609756
  Black ... 0.995142471 0.004857529
  Demogr... 0.995185185 0.004814815
  Hispan... 0.996196404 0.003803596
  Native... 0.997729852 0.002270148
  White     0.994804748 0.005195252
```

Handling missing values is another case where business rules will come into play.

Did you see any outlier or impossible values while you were exploring the data?
If so, you might want to truncate them or change them to missing. Here's how you
can replace a numeric variable with a missing value if it is larger than a
certain number (in this case, 100 percent).


```r
hist(sea_data$pct_days_absent_7)
```

<img src="../figure/pa_abs_trunc-1.png" style="display: block; margin: auto;" />

```r
sea_data$pct_days_absent_7[sea_data$pct_days_absent_7 > 100] <- NA
hist(sea_data$pct_days_absent_7)
```

<img src="../figure/pa_abs_trunc-2.png" style="display: block; margin: auto;" />

Trimming the data in this way is another example of a business rule. You
may wish to trim the absences even further in this data. You may also wish to
assign a different value other than missing for unusual values - such as the
mean or median value.


## Model

Now that we've explored the data and looked at some relationships between
potential  predictors and our outcome of interest, we're ready to fit a logistic
regression.  In R, we use the `glm()` function to fit a logistic regression.
When you run a logistic regression, R calculates the parameters of an equation
that fits the relationship between the predictor variables and the outcome. A
regression model won't be able to explain all of the variation in an outcome
variable--any variation that is left over is treated as unexplained noise in the
data, or error, even if there are additional variables not in the model which
could explain more of the variation.

### Baseline Model

Once you've run a logistic regression, you can have R generate a variable with new,
predicted outcomes for each observation in your data with the `predict()` function.
The predictions are calculated using the model equation and ignore the
unexplained noise in the data. For logistic regressions, the predicted outcomes
take the form of a probability ranging 0 and 1. To start with, let's do a
regression of on-time graduation on seventh grade math scores.


```r
math_model <- glm(ontime_grad ~ scale_score_7_math, data = sea_data,
                  family = "binomial") # family tells R we want to fit a logistic
```

The default summary output for logistic regression in R is not very helpful for
predictive modeling purposes.


```r
summary(math_model)
```

```

Call:
glm(formula = ontime_grad ~ scale_score_7_math, family = "binomial", 
    data = sea_data)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.3833  -1.3617   0.8346   0.9396   1.4221  

Coefficients:
                     Estimate Std. Error z value Pr(>|z|)    
(Intercept)        -0.5593845  0.0252679  -22.14   <2e-16 ***
scale_score_7_math  0.0310615  0.0006426   48.34   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 157769  on 122433  degrees of freedom
Residual deviance: 155302  on 122432  degrees of freedom
  (1354 observations deleted due to missingness)
AIC: 155306

Number of Fisher Scoring iterations: 4
```

Even before you use the predict command, you can use the logistic output to
learn something about the relationship between the predictor and the outcome
variable. The Pseudo $R^{2}$ (read: pseudo R-squared) is a proxy for the share
of variation in the outcome variable that is explained by the predictor.
Statisticians don't like it when you take the pseudo $R^{2}$ too seriously, but
it can be useful in predictive exercises to quickly get a sense of the
explanatory power of variables in a logistic model. You can use the function
`logit_rsquared()` in the `R/functions.R` file included with this guide to
calculate this for your model.


```r
logit_rsquared(math_model)
```

```
[1] 0.01563745
```

### Extend the Model

We want to increase the pseudo $R^{2}$ using the data we have, so we should
think about what additional information we have available that could inform
our prediction of student graduation.

One place to start is to consider that the relationship between math scores and
graduation may not be linear. We can evaluate this by looking at whether or not
adding polynomial terms increase the pseudo $R^{2}$? You can use the formula
interface in R to add functional transformations of predictors without generating
new variables and find out.


```r
math_model2 <- glm(ontime_grad ~ scale_score_7_math +
                     I(scale_score_7_math^2) + I(scale_score_7_math^3),
                   data = sea_data,
                  family = "binomial") # family tells R we want to fit a logistic
logit_rsquared(math_model2)
```

```
[1] 0.01758691
```

The model did not improve very much.  Any time you add predictors to a model,
the $R^{2}$ will increase, even if the variables are fairly meaningless, so it's
best to focus on including predictors that add meaningful explanatory power.

Now take a look at the $R^{2}$ for the absence variable.


```r
absence_model <- glm(ontime_grad ~ pct_days_absent_7, data = sea_data,
                  family = "binomial")
summary(absence_model)
```

```

Call:
glm(formula = ontime_grad ~ pct_days_absent_7, family = "binomial", 
    data = sea_data)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.4602  -1.4583   0.9188   0.9196   0.9473  

Coefficients:
                    Estimate Std. Error z value Pr(>|z|)    
(Intercept)        0.6439866  0.0066747  96.482   <2e-16 ***
pct_days_absent_7 -0.0009416  0.0008659  -1.087    0.277    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 158586  on 123060  degrees of freedom
Residual deviance: 158585  on 123059  degrees of freedom
  (727 observations deleted due to missingness)
AIC: 158589

Number of Fisher Scoring iterations: 4
```

```r
logit_rsquared(absence_model)
```

```
[1] 7.424679e-06
```

Let's combine our two predictors and test their combined power.


```r
combined_model <- glm(ontime_grad ~ pct_days_absent_7 + scale_score_7_math,
                      data = sea_data, family = "binomial")
summary(combined_model)
```

```

Call:
glm(formula = ontime_grad ~ pct_days_absent_7 + scale_score_7_math, 
    family = "binomial", data = sea_data)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.3868  -1.3614   0.8344   0.9397   1.4224  

Coefficients:
                     Estimate Std. Error z value Pr(>|z|)    
(Intercept)        -0.5601099  0.0255152 -21.952   <2e-16 ***
pct_days_absent_7  -0.0008865  0.0008804  -1.007    0.314    
scale_score_7_math  0.0311488  0.0006446  48.325   <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 156860  on 121716  degrees of freedom
Residual deviance: 154393  on 121714  degrees of freedom
  (2071 observations deleted due to missingness)
AIC: 154399

Number of Fisher Scoring iterations: 4
```

```r
logit_rsquared(combined_model)
```

```
[1] 0.01572911
```

Using this combined model, let's use the predict command to make our first
predictions.


```r
sea_data$grad_pred <- predict(combined_model, newdata = sea_data,
                   type = "response") # this tells R to give us a probability
```

This generates a new variable with the probability of on-time high school
graduation, according to the model. But if you look at the number of
observations with predictions, you'll see that it is smaller than the total
number of students. This is because R doesn't use observations that have
missing data for any of the variables in the model.


```r
table(is.na(sea_data$grad_pred))
```

```

 FALSE   TRUE 
121717   2071 
```

Let's convert this probability to a 0/1 indicator for whether or not a student
is likely to graduate on-time. A good rule of thumb when starting out is to set
the probability cutoff at the mean of the outcome variable. In this example,
we can store this value as a variable:


```r
basic_thresh <- mean(sea_data$ontime_grad)
basic_thresh
```

```
[1] 0.6550473
```

If the probability in the model is equal to or
greater than this threshold, we'll say the student is likely to graduate.


```r
sea_data$grad_indicator <- ifelse(sea_data$grad_pred > basic_thresh, 1, 0)
table(sea_data$grad_indicator, useNA = "always")
```

```

    0     1  <NA> 
61793 59924  2071 
```

You can also plot the relationship between the probability and the outcome.
Ideally, you should see the proportion of graduates steadily increase for each
percentile of the probabilities. What does this relationship tell you?


```r
sea_data <- sea_data %>%
  mutate(grad_pred_cut = ntile(grad_pred, n = 100)) %>%
  group_by(grad_pred_cut) %>% # perform the operation for each value
  mutate(grad_pred_cut_grad = mean(ontime_grad, na.rm=TRUE)) # add a new variable

plot(sea_data$grad_pred_cut, sea_data$grad_pred_cut_grad)
```

<img src="../figure/pa_unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

### Measure Model Accuracy

Lets evaluate the accuracy of the model by comparing the predictions to the
actual graduation outcomes for the students for whom we have predictions. This
type of cross tab is called a "confusion matrix." The observations in the upper
right corner, where the indicator and the actual outcome are both 0, are true
negatives. The observations in the lower right corner, where the indicator and
the outcome are both 1, are true positives. The upper right corner contains
false positives, and the lower left corner contains false negatives. Overall, if
you add up the cell percentages for true positives and true negatives, the model
got 56.1 percent of the predictions right.


```r
prop.table(
  table(
    grad = sea_data$ontime_grad, 
    pred = sea_data$grad_indicator)) %>% # shorthand way to round
  round(3)
```

```
    pred
grad     0     1
   0 0.207 0.138
   1 0.301 0.354
```

However, most of the wrong predictions are false negatives -- these are
students who have been flagged as dropout risks even though they
did graduate on-time. If you want your indicator system to be have fewer false
negatives, you can change the probability cutoff. This cutoff has a lower share
of false positives and a higher share of false negatives, with a somewhat lower
share of correct predictions.


```r
new_thresh <- basic_thresh - 0.05
prop.table(table(Observed = sea_data$ontime_grad,
                 Predicted = sea_data$grad_pred > new_thresh)) %>%
  round(3)
```

```
        Predicted
Observed FALSE  TRUE
       0 0.097 0.248
       1 0.118 0.537
```

Note that this table only includes the complete cases. To look at missing values
as well:


```r
prop.table(table(Observed = sea_data$ontime_grad,
                 Predicted = sea_data$grad_pred > new_thresh,
                 useNA = "always")) %>% round(3)
```

```
        Predicted
Observed FALSE  TRUE  <NA>
    0    0.095 0.244 0.006
    1    0.116 0.528 0.011
    <NA> 0.000 0.000 0.000
```

This table tells us that of our observations that have missing data for their 
predictors, more of them graduate than don't. 

## Missing Data

Another key business rule is how we will handle students with missing data. A
predictive analytics system is more useful if it makes an actionable prediction
for every student. It is good to check, if it is available, the graduation rates
for students with missing data:


```r
table(Grad = sea_data$ontime_grad,
      miss_math = is.na(sea_data$scale_score_7_math)) %>% 
  prop.table(2) %>%  # get proportions by columns
  round(3) # round
```

```
    miss_math
Grad FALSE  TRUE
   0 0.345 0.340
   1 0.655 0.660
```

```r
table(Grad = sea_data$ontime_grad,
      miss_abs = is.na(sea_data$pct_days_absent_7)) %>% 
  prop.table(2) %>% # get proportions by columns
  round(3) # round
```

```
    miss_abs
Grad FALSE  TRUE
   0 0.345 0.326
   1 0.655 0.674
```

Students with missing data graduate at a slightly higher rate than students with
full data. There are a number of options at this point. One is to run a model
with fewer variables for only those students, and then use that model to fill in
the missing indicators.


```r
absence_model <- glm(ontime_grad ~ pct_days_absent_7,
                     data = sea_data[is.na(sea_data$scale_score_7_math),],
                     family = "binomial")
```


```r
sea_data$grad_pred_2 <- predict(absence_model, newdata = sea_data,
                                  type = "response")
summary(absence_model)
```

```

Call:
glm(formula = ontime_grad ~ pct_days_absent_7, family = "binomial", 
    data = sea_data[is.na(sea_data$scale_score_7_math), ])

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.4793  -1.4579   0.9030   0.9101   1.1779  

Coefficients:
                   Estimate Std. Error z value Pr(>|z|)    
(Intercept)        0.686533   0.064396  10.661   <2e-16 ***
pct_days_absent_7 -0.008597   0.008193  -1.049    0.294    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 1725.8  on 1343  degrees of freedom
Residual deviance: 1724.7  on 1342  degrees of freedom
  (10 observations deleted due to missingness)
AIC: 1728.7

Number of Fisher Scoring iterations: 4
```


```r
table(sea_data$grad_indicator, useNA="always")
```

```

    0     1  <NA> 
61793 59924  2071 
```

```r
sea_data$grad_indicator[is.na(sea_data$grad_pred) &  
                            sea_data$grad_pred_2 < new_thresh] <- 0
sea_data$grad_indicator[is.na(sea_data$grad_pred) &  
                            sea_data$grad_pred_2 >= new_thresh] <- 1
table(sea_data$grad_indicator, useNA="always")
```

```

    0     1  <NA> 
61799 61262   727 
```

We now have predictions for all but a very small share of students, and those
students are split between graduates and non-graduates. We have to apply a rule
or a model to make predictions for them--we can't use information from the
future, except to develop the prediction system. We'll arbitrarily decide to
flag them as potential non-graduates, since students with lots of missing data
might merit some extra attention.


```r
table(sea_data$grad_indicator, sea_data$ontime_grad, useNA = "always")
```

```
      
           0     1  <NA>
  0    25180 36619     0
  1    17284 43978     0
  <NA>   237   490     0
```

```r
sea_data$grad_indicator[is.na(sea_data$grad_indicator)] <- 0
```

## Evaluate Fit

Now we have a complete set of predictions from our simple models. How well does
the prediction system work? Can we do better?


```r
table(Observed = sea_data$ontime_grad, Predicted = sea_data$grad_indicator) %>%
  prop.table %>% round(3)
```

```
        Predicted
Observed     0     1
       0 0.205 0.140
       1 0.300 0.355
```

A confusion matrix is one way to evaluate the success of a model and evaluate
trade offs as you are developing prediction systems. In cases like this,
where we have an uneven proportion of cases in each class (e.g. we have many more graduates than non-graduates),
it can be helpful to look at a metric like the AUC, which stands for "area under the
curve." The curve here refers to the receiver-operator characteristic, a
[statistic developed in the field of signal detection theory during WWII](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#History).  
This metric allows you to explicitly weigh the trade-off in identifying a higher
proportion of an event (e.g. non-graduations) against the additional
false-positives (declaring students who will graduate as non-graduates) it will
cost to achieve that.

To understand this, let's first, look at row percentages instead of cell
percentages in the confusion matrix.


```r
table(Observed = sea_data$ontime_grad, Predicted = sea_data$grad_indicator) %>%
  prop.table(margin = 1) %>% round(3)
```

```
        Predicted
Observed     0     1
       0 0.595 0.405
       1 0.458 0.542
```

Next, use the `roc()` function to plot the true positive rate (sensitivity in
the graph) against the false positive rate (1-specificity in the graph).


```r
roc(sea_data$ontime_grad, sea_data$grad_indicator) %>% plot
```

<img src="../figure/pa_calculateROC-1.png" style="display: block; margin: auto;" />

You can also calculate ROC on the continuous predictor as well, to help you
determine the threshold:



```r
roc(sea_data$ontime_grad, sea_data$grad_pred) %>% plot
```

<img src="../figure/pa_calculateROC2-1.png" style="display: block; margin: auto;" />

You can also calculate the numeric summaries instead of just the graphs. To
do this let's use the `caret` package:


```r
library(caret)
# We must wrap these each in calls to factor because of how this function expects
# the data to be formatted
caret::confusionMatrix(factor(sea_data$grad_indicator),
                       factor(sea_data$ontime_grad), positive = "1")
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 25417 37109
         1 17284 43978
                                          
               Accuracy : 0.5606          
                 95% CI : (0.5578, 0.5634)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.124           
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.5424          
            Specificity : 0.5952          
         Pos Pred Value : 0.7179          
         Neg Pred Value : 0.4065          
             Prevalence : 0.6550          
         Detection Rate : 0.3553          
   Detection Prevalence : 0.4949          
      Balanced Accuracy : 0.5688          
                                          
       'Positive' Class : 1               
                                          
```

A couple of last thoughts and notes. First, note that so far we haven't done any
out-of-sample testing. We know from the pre-reading that we should never trust
our model fit measures on data the model was fit to -- statistical models are
overly confident. To combat this, you should subdivide your data set. There are
many strategies you can choose from depending on how much data you have and the
nature of your problem - for the EWS case, we can use the first two cohorts to
build our models and the latter two cohorts to evaluate that fit.

Here is some code you can use to do that:


```r
# In R we can define an index of rows so we do not have to copy our data
train_idx <- row.names(sea_data[sea_data$year %in% c(2003, 2004, 2005),])
test_idx <- !row.names(sea_data) %in% train_idx

fit_model <- glm(ontime_grad ~ scale_score_7_math + frpl_7,
                 data = sea_data[train_idx, ], family = "binomial")

sea_data$grad_pred_3 <- predict(fit_model, newdata = sea_data, type = "response")
summary(sea_data$grad_pred_3)
```

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
 0.2872  0.6173  0.6603  0.6594  0.7020  0.9427    1354 
```

```r
# Check the test index only
summary(sea_data$grad_pred_3[test_idx])
```

```
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
 0.2872  0.6126  0.6560  0.6559  0.6992  0.9427     652 
```

```r
# calculate matrix of the test_index

table(Observed = sea_data$ontime_grad[test_idx],
      Predicted = sea_data$grad_pred_3[test_idx] > new_thresh) %>%
  prop.table() %>% round(4)
```

```
        Predicted
Observed  FALSE   TRUE
       0 0.1017 0.2520
       1 0.1134 0.5328
```

Second, should we use subgroup membership variables (such as demographics or
school of enrollment?) to make predictions, if they improve the accuracy of
predictions? This is more a policy question than a technical question, and you
should consider it when you are developing your models. You'll also want to
check to see how accurate your model is for different subgroups.


## Beyond Logistic Regression

The above code will be more than enough to get you started. But, if you want
to reach further into predictive analytics, the next section provides some bonus
syntax and advice for building more complex statistical models.

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
plan(multiprocess(workers = 8)) # define the number of cpus to use
registerDoFuture() # register them with R

# Caret really really really likes if you do binary classification that you
# code the variables as factors with alphabetical labels. In this case, we
# recode 0/1 to be nongrad, grad.

yvar <- sea_data[train_idx, "ontime_grad"] # save only training observations
yvar <- ifelse(yvar == 1, "grad", "nongrad")
yvar <- factor(yvar)

# On a standard desktop/laptop it can be necessary to decrease the sample size
# to train in a reasonable amount of time. For the prototype and getting feedback
# it's a good idea to stick with a reasonable sample size of under 30,000 rows.
# Let's do that here:

train_idx_small <- sample(1:nrow(preds), 2e4)

# Caret has a number of complex options, you can read about under ?trainControl
# Here we set some sensible defaults
example_control <- trainControl(
  method = "cv", # we cross-validate our model to avoid overfitting
  classProbs = TRUE,  # we want to be able to predict probabilities, not just binary outcomes
  returnData = TRUE, # we want to store the model data to allow for postestimation
  summaryFunction = prSummary, # we want to use the prSummary for better two-class accuracy measures
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
  x = preds[train_idx_small,], # specify the matrix of predictors, again subset
  method = "rpart",  # choose the algorithm - here a regression tree
  tuneLength = 24, # choose how many values of tuning parameters to test
  trControl = example_control, # set up the conditions for the test (defined above)
   metric = "AUC" # select the metric to choose the best model based on
  )


set.seed(2532)
# Repeat above but just change the `method` argument to nb for naiveBayes
nb_model <- train(y = yvar[train_idx_small],
             x = preds[train_idx_small,],
             method = "nb", tuneLength = 24,
             trControl = example_control,
             metric = "AUC")
```



```r
# Construct a list of model performance comparing the two models directly
resamps <- resamples(list(RPART = rpart_model,
              NB = nb_model))
# Summarize it
summary(resamps)
```

```

Call:
summary.resamples(object = resamps)

Models: RPART, NB 
Number of resamples: 10 

AUC 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.6640029 0.7197764 0.7382367 0.7351345 0.7634645 0.7884460    0
NB    0.7790811 0.7925580 0.7961417 0.7999324 0.8112355 0.8205061    0

F 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.7761836 0.7803174 0.7850529 0.7846857 0.7881853 0.7953008    0
NB    0.7873825 0.7881176 0.7882371 0.7884234 0.7890145 0.7893139    0

Precision 
           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.7232258 0.7279274 0.7313003 0.7327478 0.7359999 0.7457162    0
NB    0.6499750 0.6503252 0.6508015 0.6510357 0.6515475 0.6525892    0

Recall 
           Min.  1st Qu.    Median      Mean   3rd Qu.      Max. NA's
RPART 0.8316679 0.837500 0.8434615 0.8446504 0.8516227 0.8623077    0
NB    0.9976941 0.998654 0.9996157 0.9993080 1.0000000 1.0000000    0
```

```r
# plot it
# see ?resamples for more options
dotplot(resamps, metric = "AUC")
```

<img src="../figure/pa_caretmodeleval-1.png" style="display: block; margin: auto;" />

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
   grad    68529   22499
   nongrad 12558   20202
                                          
               Accuracy : 0.7168          
                 95% CI : (0.7143, 0.7193)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3368          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8451          
            Specificity : 0.4731          
         Pos Pred Value : 0.7528          
         Neg Pred Value : 0.6167          
             Prevalence : 0.6550          
         Detection Rate : 0.5536          
   Detection Prevalence : 0.7354          
      Balanced Accuracy : 0.6591          
                                          
       'Positive' Class : grad            
                                          
```

```r
# Note that making predictions from the nb classifier can take a long time
# consider alternative models or making fewer predictions if this is a bottleneck
confusionMatrix(reference = test_y, data = predict(nb_model, test_data))
```

```
Confusion Matrix and Statistics

          Reference
Prediction  grad nongrad
   grad    81016   42420
   nongrad    71     281
                                          
               Accuracy : 0.6567          
                 95% CI : (0.6541, 0.6594)
    No Information Rate : 0.655           
    P-Value [Acc > NIR] : 0.1051          
                                          
                  Kappa : 0.0075          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.999124        
            Specificity : 0.006581        
         Pos Pred Value : 0.656340        
         Neg Pred Value : 0.798295        
             Prevalence : 0.655047        
         Detection Rate : 0.654474        
   Detection Prevalence : 0.997156        
      Balanced Accuracy : 0.502853        
                                          
       'Positive' Class : grad            
                                          
```

We can also make ROC plots of our test-set, out of sample, prediction accuracy.


```r
# ROC plots
yhat <- predict(rpart_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_outofsampleROC-1.png" style="display: block; margin: auto;" />

```r
# NB ROC plot
# Note that making predictions from the nb classifier can take a long time
# consider alternative models or making fewer predictions if this is a bottleneck
yhat <- predict(nb_model, test_data, type = "prob")
yhat <- yhat$grad
roc(response = test_y, predictor = yhat <- yhat) %>% plot
```

<img src="../figure/pa_outofsampleROC-2.png" style="display: block; margin: auto;" />


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


preds_nb <- predict(nb_model, newdata = pred_data, type = "prob")$grad
preds_rpart <- predict(rpart_model, newdata = pred_data, type = "prob")$grad

current_data <- bind_cols(current_data,
                          data.frame(preds_nb),
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
 $ preds_nb            : num  0.899 0.847 0.945 0.982 0.921 ...
 $ preds_rpart         : num  0.622 0.373 0.729 0.676 0.78 ...
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

<img src="../figure/pa_unnamed-chunk-17-1.png" style="display: block; margin: auto;" />

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

<img src="../figure/pa_unnamed-chunk-17-2.png" style="display: block; margin: auto;" />

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
                   metric = "AUC",
                   trControl = trainControl(summaryFunction = prSummary,
                                            classProbs = TRUE,
                                            method = "cv"))

# Get variable importance, not available for all methods
caret::varImp(caret_mod)
```

```
rpart variable importance

                           Overall
sch_g7_gifted_per         100.0000
sch_g7_lep_per             80.3568
race_ethnicityWhite        75.1759
scale_score_7_math         54.6073
race_ethnicityHispan...    44.9445
frpl_71                    41.8397
race_ethnicityAsian        33.2768
iep_7                       8.0599
race_ethnicityNative...     0.6088
frpl_72                     0.3048
male1                       0.0000
pct_days_absent_7           0.0000
race_ethnicityDemogr...     0.0000
frpl_79                     0.0000
ell_7                       0.0000
`race_ethnicityBlack ...`   0.0000
```

```r
# Plot variable importance
plot(caret::varImp(caret_mod))
```

<img src="../figure/pa_unnamed-chunk-18-1.png" style="display: block; margin: auto;" />

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
   Grad    72587   30396
   Nongrad  5595   10746
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

<img src="../figure/pa_tidyrocplots-1.png" style="display: block; margin: auto;" />

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

<img src="../figure/pa_plotconfusionmatrix-1.png" style="display: block; margin: auto;" />

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

<img src="../figure/pa_probvsoutcomeplot-1.png" style="display: block; margin: auto;" />

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

<img src="../figure/pa_unnamed-chunk-19-1.png" style="display: block; margin: auto;" />

```r
# Use a custom probability threshold
```

### About the Analyses

**Describe purpose and methods of analyses here**

### Giving Feedback on this Guide

This guide is an open-source document hosted on Github and generated using R
Markdown. We welcome feedback, corrections, additions, and updates. Please visit
the OpenSDP equity metrics repository to read our contributor guidelines.
