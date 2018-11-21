###############################################################################
# Generate Synthetic Data In SDP PA Workshop Format
# Date: 08/21/2018
# Author: Jared E. Knowles
# Prepared for OpenSDP
################################################################################

## Identify the data structure needed 
# train <- read.csv("C:/Users/jknow/Documents/GitHub/predicting_dropouts/data/training_2009.csv", stringsAsFactors = FALSE)
## File - student grade 7 + grad
## sid
## first_hs_code
## first_dist_code
## first_hs_name
## first_dist_name
## chrt_ninth
## male
## race_ethnicity
## frpl
## iep 
## ell
## gifted
## ever_alternative
## scale_score_6_math
## scale_score_6_read
## pct_days_absent_6
## gpa_6
## transferout
## dropout
## still_enrolled
## ontime_grad
## still_enrolled
## chrt_grad
## hs_diploma

## Install the synthesizer to generate the data
## This step is optional if you already have it installed
## Current package depends on this fork of the simglm package
## Uncomment to use
#devtools::install_github("jknowles/simglm")
## Install the OpenSDP data synthesizer
## Uncomment to use
# devtools::install_github("OpenSDP/OpenSDPsynthR")

## 

library(OpenSDPsynthR)
set.seed(11213) # set the seed
library(magrittr)
library(stringr)
# The synthesizer needs some input paramaters
# As it is the defaults are not sufficient to give realistic assessment data
# These change those defaults to make the scores less deterministic


for (i in 1:45) {
  # Each iteration is a school district
  # Set up iteration specific parameters to induce more variation
  # new_seed <- 11213 + sample(1:5, 1) * i
  new_seed <- 11213 + 2 * i
  # Get assessment adjustment controls
  assess_adj <- sim_control()$assessment_adjustment
  assess_adj$school_list <- NULL
  assess_adj$perturb_school <- function(x, schid, sd, schl_par = school_list){
    val_mean <- schl_par[[which(schid == names(schl_par))]]
    val_sd <- abs(val_mean) / 3
    val_mean <- val_mean * 2
    y <- x + rnorm(1, val_mean, val_sd)
    return(y)
  }
  # Make scores spread out more
  assess_adj$perturb_base <- function(x, sd) 
  {
    mean_shift <- rnorm(1, sd = 3)
    y <- x + rnorm(1, mean_shift, sd * 0.8)
    return(y)
  }
  # Add racial inequality to assessment scores
  assess_adj$race_list <- list("White" = sample(c(0.84, 0.95, 1.2, 1.25, 0.675, 1.45), 1), 
                               "Black or African American" = sample(c(0, -0.5, -0.3, -0.125, -0.2), 1),
                               "Asian" = sample(c(0.4, 0.25, 0.1, 0.125, 0.05, -0.2, -0.15), 1), 
                               "Hispanic or Latino Ethnicity" = sample(c(0.05, 0, -0.4, -0.1, -0.125, -0.2), 1), 
                               "Demographic Race Two or More Races" = sample(c(-2:2 / 40), 1), 
                               "American Indian or Alaska Native" = sample(c(-8:8 / 40), 1), 
                               "Native Hawaiian or Other Pacific Islander" = sample(c(-8:8 / 40), 1) 
                               )
  
  # Get district specific gender gaps
  assess_adj$gender_list <- list("Male" = sample(c(0.0, -0.05, 0.25, 0.1, 0.275, 0.375), 1), 
                                 "Female" = sample(c(0.0, -0.05, 0.05, -0.1, 0.125, -0.075), 1))
  # Get district specific FRL gaps
  assess_adj$frl_list <- list("0" = sample(c(0.8, 0.6, 0.7, 0.45, 0.51, 0.6125, 0.975), 1), 
                              "1" = sample(c(0.05, 0, -0.25, -0.3, -0.125, -0.6, -0.5), 1))
  # Get assessment simulation  defaults
  assess_sim_par <- OpenSDPsynthR::sim_control()$assess_sim_par
  # Increase score variance randomly by district
  assess_sim_par$error_var <- sample(12:18, 1)
  # Increase coefficient effects randomly by district
  # assess_sim_par$fixed_param <- assess_sim_par$fixed_param * sample(9:11, 1)
  # Downgrade IEP difference randomly by district
  assess_sim_par$fixed_param[2] <- 6
  # assess_sim_par$fixed_param[4] <- sample(c(-0.75, -0.5, -.66, -1), 1)
  # Downgrade LEP difference
  assess_sim_par$fixed_param[3] <- 2
  assess_sim_par$fixed_param[4] <- -4
  assess_sim_par$fixed_param[5] <- -2
  assess_sim_par$fixed_param[6] <- -2
  assess_sim_par$fixed_param[7] <- 1.5
  assess_sim_par$lvl1_err_params$mean <- 1
  assess_sim_par$lvl1_err_params$sd <- sample(8:12, 1)
  # Set group level variances
  assess_sim_par$random_param$random_var <- c(sample(c(0.4, 0.3, 0.5, 0.375, 0.425), 1), 
                                              sample(c(0.125, 0.5, 0.66, 0.75), 1))
  # Set the school-grade size ranges
  assess_sim_par$unbalanceRange <- c(75, 550)
  
  grad_adj <- sim_control()$grad_adjustment
  grad_sim_par <- sim_control()$grad_sim_par
  
  # Covariance parameters are an imprecise way of modifying the graduation rate
  grad_sim_par$cov_param$opts[[2]]$mean <- sample(-2:16/20,1)
  grad_sim_par$cov_param$opts[[2]]$sd <- sample(9:11/10, 1)
  grad_sim_par$cov_param$opts[[1]]$mean <- sample(30:42/-15, 1)
  grad_sim_par$cov_param$opts[[1]]$sd <- sample(6:12/10, 1)
  grad_sim_par$random_var <- sample(180:380/60, 1)
  
  # Adjust up the impact of academic variables on outcomes
  grad_sim_par$fixed_param[[1]] <- sample(10:30/2.75, 1) # intercept
  grad_sim_par$fixed_param[[2]] <- sample(1:8/0.6, 1) # math
  grad_sim_par$fixed_param[[3]] <- sample(2:14/8, 1) # gpa
  # grad_sim_par$fixed_param[[4]] <- sample(2:14/8, 1) # gifted
  grad_sim_par$fixed_param[[5]] <- sample(10:20/-8, 1) # iep
  grad_sim_par$fixed_param[[6]] <- sample(6:16/-12, 1) # frpl
  grad_sim_par$fixed_param[[7]] <- sample(6:14/-14, 1) # ell
  grad_sim_par$fixed_param[[8]] <- sample(2:14/-25, 1) # male
  
  
  grad_adj$frl_list <- list("0" = sample(c(0, 0.25, 0.1, 0.125, 0.175), 1), 
                              "1" = sample(c(0.05, 0, -0.5, -0.1, -0.125, -0.3), 1))
  
  grad_adj$race_list <- list("White" = sample(c(0.4, 0.35, 0.1, 0.225, 0.175, 0.215), 1), 
                            "Black or African American" = sample(c(0, -0.35, -0.3, -0.125, -0.2), 1),
                            "Asian" = sample(c(0.4, 0.25, 0.1, 0.125, 0.05), 1), 
                            "Hispanic or Latino Ethnicity" = sample(c(0.05, 0, -0.1, -0.125, -0.2), 1), 
                            "Demographic Race Two or More Races" = sample(c(-2:2 / 40), 1), 
                            "American Indian or Alaska Native" = sample(c(-8:8 / 40), 1), 
                            "Native Hawaiian or Other Pacific Islander" = sample(c(-8:8 / 40), 1) 
                            )
  grad_adj$school_list <- NULL
  grad_adj$perturb_school <- function(x, schid, schl_par = school_list){
    val_mean <- schl_par[[which(schid == names(schl_par))]]
    val_sd <- val_mean / 4
    val_sd <- abs(val_sd)
    y <- x + num_clip(rnorm(1, mean = val_mean, sd = val_sd), -0.45, 0.45)
    y <- ifelse(y <= 0, 0.01, y)
    y <- ifelse(y >= 1, 0.98, y)
    y <- num_clip(y, 0, 1)
    return(y)
  }
  race_prob_star <- runif(7)
  race_prob_star[1] <- race_prob_star[1] * 7
  race_prob_star[2] <- race_prob_star[2] * 1.5
  race_prob_star[3] <- race_prob_star[3] / 3.75
  race_prob_star[4] <- race_prob_star[4] * 2.25
  race_prob_star[5] <- race_prob_star[5] * 2.25
  race_prob_star[6] <- race_prob_star[6] / 3.75
  race_prob_star[7] <- race_prob_star[7] / 4.25
  race_prob_star <- race_prob_star / sum(race_prob_star)
  
  ###################
  # Conduct the simulation
  stu_pop <- simpop(sample(c(2500, 2250, 2750, 3500, 1200, 5000), 1), 
                    control = sim_control(nschls = sample(8:16, 1), 
                                          race_prob = race_prob_star,
                                          n_cohorts = 4L, 
                                          minyear = 1998,
                                          maxyear = 2018, 
                                          assessment_adjustment = assess_adj,
                                          assess_sim_par = assess_sim_par, 
                                          grad_sim_parameters = grad_sim_par, 
                                          grad_adjustment = grad_adj))
  
  # Build analysis file from the datasets in the list produced by simpop
  #
  stu_pop$schools$lea_id <- paste0("0", i)
  out_data <- dplyr::left_join(stu_pop$stu_assess %>% 
                                 filter(grade == 7), 
                               stu_pop$stu_year)
  
  out_data <- out_data %>% filter(cohort_year < 2015) %>%
    filter(year == cohort_year - 2)
  #
  out_data <- out_data %>% select(-exit_type, -enrollment_status, 
                                  -grade_enrolled, -grade_advance, -ndays_attend, 
                                  -ndays_possible)
  out_data <- left_join(out_data, stu_pop$demog_master %>% 
                          select(sid, Sex, Race))
  out_data <- left_join(out_data, stu_pop$hs_outcomes %>% 
                          filter(!is.na(hs_status)) %>%
                          select(sid, grad, chrt_grad, hs_status))
  
  first_hs <- stu_pop$hs_annual %>% group_by(sid) %>% filter(grade == 9) %>% 
    summarize(first_hs = first(schid))
  last_hs <- stu_pop$hs_annual %>% group_by(sid) %>% filter(grade == 12) %>% 
    summarize(last_hs =  last(schid))
  
  
  out_data <- filter(out_data, grade == 7)
  out_data <- left_join(out_data, first_hs)
  out_data$hs_status[is.na(out_data$hs_status)] <- "disappear"
  out_data$hs_status <- factor(out_data$hs_status, 
                               levels = c("disappear", 
                                          "dropout", 
                                          "early", "late", 
                                          "ontime", "still_enroll", 
                                          "transferout"))
  
  out_data <- bind_cols(out_data, 
                        as.data.frame(model.matrix(~ 0 + hs_status, data = out_data)))
  out_data$hs_status <- NULL
  
  out_data <- left_join(out_data, 
                        stu_pop$schools %>% 
                          select(schid, name, enroll, male_per, frpl_per, 
                                 lep_per, gifted_per, lea_id, poverty_desig) %>% 
                          rename_all(funs(paste0("sch_g7_", .))), 
                        by = c("schid" = "sch_g7_schid"))
  
  out_data <- left_join(out_data, 
                        stu_pop$schools %>% 
                          select(schid, name, enroll, male_per, frpl_per, 
                                 lep_per, gifted_per, lea_id, poverty_desig) %>% 
                          rename_all(funs(paste0("first_hs_", .))), 
                        by = c("first_hs" = "first_hs_schid"))
  
  # Conver back to dataframe
  out_data <- as.data.frame(out_data)
  
  out_data$att_rate <- 1 - out_data$att_rate
  
  # Define an export
  names(out_data)[1:26] <- c("sid", "sch_g7_code", "year", "grade", "scale_score_7_math", 
                       "scale_score_7_read", "age", "frpl_7", "ell_7", "iep_7", 
                       "gifted_7", "cohort_year", "cohort_grad_year", 
                       "pct_days_absent_7", "male", "race_ethnicity", 
                       "any_grad", "grad_chrt_yr", "first_hs_code", 
                       "disappeared", "dropout", "early_grad",
                       "late_grad", "ontime_grad", 
                       "still_enrolled", "transferout")
  
  # Race = first letter
  out_data$race_ethnicity %<>% as.character %>% str_trunc(width = 9)
  # sex = first letter
  out_data$male %<>% as.character %>% substr(1, 1)
  # lep = 1/0
  out_data$ell_7 %<>% as.numeric
  # econ_dis = 1/0
  out_data$frpl_7 %<>% as.numeric
  # Reform SES variable with additional levels
  out_data$frpl_7 <- sapply(out_data$frpl_7, 
                          function(x) ifelse(runif(1) > 0.975, 9, x))
  out_data$frpl_7 <- sapply(out_data$frpl_7, 
                          function(x) ifelse(x == 1 & runif(1) > 0.75, 2, x))
  
  out_data$sch_g7_code <- paste0(out_data$sch_g7_lea_id, "-", out_data$sch_g7_code)
  out_data$first_hs_code <- paste0(out_data$first_hs_lea_id, "-", out_data$first_hs_code)
  if(i == 1){
    sea_data <- out_data
  } else {
    sea_data <- bind_rows(sea_data, out_data)
  }
  cat("********************************")
  cat(paste0("Iteration: ", i))
  cat("*************************************")
}


rm(first_hs, last_hs, out_data, stu_pop, assess_adj, 
   grad_adj, assess_sim_par, grad_sim_par); gc()



# Get sample model output to check relationships among variables
table(sea_data$ontime_grad)
base_mod <- glm(ontime_grad ~ scale_score_7_math + scale_score_7_read + 
                  pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male + race_ethnicity, 
                data = sea_data)
math_mod <- glm(ontime_grad ~ scale_score_7_math, data = sea_data)
sch_mod <- glm(ontime_grad ~ scale_score_7_math + scale_score_7_read + 
                  pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male + 
                 race_ethnicity + sch_g7_male_per + sch_g7_frpl_per + 
                 sch_g7_lep_per + sch_g7_gifted_per + first_hs_frpl_per + 
                 first_hs_lep_per + first_hs_gifted_per + first_hs_male_per, 
                data = sea_data)
sch_dist_mod <- glm(ontime_grad ~ scale_score_7_math + scale_score_7_read + 
                 pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male + 
                 race_ethnicity + sch_g7_male_per + sch_g7_frpl_per + 
                 sch_g7_lep_per + sch_g7_gifted_per + first_hs_frpl_per + 
                 first_hs_lep_per + first_hs_gifted_per + first_hs_male_per + 
                   sch_g7_lea_id, 
               data = sea_data)


source("R/funs.R")
auc.glm(base_mod)
auc.glm(math_mod)
auc.glm(sch_mod)
auc.glm(sch_dist_mod)
table(sea_data$ontime_grad)
plotdf <- sea_data %>% group_by(sch_g7_code) %>% summarize(count = n(), grad = mean(ontime_grad))
summary(plotdf$grad)
table(is.na(sea_data$scale_score_7_math))

by(sea_data$scale_score_7_math, sea_data$race_ethnicity, FUN = mean, na.rm=TRUE)
by(sea_data$scale_score_7_math, sea_data$frpl_7, FUN = mean, na.rm=TRUE)
by(sea_data$scale_score_7_math, sea_data$male, FUN = mean, na.rm=TRUE)

for(var in c("male", "race_ethnicity", "frpl_7", "iep_7", "ell_7", "gifted_7")){
     print(var)
     print(
         prop.table(
           table(grad = sea_data$ontime_grad, var = sea_data[, var], 
                 useNA = "always"), 
     margin = 2)
   )
}

cor(sea_data[, c("ontime_grad", "scale_score_7_math", "pct_days_absent_7")], 
    use = "pairwise.complete.obs")

## Overwrite absences to be more correlated with assessment scores and graduation




## G7 school mixtures
## G7 school performance
## Test score polynomials
## Scale and center predictors
## Test the models and see that they make sense


library(caret)
library(doFuture)
library(future)
plan(multiprocess(workers = 12))
registerDoFuture()

sea_data$ontime_grad_fac <- ifelse(sea_data$ontime_grad == 1, "grad", "nongrad")
sea_data$ontime_grad_fac <- factor(sea_data$ontime_grad_fac)

train_rows <- sample(1:nrow(sea_data), 25000)
test_rows <- sea_data[-train_rows,]


zzz <- train(ontime_grad_fac ~ scale_score_7_math + scale_score_7_read + 
               pct_days_absent_7 + ell_7 + iep_7 + frpl_7 + male + race_ethnicity, 
             data = sea_data[train_rows,], 
             method = "nb", 
             tuneLength = 24, 
             trControl = trainControl(method = "repeatedcv", classProbs = TRUE, 
                                      summaryFunction = twoClassSummary), 
             metric = "ROC")

max(zzz$results$ROC)
sea_data$ontime_grad_fac <- NULL

# TODO: Add example indicator for early warning to the data
# phooey <- predict(zzz, newdata = export)
sea_data$vendor_ews_score <- predict(zzz, newdata = sea_data, type = "prob")$grad

# Salt the data
sea_data$pct_days_absent_7 <- sapply(sea_data$pct_days_absent_7, 
                                   function(x) ifelse(runif(1) > 0.995, 0.8, x))
sea_data$pct_days_absent_7 <- sapply(sea_data$pct_days_absent_7, 
                                     function(x) ifelse(runif(1) > 0.999, 1.4, x))
sea_data$pct_days_absent_7 <- sapply(sea_data$pct_days_absent_7, 
                                   function(x) ifelse(runif(1) > 0.995, NA, x))

sea_data$scale_score_7_math <- sapply(sea_data$scale_score_7_math, 
                                     function(x) ifelse(runif(1) > 0.99, NA, x))

sea_data$scale_score_7_read <- sapply(sea_data$scale_score_7_read, 
                                      function(x) ifelse(runif(1) > 0.99, NA, x))


# TODO: Add missing values to "first_coop_code", "male", "race_ethnicity"
sea_data$male <- sapply(sea_data$male, 
                                   function(x) ifelse(runif(1) > 0.99, NA, x))
sea_data$race_ethnicity <- sapply(sea_data$race_ethnicity, 
                      function(x) ifelse(runif(1) > 0.99, NA, x))



lea_names <- sim_control()$school_names 
lea_names <- lea_names[!lea_names %in% unique(sea_data$sch_g7_name)]
lea_names <- sample(lea_names, length(unique(sea_data$sch_g7_lea_id)))
coop_names <- sim_control()$school_names 
coop_names <- coop_names[!coop_names %in% c(unique(sea_data$sch_g7_name), lea_names)]

# assign LEA names
lea_df <- data.frame(lea_id = unique(sea_data$sch_g7_lea_id), 
                     sch_g7_lea_name = lea_names)
sea_data <- left_join(sea_data, lea_df, 
                    by = c("sch_g7_lea_id" = "lea_id"))
lea_df$first_hs_lea_name <- lea_df$sch_g7_lea_name
lea_df$sch_g7_lea_name <- NULL
sea_data <- left_join(sea_data, lea_df, 
                    by = c("first_hs_lea_id" = "lea_id"))

sample2 <- function(x, sample.size){
  split(x, sample(ceiling(seq_along(x)/sample.size)))
}

# Add coop_codes
coop_groups <- sample2(unique(sea_data$sch_g7_lea_id), 5)
names(coop_groups) <- sample(coop_names, length(coop_groups))
sea_data$coop_name_g7 <- ""
sea_data$coop_name_first_hs <- ""
for (i in 1:length(coop_groups)) {
  sea_data$coop_name_g7 <- ifelse(sea_data$sch_g7_lea_id %in% coop_groups[[i]], 
                                names(coop_groups)[i], sea_data$coop_name_g7)
  sea_data$coop_name_first_hs <- ifelse(sea_data$first_hs_lea_id %in% coop_groups[[i]], 
                                      names(coop_groups)[i], sea_data$coop_name_first_hs)
}

# Clean up data names
sea_data$year_of_graduation <- sea_data$grad_chrt_yr
sea_data$grad_chrt_yr <- NULL
sea_data$pct_days_absent_7 <- sea_data$pct_days_absent_7 * 100

# sea_data$coop_code_g7 <- sapply(sea_data$coop_code_g7, 
#                               function(x) ifelse(runif(1) > 0.99, NA, x))

sea_data$male <- ifelse(sea_data$male == "M", 1, 0)
sea_data$gifted_7 <- as.numeric(sea_data$gifted_7)
sea_data$iep_7 <- as.numeric(sea_data$iep_7)
sea_data$year_of_graduation[!is.finite(sea_data$year_of_graduation)] <- NA

sea_data$ontime_grad[sea_data$any_grad == 1 & sea_data$cohort_grad_year == sea_data$year_of_graduation] <- 1
sea_data$ontime_grad[sea_data$any_grad == 1 & sea_data$cohort_grad_year != sea_data$year_of_graduation] <- 0
# summary(gifted_7)
# summary(iep_7)

#  nrow(train_data) == n_distinct(train_data$sid)
## Export the data
sea_data <- as.data.frame(sea_data)
save(sea_data, file = "data/montucky.rda")
haven::write_dta(sea_data, path = "data/montucky.dta", version = 11L)
