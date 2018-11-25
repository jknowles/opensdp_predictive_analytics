# Calculate the AUC of a GLM model easily
# Jared Knowles
# model = a fitted glm in R
# newdata = an optional data.frame of new fitted values
auc.glm <- function(model, newdata = NULL){
  if(missing(newdata)){
    resp <- model$y
    # if(class(resp) == "numeric"){
    #   resp <- factor(resp)
    # }
    pred <- model$fitted.values
  } else{
    newdata <- as.data.frame(newdata)
    resp <- newdata[, all.vars(model$formula)[1]]
    pred <- predict(model, newdata, type = "response")
  }
  out <- pROC::auc(resp, pred)
  return(as.numeric(out))
}

logit_rsquared <- function(model){
  nullmod <- glm(model$y ~ 1, family="binomial")
  as.numeric(1-logLik(model)/logLik(nullmod))
}

# Calculate the prevalence of the second class in a two-class outcome
# Jared Knowles
# resp = a vector of bivariate responses (0, 1)
get_prevl <- function(resp){
  table(resp)[[2]] / sum(table(resp))
}


# Need to import pROC
# Calculate the optimal topleft threshold
# Jared KNowles
# resp = vector of outcome
# pred = predicted outcome
get_thresh <- function(resp, pred){
  prev <- get_prevl(resp)
  rocobj <- roc(resp, pred)
  out <- coords(rocobj, "best",  ret = "threshold",
                best.method = "closest.topleft", best.weights = c(0.3, prev))
  return(out)
}

# From a glm or lm or lmer or glmer model create a confusion matrix
# model = model object
# data = newdata to use if needed
# thresh = optional value from 0 to 1 to cut predictions
conf_matrix <- function(model, data = NULL, thresh = NULL) {
  if(missing(data)){
    if(class(model)[1] %in% c("lmerMod", "glmerMod")){
      data <- model@frame
    } else{
      data <- model$model
    }
  }
  if(missing(thresh)){
    cut_thresh <- mean(predict(model, data, type ="response"))
  } else{
    cut_thresh <- thresh
  }

  prediction <- ifelse(predict(model, data, type='response') > cut_thresh, TRUE, FALSE)
  if(class(model)[1] %in% c("lmerMod", "glmerMod")){
    confusion  <- table(pred = prediction, obs = as.logical(model@frame[, 1]))
  } else{
    confusion  <- table(pred = prediction, obs = as.logical(model$y))
  }
  confusion  <-   confusion  <- cbind(confusion,
                                      c(1 - confusion[1,1]/(confusion[1,1]+
                                                              confusion[2,1]),
                                        1 - confusion[2,2]/(confusion[2,2]+confusion[1,2])))
  confusion  <- as.data.frame(confusion)
  names(confusion) <- c('Obs FALSE', 'Obs TRUE', 'class.error')
  rownames(confusion) <- c('Pred FALSE', 'Pred TRUE')
  confusion
}

# Replace all missing values in a vector with a numeric 0
zeroNA <- function(x){
  x[is.na(x)] <- 0
  return(x)
}

# Cluster standard errors
get_CL_vcov <- function(model, cluster){
  # cluster is an actual vector of clusters from data passed to model
  # from: http://rforpublichealth.blogspot.com/2014/10/easy-clustered-standard-errors-in-r.html
  require(sandwich, quietly = TRUE)
  require(lmtest, quietly = TRUE)
  cluster <- as.character(cluster)
  # calculate degree of freedom adjustment
  M <- length(unique(cluster))
  N <- length(cluster)
  K <- model$rank
  dfc <- (M/(M-1))*((N-1)/(N-K))
  # calculate the uj's
  uj  <- apply(estfun(model), 2, function(x) tapply(x, cluster, sum))
  # use sandwich to get the var-covar matrix
  vcovCL <- dfc*sandwich(model, meat=crossprod(uj)/N)
  return(vcovCL)
}

get_bowers_data <- function() {
  url <- "https://raw.githubusercontent.com/jknowles/DEWSatDoGoodData2016/master/data/BowersEWSReviewData.csv"
  ews <- read.csv(url)
  return(ews)
}

bowers_plot <- function() {
  # Get data
  ews <- get_bowers_data()
  # format data
  # Clean up the coding of the data with better labels
  # This is optional, but it highlights some reference
  # models
  ews$flag <- "Other EWI"
  ews$flag[ews$id == 1 | ews$id == 2] <- "Chicago On-Track"
  ews$flag[ews$id > 3 & ews$id < 14] <- "Balfanz ABC"
  ews$flag[ews$id == 85] <- "Muthen Math GMM"
  ews$flag[ews$id == 19] <- "Bowers GPA GMM"
  ews$flag <- factor(ews$flag)
  ews$flag <- relevel(ews$flag, ref = "Other EWI")
  # Set colors
  mycol <- c("Other EWI" = "gray70", "Chicago On-Track" = "blue",
             "Balfanz ABC" = "purple",
             "Muthen Math GMM" = "orange",
             "Bowers GPA GMM" = "dark red")
  # The big block of plotting code, you only need to modify the last
  # couple of lines, the rest adds annotations and labels which you
  # can switch on and off if you like
  p1 <- ggplot(ews) + aes(x = 1-specificity, y = sensitivity,
                    shape = flag, size = I(4), color = flag) +
    geom_point() +
    # Label the shape legend
    scale_shape("EWI Type") +
    # Label the color legend and customize it
    scale_color_manual("EWI Type", values = mycol) +
    # Add a reference 45 degree line representing random chance
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    # Set the scales of the chart to not distort distances
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    # Add a simple theme to avoid clutter
    theme_bw() +
    # Label the axes
    labs(x = "False Alarm Proportion", y = "True Positive Proportion",
         title = "ROC Accuracy of Early Warning Indicators") +
    # Place the legend in the bottom right
    theme(legend.position = c(0.8, 0.2),
          legend.background = element_rect(fill = NULL,
                                           color = "black")) +
    # Add arrows to annotate the plot
    annotate(geom = "segment", x = 0.55, y = 0.625,
             yend = 0.785, xend = 0.4,
             arrow = arrow(length = unit(0.5, "cm"))) +
    # Label the arrow
    annotate(geom="text", x = .365, y = .81, label="Better Prediction") +
    annotate(geom="segment", x = 0.65, y = 0.625, yend = 0.5,
             xend = 0.75, arrow = arrow(length = unit(0.5, "cm"))) +
    annotate(geom="text", x = .75, y = .48, label="Worse Prediction") +
    annotate(geom="text", x = .75, y = .78, angle = 37, label="Random Guess")
  
  return(p1)
  
    
}
