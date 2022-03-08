#This is the R script for the machine learning modeling task as required
#Machine Learning/modeling part only as instructed in the course

#Submitted by Dodgecarl Incila
# DATA Science Capstone
# Heart Disease Prediction

#Part 1. Preparation
#install necessary packages


#run libraries for this report
library(caret)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggcorrplot)
library(ggplot2)
library(randomForest)



#Retrieve Data from github repo
dat <- read.csv("https://raw.githubusercontent.com/inciladc/rfiles/main/heart101.csv")

#examine structure
str(dat)

#compute correlation
cor_dat <- round(cor(dat),2)
cor_dat
#Visualize output
ggcorrplot(cor_dat, hc.order = TRUE, type = "lower", lab = FALSE, title = "Correlation Matrix on Heart Disease features")

#store chosen predictors on object mod_dat 
mod_dat <- dat[,c(3,9,10,11,13,14)]

#examine structure of mod_dat
str(mod_dat)


# Build using Linear Regression

#Check Linearity
pairs(mod_dat[1:6])

#run regression on identified features from mod_dat
result <- lm(condition~cp+exang+oldpeak+slope+thal, data = mod_dat)

#view summary
summary(result)

#Improved model: we will use cp and thal only for this regression
imp <- lm(condition~cp+thal,data = mod_dat)

#ANOVA test
anova(result,imp)

#Prediction

#cp values = 0,1,2,3
#thal values = 0,1,2
#Scenario 1: typical angina, normal thal
predict(imp,data.frame(cp=0,thal=0))

#Scenario 2: non-anginal pain, normal thal
predict(imp,data.frame(cp=1,thal=0))

#Scenario 3: asymptomatic, fixed defect
predict(imp,data.frame(cp=3,thal=1))

###########################################

#Build using Random Forest

#set seed for reproducibility
set.seed(222)

##Data Partition
#store mod_dat to dp for partition
dp <- mod_dat

#set condition as a factor
dp$condition <- as.factor(dp$condition)

#set partition at 70/30 for train and test respectively
ind <- sample(2, nrow(dp), replac=T, prob = c(0.7,0.3))
rf_train <- dp[ind==1,]
rf_test <- dp[ind==2,]

#See structures of train and test set
str(rf_train)

str(rf_test)

#Run RF on train set
rf <- randomForest(condition~cp+thal+exang+oldpeak+slope, data = rf_train)
print(rf)


#Predict using RF model
#predict
p1 <- predict(rf,rf_train)

#see confusion matrix
confusionMatrix(p1, rf_train$condition, positive = "1")

#Apply on test set
p2 <- predict(rf,rf_test)
confusionMatrix(p2,rf_test$condition,positive = "1")

########end#######










