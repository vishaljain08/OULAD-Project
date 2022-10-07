setwd("C:/Users/visha/OneDrive - Coventry University/UPV/ADE/Project/anonymisedData")
library(dplyr)
library(tidyverse)
library(caret)
library(doSNOW)
library(xgboost)
library(ggplot2)

student.info <- read.csv("studentInfo.csv")

student.info$Dropout <- ifelse((student.info$final_result == "Withdrawn"),
                           "Y", "N")


#Converting any character variables to factors
student.info <- 
  student.info %>%
  mutate_if(is.character, as.factor)

student.info <-
  student.info %>%
  select(-final_result)

student.info <- na.omit(student.info)


# Setting a seed to ensure the reproducibility of our data partition.
set.seed(2020)

#Creating a trainIndex object that stores 70% of the data, to follow the widely
#used 70-30 pattern for training and testing. 

trainIndex <- createDataPartition(student.info$Dropout,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

dropout.train <- student.info[trainIndex,]
dropout.test <- student.info[-trainIndex,]

train.control <- trainControl(number = 10,
                              repeats = 3,
                              method = "repeatedcv")

summary(dropout.test)

cl <- makeCluster(8, type = "SOCK")
registerDoSNOW(cl)

forestFit <- train(Dropout ~ .,
                   data = dropout.train,
                   method = "xgbTree",
                   trControl = train.control)

stopCluster(cl)
forestFit

preds <- predict(forestFit, dropout.test)

dropout.test.augmented <-
  dropout.test %>%
  mutate(pred = predict(forestFit, dropout.test),
         obs = Dropout)

defaultSummary(as.data.frame(dropout.test.augmented))


confusionMatrix(preds, dropout.test.augmented$Dropout)
