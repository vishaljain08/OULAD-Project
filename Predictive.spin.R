setwd("C:/Users/visha/OneDrive - Coventry University/UPV/ADE/Project/anonymisedData")
library(dplyr)
library(tidyverse)
library(caret)
library(doSNOW)
library(xgboost)
library(ggplot2)

student.info <- read.csv("studentInfo.csv")
student.assessment <- read.csv("studentAssessment.csv")

# Merging the studentInfo.csv and studentAssessment.csv to get more info about
#the students in the same dataframe
mergedInfo <- merge(x=student.assessment, y=student.info[,c("id_student","gender", "age_band", "studied_credits")], by="id_student")

#Omitting any rows with NA values 
mergedInfo <- na.omit(mergedInfo)


#Converting any character variables to factors
mergedInfo <- 
  mergedInfo %>%
  mutate_if(is.character, as.factor)

nearZeroVar(mergedInfo, saveMetrics = TRUE)

mergedInfo <-
  mergedInfo %>%
  select(-is_banked)


# Setting a seed to ensure the reproducibility of our data partition.
set.seed(2020)

#Creating a trainIndex object that stores 70% of the data, to follow the widely
#used 70-30 pattern for training and testing. 

trainIndex <- createDataPartition(mergedInfo$score,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

score.train <- mergedInfo[trainIndex,]
score.test <- mergedInfo[-trainIndex,]

train.control <- trainControl(number = 10,
                              repeats = 1,
                              method = "repeatedcv",
                              verboseIter = TRUE)

cl <- makeCluster(8, type = "SOCK")
registerDoSNOW(cl)

forestFit <- train(score ~ .,
                 data = score.train,
                 method = "svmLinear",
                 trControl = train.control,
                 verbose = TRUE)

stopCluster(cl)
forestFit

preds <- predict(forestFit, score.test)

score.test.augmented <-
  score.test %>%
  mutate(pred = predict(forestFit, score.test),
         obs = score)

defaultSummary(as.data.frame(score.test.augmented))


ggplot(score.test.augmented, aes(x=pred,y=score)) +
  geom_point() +
  geom_smooth(method='lm', formula= y~x)