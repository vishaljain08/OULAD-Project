# loading the tidyverse and caret packages
library(tidyverse)
library(caret)
library(readr)
library(doSNOW)
library(randomForest)

# Loading the data
studentInfo <- read_csv("studentInfo.csv")
studentVle <- read_csv("studentVle.csv")

#selecting variables
studentInfo <- studentInfo %>% select(code_module, code_presentation, id_student, gender, region, 
                                      highest_education, imd_band, 
                                      age_band, num_of_prev_attempts, studied_credits, 
                                      disability, final_result)
studentVle <- studentVle %>% select(code_module, code_presentation, id_student, sum_click)
#sum by group number of clicks in VLE by each student
studentVle <- aggregate(x = studentVle$sum_click, by = list(studentVle$code_module, studentVle$id_student, studentVle$code_presentation), FUN = sum)
studentVle <- studentVle %>% rename(code_module = Group.1, id_student = Group.2, code_presentation=Group.3, sum_click=x)

#join both tables to have student background with his VLE activity
df <- inner_join(studentInfo, studentVle, by = c("id_student", "code_module", "code_presentation"))

# eliminating any rows that have any missing data
df <- na.omit(df)

# run the nearZeroVar function to determine
# if there are variables with NO variability
nearZeroVar(df, saveMetrics = TRUE)

# converting the text variables in dataset into factors
df <-  df %>%  mutate_if(is.character, as.factor)

#seed to ensure the reproducibility of data partition
set.seed(2020)
#creating new object to split the data for training and testing sets
trainIndex <- createDataPartition(df$final_result,
                                  p = .3,
                                  list = FALSE,
                                  times = 1)

df_train <- df[trainIndex,]

df_test <- df[-trainIndex,]

#removing additional variable
df_test <-  df_test %>%  select(-temp_id)
df_train <- df_train %>% select(-temp_id)
df <- df %>% select(-temp_id)

#train control
train_control <-  trainControl(method = "repeatedcv", number = 10, repeats = 10)

#switching on parallel computing
cl <- makeCluster(7, type = "SOCK")
registerDoSNOW(cl)

#training the model
rf_fit <- train(final_result ~ .,
                data = df_train,
                method = "cforest",
                metric = "Accuracy")

#stop paralel computing
stopCluster(cl)

# summary of the model
rf_fit

# Plotting predictor importance (requires library(randomForest) to be installed)
ggplot(varImp(rf_fit))

#creating a new object for the testing data including predicted values
df_test_augmented <-  df_test %>%  mutate(pred = predict(rf_fit, df_test),
                                          obs = final_result)

# Transform this new object into a data frame
defaultSummary(as.data.frame(df_test_augmented))

plot(df_test_augmented$pred, df_test_augmented$obs,col="lightblue")
abline(lm(df_test_augmented$obs ~ df_test_augmented$pred), col="red", lwd=3)