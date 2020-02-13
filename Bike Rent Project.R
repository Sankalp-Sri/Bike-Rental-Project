rm(list = ls())

#setting the working directory
setwd("C:/Users/acer/Desktop/edWisor Project 2/Bike-Rent-R")
getwd()

seed=112
#Loading important packages & libraries
#install.packages("randomForest)
# install.packages("rpart")
#install.packages("dplyr)
#install.packages("caTools")
#install.packages("caret")
#install.packages("MASS")
#install.packages("MLmetrics")
library(MLmetrics)
library(rpart)
library(randomForest)
library(corrplot)
library(dplyr)
library(caTools)
library(MASS)
library(caret)
library(ggplot2)
library(lattice)

#importing dataset
dataset = read.csv("day.csv")


####### Exploratory Data Analysis #################
View(dataset)

#see the structure of dataset
str(dataset)

#1. We can see from the info that there are 16 variables in the dataset in which casual,registered & cnt are dependent variable and rest of them are independent variable.
#2. It can be seen that there are no null values in the data set and each of the variable is having 731 non-null entries
#3. It can also be infered that only the dteday variable has the dtype as factor and rest of the variables are having numeric dtype i.e. either num or int.

#see the summary of given data set
summary(dataset)


######### 2.1 Data Cleaning ######

### Renaming the data features
dataset = dataset %>% 
           rename(
            date = dteday,
            year = yr,
            month= mnth,
            weather_condition= weathersit,
            temperature = temp,
            feeled_temperature = atemp,
            humidity = hum,
            casual_users = casual,
            registered_users = registered,
            total_count = cnt
          )

#Variable - "instant"
#It can be seen that instant variable is just representing the index of the samples so we drop it from our dataset
dataset = dataset[,-1]

#Variable - "date"
#Now it can also be seen that date variable has 731 unique values which is not going to help us infer anything about the bike count
#So let's remove this variable from our dataset
dataset = dataset[,-1]

#Missing Value Analysis
sum(is.na(dataset))

#Outlier Analysis
boxplot(dataset[,c(8,9,10,11)],xlab = "Continuous Features", ylab = "Range",col = "green")

# as some outliers can be seen in the windspeed and humidity variable but they can be ignore as they are natural phenomenon and can be trated as natural outliers

######### Feature Engineering #############

#Let's first split the data into train and test data set to avoid data leakage problem


set.seed(seed)

split = sample.split(dataset$total_count,SplitRatio = 0.75)

train_set = subset(dataset, split==TRUE)
test_set = subset(dataset,split==FALSE)

print(dim(train_set))
print(dim(test_set))


#Checking the correlation between dependent variables
dependent_variable = train_set[c(12,13,14)]

corrplot(cor(dependent_variable))

#Dropping registered_users & casual_users from the train and test dataset
train_set = train_set[c(-12,-13)]
test_set = test_set[c(-12,-13)]


# Let's see the relationship between numerical features and target variable with scatter plot
numeric_features = train_set[c(8,9,10,11)] 
plot(train_set$temperature,train_set$total_count, col = "cyan")
plot(train_set$feeled_temperature,train_set$total_count,col = "green")
plot(train_set$windspeed,train_set$total_count,col = "yellow")
plot(train_set$humidity,train_set$total_count,col="red")

#Check the correlation among other numeric features
corrplot(cor(numeric_features),method = "number")

#As the correlation between both the temerature variables is very high we would be dropping feeled_tempereature variable from our dataset

train_set = train_set[c(-9)]
test_set = test_set[c(-9)]


#### Let's check the correlation between categorical variable

# Let's analyise the categorical variables now

##Variable - "Season"

table(train_set$season)

#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,season),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each season but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$season = as.factor(train_set$season)
test_set$season = as.factor(test_set$season)

anova_1 = aov(total_count~season,data = train_set)
summary(anova_1)

#Calculate the relative mean deviation

bikecount_mean = mean(train_set$total_count)

season_1 = round((mean(train_set$total_count[train_set$season==1])-bikecount_mean)/bikecount_mean,3)
season_2 = round((mean(train_set$total_count[train_set$season==2])-bikecount_mean)/bikecount_mean,3)
season_3 = round((mean(train_set$total_count[train_set$season==3])-bikecount_mean)/bikecount_mean,3)
season_4 = round((mean(train_set$total_count[train_set$season==4])-bikecount_mean)/bikecount_mean,3)


#Encode the season levels in train and test dataset
train_set$season = as.numeric(train_set$season)
test_set$season = as.numeric(test_set$season)

train_set$season[train_set$season==1] <- season_1
train_set$season[train_set$season==2] <- season_2
train_set$season[train_set$season==3] <- season_3
train_set$season[train_set$season==4] <- season_4

test_set$season[test_set$season==1] <- season_1
test_set$season[test_set$season==2] <- season_2
test_set$season[test_set$season==3] <- season_3
test_set$season[test_set$season==4] <- season_4



### Variable - "year"

table(train_set$year)


#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,year),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each year but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$year = as.factor(train_set$year)
test_set$year = as.factor(test_set$year)
anova_2 = aov(total_count~year,data = train_set)
summary(anova_2)

#Calculating the rel mean deviations
year_0 = round((mean(train_set$total_count[train_set$year==0])-bikecount_mean)/bikecount_mean,3)
year_1 = round((mean(train_set$total_count[train_set$year==1])-bikecount_mean)/bikecount_mean,3)

#Encode the year levels in train & test set
train_set$year = as.numeric(train_set$year)
test_set$year = as.numeric(test_set$year)

train_set$year[train_set$year==0] <- year_0
train_set$year[train_set$year==1] <- year_1

test_set$year[test_set$year==0] <- year_0
test_set$year[test_set$year==1] <- year_1

###########  Variable - "Month"  ########################
table(train_set$month)
#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,month),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each month but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$month = as.factor(train_set$month)
test_set$month = as.factor(test_set$month)
anova_3 = aov(total_count~month,data = train_set)
summary(anova_3)

#Calculating the rel mean deviations
month_1 = round((mean(train_set$total_count[train_set$month==1])-bikecount_mean)/bikecount_mean,3)
month_2 = round((mean(train_set$total_count[train_set$month==2])-bikecount_mean)/bikecount_mean,3)
month_3 = round((mean(train_set$total_count[train_set$month==3])-bikecount_mean)/bikecount_mean,3)
month_4 = round((mean(train_set$total_count[train_set$month==4])-bikecount_mean)/bikecount_mean,3)
month_5 = round((mean(train_set$total_count[train_set$month==5])-bikecount_mean)/bikecount_mean,3)
month_6 = round((mean(train_set$total_count[train_set$month==6])-bikecount_mean)/bikecount_mean,3)
month_7 = round((mean(train_set$total_count[train_set$month==7])-bikecount_mean)/bikecount_mean,3)
month_8 = round((mean(train_set$total_count[train_set$month==8])-bikecount_mean)/bikecount_mean,3)
month_9 = round((mean(train_set$total_count[train_set$month==9])-bikecount_mean)/bikecount_mean,3)
month_10 = round((mean(train_set$total_count[train_set$month==10])-bikecount_mean)/bikecount_mean,3)
month_11= round((mean(train_set$total_count[train_set$month==11])-bikecount_mean)/bikecount_mean,3)
month_12 = round((mean(train_set$total_count[train_set$month==12])-bikecount_mean)/bikecount_mean,3)

#Encoding the month levels in train and test dataset
train_set$month = as.numeric(train_set$month)
test_set$month = as.numeric(test_set$month)

  train_set$month[train_set$month==1] <- month_1
  train_set$month[train_set$month==2] <- month_2
  train_set$month[train_set$month==3] <- month_3
  train_set$month[train_set$month==4] <- month_4
  train_set$month[train_set$month==5] <- month_5
  train_set$month[train_set$month==6] <- month_6
  train_set$month[train_set$month==7] <- month_7
  train_set$month[train_set$month==8] <- month_8
  train_set$month[train_set$month==9] <- month_9
  train_set$month[train_set$month==10] <- month_10
  train_set$month[train_set$month==11] <- month_11
  train_set$month[train_set$month==12] <- month_12
  
  test_set$month[test_set$month==1] <- month_1
  test_set$month[test_set$month==2] <- month_2
  test_set$month[test_set$month==3] <- month_3
  test_set$month[test_set$month==4] <- month_4
  test_set$month[test_set$month==5] <- month_5
  test_set$month[test_set$month==6] <- month_6
  test_set$month[test_set$month==7] <- month_7
  test_set$month[test_set$month==8] <- month_8
  test_set$month[test_set$month==9] <- month_9
  test_set$month[test_set$month==10] <- month_10
  test_set$month[test_set$month==11] <- month_11
  test_set$month[test_set$month==12] <- month_12
  

########  Variable - "Holiday"  ##########

table(train_set$holiday)
#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,holiday),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each holiday level but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$holiday = as.factor(train_set$holiday)
anova_4 = aov(total_count~holiday,data = train_set)
summary(anova_4)

#As the mean for each holiday level doesn't vary significantly so just drop this variable from our train as well as test dataset

train_set = train_set[-4]
test_set = test_set[-4]

#########  Variable - "weekday" ############

table(train_set$weekday)

#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,weekday),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each weekday level but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$weekday = as.factor(train_set$weekday)
anova_5 = aov(total_count~weekday,data = train_set)
summary(anova_5)

#As the mean for each weekday level doesn't vary significantly so just drop this variable from our train as well as test dataset

train_set = train_set[-4]
test_set = test_set[-4]

##Variable - "workingday"

table(train_set$workingday)


#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,workingday),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each holiday level but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$workingday = as.factor(train_set$workingday)
anova_6 = aov(total_count~workingday,data = train_set)
summary(anova_6)

#As the mean for each working day level doesn't vary significantly so just drop this variable from our train as well as test dataset

train_set = train_set[-4]
test_set = test_set[-4]


#######  Variable - "weather_condition" ########

table(train_set$weather_condition)

#Creating a subset of our dataset according to season levels

summarise_at(group_by(train_set,weather_condition),vars(total_count),funs(mean(.,na.rm=TRUE)))

# As can be seen that the mean value of bike count is different for each holiday level but we need to check wheather this difference is significant or not
# For this we need to apply one way anova test on our dataset

#Now let's run the one way anova test to see the significance in the means
train_set$weather_condition = as.factor(train_set$weather_condition)
test_set$weather_condition = as.factor(test_set$weather_condition)
anova_7 = aov(total_count~weather_condition,data = train_set)
summary(anova_7)

#As the mean for each weather_condition level vary significantly so we'll keep this variable

#Calculating the rel mean deviation
weather_condition_1 = round((mean(train_set$total_count[train_set$weather_condition==1])-bikecount_mean)/bikecount_mean,3)
weather_condition_2 = round((mean(train_set$total_count[train_set$weather_condition==2])-bikecount_mean)/bikecount_mean,3)
weather_condition_3 = round((mean(train_set$total_count[train_set$weather_condition==3])-bikecount_mean)/bikecount_mean,3)


#Encode the weather_condition levels of train and test dataset
train_set$weather_condition = as.numeric(train_set$weather_condition)
test_set$weather_condition = as.numeric(test_set$weather_condition)

train_set$weather_condition[train_set$weather_condition==1] <- weather_condition_1
train_set$weather_condition[train_set$weather_condition==2] <- weather_condition_2
train_set$weather_condition[train_set$weather_condition==3] <- weather_condition_3

test_set$weather_condition[test_set$weather_condition==1] <- weather_condition_1
test_set$weather_condition[test_set$weather_condition==2] <- weather_condition_2
test_set$weather_condition[test_set$weather_condition==3] <- weather_condition_3


#correlation between season and month variable
chisq.test(table(train_set$season,train_set$month))
#p-value <<0.05 and chi square test statistics is very high thus high dependency between these variables
# We will drop the season variable form our test and train dataset

train_set = train_set[-1]
test_set = test_set[-1]

#correlation between month and year variable
chisq.test(table(train_set$month,train_set$year))
#AS p-value is 0.99 they are totally independent

#correlation between month and weather condition
chisq.test(table(train_set$weather_condition,train_set$month))
#month and weather_condition are independent as p-value >0.05



############### Modeling #####################

######## Linear Regression ##########

set.seed(seed)
model_LR = glm(total_count~.,data = train_set)
summary(model_LR)

y_pred_LR = predict(model_LR,newdata = test_set[-7],sees=seed)

######### Decision Tree Regressor #######

set.seed(seed)
model_DT = rpart(total_count~.,data = train_set,method = "anova")
summary(model_DT)
plot(model_DT)
text(model_DT)
y_pred_DT = predict(model_DT,newdata = test_set[-7])


############ Random Forest Regressor ############

set.seed(seed)
model_RF = randomForest(total_count~.,data = train_set)
summary(model_RF)
plot(model_RF)
y_pred_RF = predict(model_RF,newdata = test_set[-7])



######## MODEL EVALUATION ################
y_true = Matrix::t(test_set[7])


performance_metrics = function(y_true,y_pred) {
  r2score = R2_Score(y_pred,y_true)
  RMSE_E = RMSE(y_pred,y_true)
  RMSLE_E = RMSLE(y_pred,y_true)
  paste0("R2_Score :",r2score, ", RMSE :",RMSE_E, ", RMSLE :",RMSLE_E)
}

#Performance metrics for Linear Regression model
performance_metrics(y_true,y_pred_LR)

#Performance metrics for Decision Tree Regressor
performance_metrics(y_true,y_pred_DT)

#Performance metrics for Random Forest Regressor
performance_metrics(y_true,y_pred_RF)


######## MODEL SELECTION ############

# Based on the performance metrics Random Forest model is selected as it gives the best performance
#Now let us try to optimize it's result even more by hyperparameter tuning


######### MODEL OPTIMIZATION #############

plot(model_RF)


# We can see from the plot that the error is hugely minimized for ntree = 200 and after that adding
# trees doesn't effect the error much and as we already have by default 500 trees, and it is not 
# necessary that we add more tree for the sake of very minor reduction in error.
#So we will be keeping 500 tree while optimization


# We try to extract the best value for hyper parameter "mtry" using caret library's 'rf' method 
 #set.seed(seed)
 #control <- trainControl(method="repeatedcv", number=5, repeats=3, search="random")
 #tuner = train(total_count~.,data = train_set,method = "rf",metric = "RMSE",seed = seed,trControl= control)
 #tuner


#This random search gives result mtry = 4 as best value and ntree is set at by default value 500
set.seed(seed)
model_RF_tuned = randomForest(total_count~.,data = train_set,ntree=500,mtry=4)
summary(model_RF_tuned)
y_pred_RF_tuned = predict(model_RF_tuned,newdata = test_set[-7])
performance_metrics(y_true,y_pred_RF_tuned)
plot(model_RF_tuned)



bike_prediction = data.frame(t(y_true))
bike_prediction$Prediction = y_pred_RF_tuned
rownames(bike_prediction) = NULL

write.csv(bike_prediction,file = "submission_bike_R.csv")
