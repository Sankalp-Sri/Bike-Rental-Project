Problem Statement:
The objective of this Case is to Predication of bike rental count on daily basis on the
environmental and seasonal settings.

Data Set & Attributes:
We are provided with a dataset “day” which is in CSV format.
The day.csv file has the dimension of 731x16 i.e. 731 observations and 16 features/variables among which
13 are independent features or variable and 3 are target variable or dependent variable.

The details of data attributes in the dataset are as follows -
instant: Record index
dteday: Date
season: Season (1:springer, 2:summer, 3:fall, 4:winter)
yr: Year (0: 2011, 1:2012)
mnth: Month (1 to 12)
hr: Hour (0 to 23)
holiday: weather day is holiday or not (extracted from Holiday Schedule)
weekday: Day of the week
workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
weathersit: (extracted from Freemeteo)
1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp: Normalized temperature in Celsius. The values are derived via
(t-t_min)/(t_max-t_min),
t_min=-8, t_max=+39 (only in hourly scale)
atemp: Normalized feeling temperature in Celsius. The values are derived via
(t-t_min)/(t_maxt_
min), t_min=-16, t_max=+50 (only in hourly scale)
hum: Normalized humidity. The values are divided to 100 (max)
windspeed: Normalized wind speed. The values are divided to 67 (max)
casual: count of casual users
registered: count of registered users
cnt: count of total rental bikes including both casual and registered

Methodology:
Data Preprocessing & EDA:
Data preprocessing is a data mining technique which is used to transform the raw data in a
useful and efficient format which can be understood by the machine.
This Data Preprocessing technique can be implemented using various steps:
a) Examining the data
b) Data Cleaning
c) Data Transformation
d) Data Reduction

Modeling:

Model Selection:-
Now as we are done with all the pre-processing and feature engineering process the next task is to select a model which can perform well on the given data.
Our goal is to apply a statistical learning method to the training data in order to estimate the unknown function f. In other words, we want to find a function f such that 
Y ≈ ˆ f(X) for any observation (X, Y ). Broadly speaking, most statistical learning methods for this task can be characterized as either parametric or non-parametric.


Parametric Method: 
This method basically involves a two step model based approach.
i) In first step we make an assumption about the functional form or shape of function ‘ f ’. Like in case of Mulitple Linear Regression we make the assumption that the 
weighted sum independent variables (β0+β1X1+β2X2…βnXn) is linear function of dependent variable Y.
Now once we assumed the shape of function linear the problem of estimation of f is greatly simplified because now we need not estimate an entire ndimensional space instead
we only need to estimate the (n+1) co-efficients.

ii) After making an assumption about the shape of the function ‘ f ‘ we need a procedure which uses the training data to fit or train the model. Now in case of a linear model
one of the general procedure we use is the “least square method”

Non Parametric Method: 
Now as the performance of the model in parametric methods greatly rely on the assumptions that we have made about the shape of function ‘ f ‘, we may get a very bad result 
if the actual shape of function f varies a lot from the assumed shape. Thus in this scenario we need more flexible models that can fit many different functional shapes for 
function ‘ f ‘ . So the in Non Parametric Method, we find an estimate of ‘f’ that gets as close to the data point as possible and by avoiding the pre-assumption about 
functional shape this method has the potential to accurately fit a wider range of possible functional forms or shapes.
The disadvantage of such method is that we have to use entire n-dimensional functional space to estimate f. Thus a larger number of observations will be required
to accurately estimate the function ‘f’.
Before we proceed to the selection of relationship between It can be seen from the plots that the relationship between the predictors and the dependent variable is not exactly
linear as the data points are not aligned with the regression line much and they are scattered throughout the space.

Multiple Linear Regression:
Now at first we will go with the Parametric Approach and assume that the estimate function has a linear form. So we will use we will be estimating the (n+1) coefficient which 
has minimum least square error for the predicted variable.


Decision Tree:
The second approach will be a Non assuming a functional shape of our estimator i.e. our model will now have greater flexibility and can deal with variety of function modeling 
method for our problem, each independent and dependent variables. n Regression Multiple Linear Regression Model the Non-Parametric approach which lifts the restriction of shapes.




The Decision Tree Algorithm utilizes a Non-Parametric approach to estimate the
target variables. The tree based methods perform segmentation/stratification of
feature space into a number of simpler regions & in order to do the prediction we just
uses the mean or mode of that region in which a particular test observation lies
The goal in a regression tree is to find regions, which minimizes the RSS (residual
sum of squares) and we use the Recursive Binary Splitting (Greedy) approach to
create these regions.
We select a feature and a cut point p such that splitting of the predictor space results
into greatest reduction in RSS.

Random Forest:
The accuracy of the tree based model can be improved substantially by aggregating many decision trees by bagging and boosting methods.
Random forest model utilizes bagging method to aggregate many decision tree in order to reduce the variance.
In order to understand the random forest algorithm we first need to have a look at the concept of bagging.
Recall that given a set of n independent observations Z1, . . . , Zn, each with variance σ2, the variance of the mean ¯ Z of the observations is given by σ2/n. In other words,
averaging a set of observations reduces variance . Hence a natural way to reduce the variance and hence increase the prediction accuracy of a statistical learning method is to 
take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions.

In other words, we could calculate f1(x), f2(x), . . . , fB(x) using B separate training sets, and average them in order to obtain a single low-variance statistical learning
model. Of course, this is not practical because we generally do not have access to multiple training sets. Instead, we can bootstrap, by taking repeated samples from the (single)
training data set.

This concept is known as BAGGING.
Random forests provide an improvement over bagged trees by de-correlating the trees. As in bagging, we build a number forest of decision trees on bootstrapped training samples.
But when building these decision trees, each time a split in a tree isconsidered, a random sample of m predictors is chosen as split candidates from the full set of n predictors.
The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose m = sqrt(n) i.e., the number of 
predictors considered at each split is approximately equal to the square root of the total number of predictors.

This process induces more randomness in our model and it will help the model to perform better on the unseen data.

Conclusions:
3.1 Model Evaluation:
As we have decided which models to choose for our dataset we now discuss some performance metrics which are used in the regression problems and then we will select the metrics
which suits our problem statement.

i) R2 Score: It measures the proportion of variability in dependent variable explained by the independent variables so it can be defined as : 
R2 = Variance explained by the model/Total Variance
Thus its value ranges between 0 to 1. A Value of R2 close to 0 suggest that model fails to explain the variance in target variable with the help of predictors and a value close
to 1 suggests that model successfully  explains the variance in target variable with the help of predictors. That’s why it is also termed as “Goodness of Fit”

ii) Mean Absolute Error (MAE): MAE measures the average magnitude of the errors in a set of predictions, without considering their direction i.e. it takes the absolute values
of error nullifying the direction or error.

iii) Root Mean Squared Error (RMSE): RMSE measures the root mean of squared errors in a set of predictions. It squares the error of each prediction, takes the average of sum of
error squares and then takes the square root.
By squaring the error it basically increases the weight of higher error terms with respect to the lower error terms. Thus it can be useful when it is desired that there should
be no large errors in the dataset.

iv) Root Mean Squared Log Error(RMSLE): It takes the difference between log values of the prediction and actual test values and then square them.

The ideas behind taking log values is as follows:
a) It suppresses the outliers in prediction set by taking log values which can be highly noisy if previous metrics are used.
b) It basically provides us the relative error as according to log rule: log(m)-log(n) = log(m/n) The relative error thus obtained is free from the scale of prediction set.
c) It imposes a biased penalty on the underestimated and overestimated
errors. It imposes higher penalty for underestimated errors in the prediction set. Thus the use cases where the underestimation of target variable is not desired we should use 
this error metrics.

After discussing all these metrics we would consider R2 Score, RMSE & RMSLE for evaluation of our model.

Model Selection:
As it can be seen from the table presented in the previous chapter that Random Forest Regressor has R2 Score of 0.847(python) & 0.877(R) i.e. it explains 84.7%
(python) & 87.7% (R) variance in the dependent variable. It also has lowest RMSE and RMSLE errors. So for this particular problem statement we would choose the
Random Forest Regressor as our final model.

In our problem statement the estimation of bikes as per the supply vs. demand point of view is very important and thus the RMSLE metrics is very useful as it
punishes the underestimated count error more than the overestimated count.

Model Optimization:
So, as we have selected our final model for this particular problem statement, we will now focus on how we can further optimize the results we were getting from
the Random Forest model by default hyper parameters. Now we will be using the Random Search method in Python & R to tune the hyper parameters of the finalized model instead 
of Grid Search Method.

Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It tries random combinations of a range of 
values. To optimise with random search, the function is evaluated at some number of random configurations in the parameter space.

The chances of finding the optimal parameter are comparatively higher in random search because of the random search pattern where the model might end up being trained on the
optimised parameters without any aliasing. Random search works best for lower dimensional data since the time taken to find the right set is less with less number of iterations.

Model Hyper Parameters to be tuned Resulted values Python RandomizedSearch CV Number of trees to be used(n_estimators), maximum number of features to be considered while bagging
(max_features) n_estimator =400 max_features = “log2”

As can be seen from the table that after optimization the performance of our model increased by a fair margin.
Now we will again predict the target variable using this optimized model and finally store it in a submission file.
