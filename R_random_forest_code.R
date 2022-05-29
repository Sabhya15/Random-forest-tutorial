##### Set Up #####

# install.packages("haven")
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("caTools")
# install.packages("datasets")


library(haven)
library(randomForest)
library(dplyr)
library(caTools)
library(datasets)

##### Classification #####

data(iris)
summary(iris)
names(iris) <- tolower(names(iris))

## Split into training and validation 

set.seed(1234)

iris$sample = sample.split(iris$species, SplitRatio = .8)
training <- subset(iris, iris$sample == TRUE)
validation <-subset(iris, iris$sample== FALSE)
training$sample = NULL                                                          #removing the sample variable
validation$sample = NULL 

rf_classification <- randomForest(species ~ ., data=training, ntree=500, 
                                  type=classification, mtry=2, importance=TRUE)

# Species ~. : We will try to predict the species using all the other variables
# Type: specifies that we are performing a classification. Others are regression, or unsupervised.
# Mtry: Number of variables to try for each tree
# ntree: Number of trees
# importance: INstruct it to calculate variable importance

rf_classification

# OOB - At each step, a sub-sample is taken (bootstrapped) from the training sample to make a tree
# the out-of-the bag rate is the error rate in the rest of the sample (training sample - bootstrapped sample)

# The Confusion matrix: each row is the observed value and the columns shows the predicted value. 
# For example, 37 actual versicolor flowers are classified correctly and 3 actual versicolor are classified as virginica
# This gives a class error rate for versicolor of 0.075 

varImpPlot(rf_classification)

# MeanDecreaseAccuracy is the decrease in prediction performance when that particular
# variable is omitted for estimating a tree. High values mean that the var increases accuracy

# MeanDecreaseGini: Gini is the measure of "node impurity". If you use this feature to split 
# the data, how "pure" will the nodes be. High purity means that a node only contains one class (say only versicolor)
# The decrease in Gini shows how much the Gini decreases when the variable is included. A high value = good

## Predicting on the validation dataset

prediction <- predict(rf_classification,validation[,-5])                        #the model and the independent variables
table(observed=validation[,5],predicted=prediction)
sum(prediction!=validation[,5])/nrow(validation[,5])                            #validation error

##### Tuning the model ##### 

## Number of trees in the forest

# We can plot the out-of-the bag error by number of trees
plot(rf_classification)

# This shows that the out-of-the bag error (black) and the class error per class

# The error is stable after after around 200 trees


## Number of variables to try at each tree

# we estimate the random forest with different values of m

# M is the number of variables the algorithm will try to use for each tree
set.seed(101)
rf_tree_1 = randomForest(species ~ ., data=training, ntree=250, 
                         type=classification, mtry=1, importance=TRUE)
  
set.seed(101)
rf_tree_2 = randomForest(species ~ ., data=training, ntree=250, 
                         type=classification, mtry=2, importance=TRUE)

set.seed(101)
rf_tree_3 = randomForest(species ~ ., data=training, ntree=250, 
                         type=classification, mtry=3, importance=TRUE)

set.seed(101)
rf_tree_4 = randomForest(species ~ ., data=training, ntree=250, 
                         type=classification, mtry=4, importance=TRUE)

plot(1:250, rf_tree_1$err.rate[,1], type = "l", col="orange", xlab = "Number of Trees",
     ylab = "Out of bag error", ylim = c(0,0.1))
lines(1:250, rf_tree_2$err.rate[,1], col = "red", type = "l")
lines(1:250, rf_tree_3$err.rate[,1], col = "blue", type = "l")
lines(1:250, rf_tree_4$err.rate[,1], col = "green", type = "l")
legend("topright", c("m=1", "m=2", "m=3", "m=4"), col = c("orange", "green", "red", "blue"))


# The error seems to be lowest for m = 2 and m = 3. We prefer to use the lowest possible 
# m with a low error to decrease overfitting

# We can also use other measures (AUC for example) to tune our model




##### Resources used #####

# https://www.blopig.com/blog/2017/04/a-very-basic-introduction-to-random-forests-using-r/ 
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d