library(ranger)  # A fast implementation of Random Forests, particularly suited for high dimensional data. Ensembles of classification, regression, survival and probability prediction trees are supported. 
library(caret)  # The caret package (short for Classification And REgression Training) contains functions to streamline the model training process for complex regression
library(data.table)  # Fast aggregation of large data (e.g. 100GB in RAM), fast ordered joins, fast add/modify/delete of col
library(pROC)
creditcard_data <- read.csv("D:/SourceCodeforMSProjects/R-Project/Credit Card Fraud Detection/creditcard.csv")
dim(creditcard_data)
head(creditcard_data,6)
tail(creditcard_data,6)
table(creditcard_data$Class)
summary(creditcard_data$Amount)
names(creditcard_data)
var(creditcard_data$Amount)
sd(creditcard_data$Amount)
head(creditcard_data)
# Assuming 'creditcard_data' is your dataset
creditcard_data$Amount <- scale(creditcard_data$Amount)
NewData=creditcard_data[,-c(1)]
head(NewData)
library(caTools)  # Moving Window Statistics, GIF, Base64, ROC AUC, etc
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)
Logistic_Model = glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)                     
plot(Logistic_Model)

Logistic_Model = glm(Class~.,train_data,family=binomial())
summary(Logistic_Model)
library(pROC)  # display and analyze ROC curves
lr.predict <- predict(Logistic_Model,test_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")

library(rpart)  # Recursive partitioning for classification, regression and survival trees.
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , creditcard_data, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard_data, type = 'class')
probability <- predict(decisionTree_model, creditcard_data, type = 'prob')

rpart.plot(decisionTree_model)
library(neuralnet) # nn for computation of a given neural network for given covariate vectors (formerly compute)
ANN_model <- neuralnet(Class~.,data=train_data,linear.output=F)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)
library(gbm, quietly=TRUE)  # An implementation of extensions to Freund and Schapire's AdaBoost algorithm and Friedman's gradient boosting machine.

# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)

# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)

plot(model_gbm)
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)
                     
                     
