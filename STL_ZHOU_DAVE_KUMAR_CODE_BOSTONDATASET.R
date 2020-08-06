#if the below packages are not installed, please uncomment 
#and install them

#install.packages('ISLR')
#install.packages('MASS')
#install.packages('car')
#install.packages('tidyverse')
#install.packages('e1071')
#install.packages('caret')
#install.packages('sparseSVM')

#used for regularization
library(glmnet) 
#used to import datasets
library(ISLR)
library(MASS)
library(tidyverse)
library(car)

set.seed(100)
par(mar = rep(2,4))
df_Boston <- Boston
#the Boston dataset has many variables, we will select
#crim as the target variable.We will check if the data
#requires any preprocessing due to missing values
#and create a formula where
# target variable has relation with all variables. 
sum(is.na(df_Boston))
#Number of 'NA' values are 0, which means data is perfect

#creating a formula: "." represents all columns and crim is the target
fmla_linear_allparam = as.formula(crim~.,df_Boston)
#lm is the linear model
linear.model = lm(fmla_linear_allparam,df_Boston)
plot(linear.model)

#To understand the significance of each predictor
#the null hypothesis can be rejected or passed based on '*'
#present on each of the predictor
summary(linear.model)
#adjusted Rsquared is 0.4396


#on seeing the standardised residuals Vs Leverage plot
# we find 3 outliers present at row number 419,406,381
#removing them should help improve the rsquared value
df_Boston <- df_Boston[c(-419,-406,-381),]
#we are removing the rows in descending order so that
#row index is not altered.
fmla_linear_allparam = as.formula(crim~.,df_Boston)
linear.model = lm(fmla_linear_allparam,df_Boston)
summary(linear.model)
#rsquared is 0.5992 
plot(linear.model)
#the null hypothesis can be rejected for intercept,zn,nox,dis,rad,black,lstat


#adjusted rsquared(Ordinary least square) has now increased to 0.59

#system returns high leverage value
which.max(hatvalues(linear.model))
#these rows follow the trend for data but have very high values
#for predictors.

#making a copy of df_Boston to delete and see and the impact
df_Boston_hl <- df_Boston
df_Boston_hl <- df_Boston_hl[-369,]
fmla_linear_allparam = as.formula(crim~.,df_Boston_hl)
linear.model = lm(fmla_linear_allparam,df_Boston_hl)
summary(linear.model)
#But not all high leverage values are influential
#we can see the rsquared value not moving by much if we exclude
#the data point.
plot(linear.model)
#so we continue keeping the two data points.

#using VIF to see which if variation could be a problem in the dataset
#values <= 5 signify of stability in the model
vif(linear.model)
#tax and rad have a high correlation as seen from vif

#In Boston Dataset we have the highest correlation between tax and rad
# We will pick those and check the relation with target variable
# both : independent and interaction
cor_df_Boston <- cor(df_Boston)
for (row in 1:nrow(cor_df_Boston)){
  for(col in 1:ncol(cor_df_Boston)){
    if (cor_df_Boston[row,col] != 1) {
      if (cor_df_Boston[row,col] >= 0.7){
        #https://stackoverflow.com/questions/12464731/retrieve-row-and-column-name-of-particular-cell-in-r
        str <- paste(rownames(cor_df_Boston)[row], colnames(cor_df_Boston)[col],'High Correlation',cor_df_Boston[row,col], sep=", ")
        print(str)
      } else if(cor_df_Boston[row,col] > 0.3 & cor_df_Boston[row,col]<= 0.69){
        #https://stackoverflow.com/questions/12464731/retrieve-row-and-column-name-of-particular-cell-in-r
        str <- paste(rownames(cor_df_Boston)[row], colnames(cor_df_Boston)[col],'Medium Correlation',cor_df_Boston[row,col], sep=", ")
        print(str)
      }
    }
  }
}

#rad and tax have high collinearity
fmla_linear_two = as.formula(crim~rad + tax,df_Boston)
linear.model = lm(fmla_linear_two,df_Boston)
summary(linear.model)
#adjusted R-squared is 0.5305
coef(linear.model)

#But when we consider interactions
fmla_linear_inter = as.formula(crim~rad*tax,df_Boston)
linear.model = lm(fmla_linear_inter,df_Boston)
summary(linear.model)
#adjusted R-squared is 0.5402
coef(linear.model)


#When considering only tax
fmla_linear_tax = as.formula(crim~tax,df_Boston)
linear.model = lm(fmla_linear_tax,df_Boston)
summary(linear.model)
#adjusted R-squared is 0.46
coef(linear.model)

#When considering only rad
fmla_linear_rad = as.formula(crim~rad,df_Boston)
linear.model = lm(fmla_linear_rad,df_Boston)
summary(linear.model)
#adjusted R-squared is 0.529
coef(linear.model)

#divide the dataset into training and test set 70-30 ratio
X = model.matrix(crim~.,df_Boston)[,-1] # -1 removes the intercept column
y = df_Boston$crim
sample_data = sample(nrow(X),round(nrow(X) * 0.7))


#Perform regression with regularization.

#creating a range of lambda's
lambda_range = 2^seq(-5,20)
par(mar = rep(2,4))

set.seed(100)

#ridge regularisation alpha = 0
ridge.model <- glmnet(X[sample_data,],y[sample_data],alpha = 0 )
ridge.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=0,nfolds=10,lambda = lambda_range)
ridge.model_cv$lambda.min
plot(ridge.model_cv)
plot(ridge.model,xvar="lambda")
ridge.model_pred <- predict(ridge.model_cv,s=ridge.model_cv$lambda.min,X[-sample_data,])
sse <- sum((ridge.model_pred - y[-sample_data])^2)
sst <- sum((ridge.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared is 0.73
coef(ridge.model_cv)

#lasso regression , alpha = 1
lasso.model <- glmnet(X[sample_data,],y[sample_data],alpha = 1 )
lasso.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=1,nfolds=10,lambda=lambda_range)
lasso.model_cv$lambda.min
plot(lasso.model,xvar="lambda")
plot(lasso.model_cv)
lasso.model_pred <- predict(lasso.model_cv,s=lasso.model_cv$lambda.min,X[-sample_data,])
sse <- sum((lasso.model_pred - y[-sample_data])^2)
sst <- sum((lasso.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared is 0.7364
coef(lasso.model_cv)


#Elastic net
elastic.model_lq <- glmnet(X[sample_data,],y[sample_data],alpha = 0.25)
elastic.model_med <- glmnet(X[sample_data,],y[sample_data],alpha = 0.50)
elastic.model_uq <- glmnet(X[sample_data,],y[sample_data],alpha = 0.75)
elastic.model_sm <- glmnet(X[sample_data,],y[sample_data],alpha = 0.01)
plot(elastic.model_lq,xvar="lambda")
plot(elastic.model_med,xvar="lambda")
plot(elastic.model_uq,xvar="lambda")
plot(elastic.model_sm,xvar='lambda')


#The best model from regularization
elastic.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=0.5,nfolds=10)
min(elastic.model_cv$cvm)
elastic.model_pred <- predict(elastic.model_cv,s=elastic.model_cv$lambda.min,X[-sample_data,])
elastic.model_cv$lambda.min
sse <- sum((elastic.model_pred - y[-sample_data])^2)
sst <- sum((elastic.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared 0.7295
plot(elastic.model_cv)
coef(elastic.model_cv)

#SVM on boston dataset
#install.packages('e1071')
#install.packages('caret')
library(e1071)
library(caret)
library(tidyverse)

set.seed(100)

df_boston_1 <- df_Boston %>%
  filter( crim >= mean(crim) ) %>%
  mutate( crim = 1)

df_boston_0 <- df_Boston %>%
  filter( crim < mean(crim)) %>%
  mutate(crim = 0)

df_boston_svm <- bind_rows(df_boston_1,df_boston_0)
df_boston_svm$crim <- as.factor(df_boston_svm$crim)

X <- 0
y <- 0

X <- model.matrix(crim~.,df_boston_svm)[,-1] # -1 removes the intercept column
y <- df_boston_svm$crim

svm_model_1 = svm(X[sample_data,],y[sample_data],kernel="linear",cost=1000,scale=TRUE)
summary(svm_model_1)
y_pred_svm = predict(svm_model_1,X[-sample_data,])
confusionMatrix(y_pred_svm,y[-sample_data])
#number of support vectors 12; cost = 1000; accuracy 98.01%

tuned_model_1 <- tune(svm,X[sample_data,],y[sample_data],kernel="radial",ranges =list(cost=c(10^seq(-3,3)) ),scale=TRUE)
summary(tuned_model_1)
y_pred_svm = predict(tuned_model_1$best.model,X[-sample_data,])
#radial kernel , bestcost = 1, best performance 0.014

confusionMatrix(y_pred_svm,y[-sample_data])
# accuracy = 99.34
#correct classification = 150; misclassification = type1 0 + type2 1 = 1


svm_model_2 = svm(X[sample_data,],y[sample_data],kernel="linear",cost=1e-9,scale=TRUE)
summary(svm_model_2)
y_pred_svm_subopti = predict(svm_model_2,X[-sample_data,])
confusionMatrix(y_pred_svm_subopti,y[-sample_data])
#number of SV = 174, cost = 1e-9
#we know that cost = 1 is the best model for the given parameter setting
#but on checking with a massively low value of cost = 1e-9 we find that 
# accuracy has dipped to 72.19%, which means the model has not been penalised
# and misclassifications have risen.


#However we can make the model clearer by doing some regularisation on the data.
#install.packages('sparseSVM')
library(sparseSVM)
set.seed(100)
#for sparsesvm with cv with l2 regression
svm_model_ridge = cv.sparseSVM(X[sample_data,],y[sample_data],seed = 100)
y_pred_svm_ridge = predict(svm_model_ridge,X[-sample_data,],lambda=svm_model_ridge$lambda.min)
y_pred_svm_ridge <- as.factor(y_pred_svm_ridge)
confusionMatrix(y_pred_svm_ridge,y[-sample_data])
svm_model_ridge$lambda.min
#accuracy 97.35
coef(svm_model_ridge)

#for lasso i.e. alpha = 1
svm_model_lasso <- cv.sparseSVM(X[sample_data,],y[sample_data],alpha=1,seed = 100)
y_pred_svm_lasso <- predict(svm_model_lasso,X[-sample_data,],lambda = svm_model_lasso$lambda.min)
y_pred_svm_lasso <- as.factor(y_pred_svm_lasso)
confusionMatrix(y_pred_svm_lasso,y[-sample_data])
svm_model_lasso$lambda.min
#accuracy 97.35
coef(svm_model_lasso)

#for elastic i.e. alpha = 0 > & <1 
svm_model_elastic <- cv.sparseSVM(X[sample_data,],y[sample_data],alpha=0.5,seed = 100)
summary(svm_model_elastic)
y_pred_svm_elastic <- predict(svm_model_elastic,X[-sample_data,],lambda = svm_model_elastic$lambda.min)
y_pred_svm_elastic <- as.factor(y_pred_svm_elastic)
confusionMatrix(y_pred_svm_elastic,y[-sample_data])
#accuracy = 99.34
coef(svm_model_elastic)

#elastic net performed worse as compared to other regression models
# ridge and lasso performed same on Boston dataset

par(mar = rep(2,4))
plot(svm_model_lasso)
plot(svm_model_ridge)
plot(svm_model_elastic)

#this library is used for plotting AUC curve
library('ROCR')
pred_1 <- prediction(as.numeric(y_pred_svm_ridge),as.numeric(y[-sample_data]))
pred_2 <- prediction(as.numeric(y_pred_svm_lasso),as.numeric(y[-sample_data]))
pred_3 <- prediction(as.numeric(y_pred_svm_elastic),as.numeric(y[-sample_data]))

#finding the true positive and false positive rates
#of all 3 regularisation techniques
perf_1 <- performance(pred_1,"tpr","fpr")
perf_2 <- performance(pred_2,"tpr","fpr")
perf_3 <- performance(pred_3,"tpr","fpr")

#to plot AUC-ROC curve
plot(perf_1,col=rainbow(10))
abline(a=0, b= 1)
plot(perf_2,col=rainbow(10))
abline(a=0, b= 1)
plot(perf_3,col=rainbow(10))
abline(a=0, b= 1)

#install.packages('neuralnet')
#install.packages('ANN2')
library(ANN2)
library(neuralnet)
set.seed(100)
mins <- apply(df_Boston,2,min)
maxs <- apply(df_Boston,2,max)

#scaling all the data of the columns
scaleddata <- scale(df_Boston,center = mins,scale = maxs - mins)

#saving in a dataframe
scaleddata_df <- data.frame(scaleddata)

X <- model.matrix(crim~.,scaleddata_df)[,-1] # -1 removes the intercept column
y <- scaleddata_df$crim

nn_model1 = neuralnet(crim~.,scaleddata[sample_data,],hidden = c(1,2),linear.output = TRUE)
plot(nn_model1)

#just like predict function in other algorithms, we use compute for neuralnets
y_pred_nn_1 <- neuralnet::compute(nn_model1,scaleddata[-sample_data,])

#unscaling the data
true_y_pred_nn_1 <- (y_pred_nn_1$net.result * (max(df_Boston$crim) - min(df_Boston$crim))) + min(df_Boston$crim)

sse <- sum((true_y_pred_nn_1 - df_Boston$crim[-sample_data])^2)
sst <- sum((true_y_pred_nn_1 - mean(df_Boston$crim[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared value of 0.72

nn_model2 = neuralnetwork(X[sample_data,],y[sample_data],learn.rates=0.009,hidden.layers = c(1,2),verbose = FALSE,L1=1e-4,L2=1e-1,regression = TRUE,loss.type = 'squared',random.seed = 100)
y_pred_nn_2 <- predict(nn_model2,X[-sample_data,])

#unscaling back the target variable
true_y_pred_nn_2 <- (as.numeric(y_pred_nn_2$predictions) * (max(scaleddata_df$crim) - min(scaleddata_df$crim))) + min(scaleddata_df$crim)
sse <- sum((true_y_pred_nn_2 - y[-sample_data])^2)
sst <- sum((true_y_pred_nn_2 - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared value 0.81
plot(nn_model2)
coef(nn_model2)

