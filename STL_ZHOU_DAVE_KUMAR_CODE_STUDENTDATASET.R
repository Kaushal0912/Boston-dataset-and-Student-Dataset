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
library(ISLR)
library(MASS)
library(tidyverse)
library(e1071)
library(caret)
library(sparseSVM)
library(car)

par(mar = rep(2,4))
set.seed(100)
#These are two files which have scores related to Maths and another
#portugese subject. there are 33 columns. The Dataset is about
#performance of multiple students and ability to predict absenteeism
file1 = read.csv('student-mat.csv',sep=';',header = TRUE)
file2 = read.csv('student-por.csv',sep=';',header = TRUE)

df_student <- rbind(file1,file2)
summary(df_student)

X = model.matrix(G3~.,df_student )[,-1] #remove the intercept
y = df_student$G3
df_converted <- bind_cols(data.frame(X),data.frame(y))
names(df_converted)[42] <- 'G3'

#we will check if the dataset requires any null value issues or not
#Number of 'NA' values are 0, which means data is perfect
sum(is.na(df_converted))


#creating a formula: "." represents all columns and G3 is the target
#lm is the linear model
#the adjusted rsquared value is 0.8349, which means 83% of the time
#the linearity is present
fmla_linear_allparam = as.formula(G3~.,df_converted)
linear.model = lm(fmla_linear_allparam,df_converted)
summary(linear.model)


#draws 4 diagnostic plots of data
plot(linear.model)
vif(linear.model)


#on seeing the standardised residuals Vs Leverage plot
# we find 3 outliers present at row number 342,265,335
#removing them should help improve the rsquared value
df_converted <- df_converted[c(-342,-335,-265),]
#we are removing the rows in descending order so that
#row index is not altered.
#adjusted rsquared has now increased from 0.83 to 0.8481

fmla_linear_allparam = as.formula(G3~.,df_converted)
linear.model = lm(fmla_linear_allparam,df_converted)
summary(linear.model)

#observation#277,276 have a high leverage value
which.max(hatvalues(linear.model))
#these rows follow the trend for data but have very high values
#for predictors.

#making a copy of df_converted
df_converted_hl <- df_converted
#so we delete the observation 277
df_converted_hl <- df_converted_hl[c(-277,-276),]
fmla_linear_allparam = as.formula(G3~.,df_converted_hl)
linear.model = lm(fmla_linear_allparam,df_converted_hl)
summary(linear.model)
#after removing the two leverage points, we see that
#there was hardly any improvement in the r-squared value
#so we let the two points be as is

#draws 4 diagnostic plots of data
plot(linear.model)

#a plot between one of the important predictors and target variable
par(mar = rep(2,4))
fmla_linear_absences = as.formula(G3~absences,df_converted)
linear.model = lm(fmla_linear_absences,df_converted)
plot(df_converted$absences,df_converted$G3,caption=c('a','b','c'))
abline(linear.model,lwd=2,col="blue")

#Multicollinearity between the variables
#below code show the correlation in much more easier way
#where user doesn't have to look at the graph, pretty handy
#when there are multiple variables.
cor_df_converted <- cor(df_converted)
for (row in 1:nrow(cor_df_converted)){
  for(col in 1:ncol(cor_df_converted)){
    if (cor_df_converted[row,col] != 1) {
      if (cor_df_converted[row,col] >= 0.7){
        #https://stackoverflow.com/questions/12464731/retrieve-row-and-column-name-of-particular-cell-in-r
        str <- paste(rownames(cor_df_converted)[row], colnames(cor_df_converted)[col],'High Correlation',cor_df_converted[row,col], sep=", ")
        print(str)
      } else if(cor_df_converted[row,col] > 0.3 & cor_df_converted[row,col]<= 0.69){
        #https://stackoverflow.com/questions/12464731/retrieve-row-and-column-name-of-particular-cell-in-r
        str <- paste(rownames(cor_df_converted)[row], colnames(cor_df_converted)[col],'Medium Correlation',cor_df_converted[row,col], sep=", ")
        print(str)
      }
    }
  }
}

#g2 and g3 have high correlation between each other
fmla_linear_one = as.formula(G3~ G2,df_converted)
linear.model = lm(fmla_linear_one,df_converted)
summary(linear.model)
#adjusted R-squared is 0.8435
coef(linear.model)

#g1 and g3 have high correlation between each other
fmla_linear_one = as.formula(G3~ G1,df_converted)
linear.model = lm(fmla_linear_one,df_converted)
summary(linear.model)
#adjusted R-squared is0.66
coef(linear.model)

#correlation of g2 and G1 were quite high, lets consider them indivdually 
fmla_linear_two = as.formula(G3~ G2 + G1,df_converted)
linear.model = lm(fmla_linear_two,df_converted)
summary(linear.model)
#adjusted R-squared is 0.846
coef(linear.model)

#But when we consider interactions of g2 and G1
fmla_linear_inter = as.formula(G3~G1*G2,df_converted)
linear.model = lm(fmla_linear_inter,df_converted)
summary(linear.model)
#adjusted R-squared is 0.8489
coef(linear.model)

#for us to be able to replicate the data in other systems or
# multiple runs, we avoid the randomness by setting the seed
# this will allow us to compare results after each run
set.seed(100)

#creating a 70-30 ratio of train and test data set
# [sample_data] means 70% and [-sample_data] means remaining
# 30% of data

sample_data = sample(nrow(X),round(nrow(X) * 0.7))

#Perform regression with regularization.
#Reasons that lead to regularization and inferences that
#can be drawn from this regularization


lambda_range = 2^seq(-5,10)
par(mar = rep(2,4))

#ridge regularisation alpha = 0
ridge.model <- glmnet(X[sample_data,],y[sample_data],alpha = 0 )
ridge.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=0,nfolds=10,lambda = lambda_range)
ridge.model_cv$lambda.min
plot(ridge.model_cv,sign.lambda=1)
plot(ridge.model)
ridge.model_pred <- predict(ridge.model_cv,s=ridge.model_cv$lambda.min,X[-sample_data,])
sse <- sum((ridge.model_pred - y[-sample_data])^2)
sst <- sum((ridge.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared = 0.754
coef(ridge.model_cv)


#lasso regression , alpha = 1
lasso.model <- glmnet(X[sample_data,],y[sample_data],alpha = 1 )
lasso.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=1,nfolds=10,lambda=lambda_range)
lasso.model_cv$lambda.min
plot(lasso.model_cv,sign.lambda=1)
plot(lasso.model)
lasso.model_pred <- predict(lasso.model_cv,s=lasso.model_cv$lambda.min,X[-sample_data,])
sse <- sum((lasso.model_pred - y[-sample_data])^2)
sst <- sum((lasso.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared is 0.747
coef(lasso.model_cv)


#elastic net, when alpha is neither 0 nor 1
elastic.model_lq <- glmnet(X[sample_data,],y[sample_data],alpha = 0.25)
elastic.model_med <- glmnet(X[sample_data,],y[sample_data],alpha = 0.50)
elastic.model_uq <- glmnet(X[sample_data,],y[sample_data],alpha = 0.75)
elastic.model_sm <- glmnet(X[sample_data,],y[sample_data],alpha = 0.01)
plot(elastic.model_lq,xvar="lambda")
plot(elastic.model_med,xvar="lambda")
plot(elastic.model_uq,xvar="lambda")
plot(elastic.model_sm,xvar='lambda')

#The best lambda from elastic regularization and alpha 0.5 is: ~2.320
elastic.model_cv <- cv.glmnet(X[sample_data,],y[sample_data],alpha=0.5,nfolds=10,lambda = lambda_range)
elastic.model_cv$lambda.min
plot(elastic.model_cv,sign.lambda=1)

#prediction that can be drawn from this best elastic model is: rsquared = 0.748
elastic.model_pred <- predict(elastic.model_cv,s=elastic.model_cv$lambda.min,X[-sample_data,])
sse <- sum((elastic.model_pred - y[-sample_data])^2)
sst <- sum((elastic.model_pred - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
#rsquared of 0.739
coef(elastic.model_cv)


#--------------------------------------------------------------------------------------------------
#SVM
#we are converting the G3 column from continuous variable to
# categorical variable. So any grade >=10 is considered to be pass
# and anything less than that is fail
df_converted_1 <- df_converted %>%
  filter( G3 >= 10 ) %>%
  mutate( G3 = 1)

df_converted_0 <- df_converted %>%
  filter(G3 < 10) %>%
  mutate(G3 = 0)

#combining rows of pass and fail under one dataframe
df_converted_svm = bind_rows(df_converted_1,df_converted_0)
df_converted_svm$G3 <- as.factor(df_converted_svm$G3)

#clearing the variables again
X <- 0
y <- 0

X <- model.matrix(G3~.,df_converted_svm)[,-1] # -1 removes the intercept column
y <- df_converted_svm$G3

sample_data = sample(nrow(X),round(nrow(X) * 0.7))

#linear kernel :number of support vectors 117, cost = 10; Accuracy = 92.63
svm_model_1 = svm(X[sample_data,],y[sample_data],kernel="linear",cost=10,scale=TRUE)
summary(svm_model_1)
y_pred_svm = predict(svm_model_1,X[-sample_data,])
confusionMatrix(y_pred_svm,y[-sample_data])


#radial kernel :Number of support vectors 286, cost = 10; Accuracy = 89.74
svm_model_1 = svm(X[sample_data,],y[sample_data],kernel="radial",cost=10,scale=TRUE)
summary(svm_model_1)
y_pred_svm = predict(svm_model_1,X[-sample_data,])
confusionMatrix(y_pred_svm,y[-sample_data])


#we can keep changing the value of kernel argument to linear,radial etc
tuned_model_1 <- tune(svm,X[sample_data,],y[sample_data],kernel="radial",ranges =list(cost=c(10^seq(-3,3)) ),scale=TRUE,cross=10)
summary(tuned_model_1$best.model)
#linear kernel(best model) : number of support vectors 170, cost = 0.1,cross=10
#radial kernel(best model) : number of support vectors 191, cost = 1000,cross=10
#polynomial kernel(best model) : number of support vectors 318, cost = 1000,cross=10
#sigmoid kernel(best model) : number of support vectors 214,cost = 1,cross = 10

y_pred_svm = predict(tuned_model_1$best.model,X[-sample_data,])
confusionMatrix(y_pred_svm,y[-sample_data])
#linear model tuned model : Accuracy = 91.67
#radial kernel tuned model : Accuracy = 91.35
#polynomial kernel tuned model: Accuracy = 86.41
#sigmoid kernel tuned model : Accuracy = 90.06

#However we can make the model clearer by doing some regularisation on the data.
#install.packages('sparseSVM')
library(sparseSVM)

#for sparsesvm with cv with l2 regression
svm_model_ridge = cv.sparseSVM(X[sample_data,],y[sample_data],seed = 100)
y_pred_svm_ridge = predict(svm_model_ridge,X[-sample_data,],lambda=svm_model_ridge$lambda.min)
y_pred_svm_ridge <- as.factor(y_pred_svm_ridge)
svm_model_ridge$lambda.min
confusionMatrix(y_pred_svm_ridge,y[-sample_data])
# accuracy 82.05%
coef(svm_model_ridge)

#for lasso i.e. alpha = 1
svm_model_lasso <- cv.sparseSVM(X[sample_data,],y[sample_data],alpha=1,seed = 100)
y_pred_svm_lasso <- predict(svm_model_lasso,X[-sample_data,],lambda = svm_model_lasso$lambda.min)
y_pred_svm_lasso <- as.factor(y_pred_svm_lasso)
confusionMatrix(y_pred_svm_lasso,y[-sample_data])
# accuracy 82.05%
coef(svm_model_lasso)

#for elastic i.e. alpha = 0 > & <1 
svm_model_elastic <- cv.sparseSVM(X[sample_data,],y[sample_data],alpha=0.5,seed = 100)
y_pred_svm_elastic <- predict(svm_model_elastic,X[-sample_data,],lambda = svm_model_elastic$lambda.min)
y_pred_svm_elastic <- as.factor(y_pred_svm_elastic)
confusionMatrix(y_pred_svm_elastic,y[-sample_data])
#accuracy 82.05% for alpha = 0.5
coef(svm_model_elastic)

par(mar = rep(2,4))
plot(svm_model_lasso)
plot(svm_model_ridge)
plot(svm_model_elastic)

library('ROCR')
pred_1 <- prediction(as.numeric(y_pred_svm_ridge),as.numeric(y[-sample_data]))
pred_2 <- prediction(as.numeric(y_pred_svm_lasso),as.numeric(y[-sample_data]))
pred_3 <- prediction(as.numeric(y_pred_svm_elastic),as.numeric(y[-sample_data]))

perf_1 <- performance(pred_1,"tpr","fpr")
perf_2 <- performance(pred_2,"tpr","fpr")
perf_3 <- performance(pred_3,"tpr","fpr")

plot(perf_1,col=rainbow(10))
abline(a=0, b= 1)
plot(perf_2,col=rainbow(10))
abline(a=0, b= 1)
plot(perf_3,col=rainbow(10))
abline(a=0, b= 1)

#install.packages('neuralnet')
#install.packages('ANN2')

set.seed(100)
library(ANN2)
library(neuralnet)
mins <- apply(df_converted,2,min)
maxs <- apply(df_converted,2,max)
scaleddata <- scale(df_converted,center = mins,scale = maxs - mins)

scaleddata_df <- data.frame(scaleddata)

X <- model.matrix(G3~.,scaleddata_df)[,-1] # -1 removes the intercept column
y <- scaleddata_df$G3

nn_model1 = neuralnet(G3~.,scaleddata[sample_data,],hidden = c(1,3),linear.output = TRUE)
plot(nn_model1)
y_pred_nn_1 <- neuralnet::compute(nn_model1,scaleddata[-sample_data,])
true_y_pred_nn_1 <- (y_pred_nn_1$net.result * (max(scaleddata_df$G3) - min(scaleddata_df$G3))) + min(scaleddata_df$G3)
sse <- sum((true_y_pred_nn_1 - scaleddata_df$G3[-sample_data])^2)
sst <- sum((true_y_pred_nn_1 - mean(scaleddata_df$G3[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
coef(nn_model1)

nn_model2 = neuralnetwork(X[sample_data,],y[sample_data],learn.rates = 0.001,hidden.layers = c(1,3),verbose = FALSE,regression = TRUE,loss.type = 'squared',L1=1,L2=1,random.seed = 100)
plot(nn_model2)
y_pred_nn_2 <- predict(nn_model2,X[-sample_data,])
true_y_pred_nn_2 <- (as.numeric(y_pred_nn_2$predictions) * (max(scaleddata_df$G3) - min(scaleddata_df$G3))) + min(scaleddata_df$G3)
sse <- sum((true_y_pred_nn_2 - y[-sample_data])^2)
sst <- sum((true_y_pred_nn_2 - mean(y[-sample_data]))^2)
rsquared <- 1 - (sse/sst)
rsquared
coef(nn_model2)


