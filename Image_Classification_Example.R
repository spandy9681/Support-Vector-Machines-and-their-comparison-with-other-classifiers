library(keras)
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Determining train and test subdata sizes
trn.size = 2*10^3;tst.size = 5*10^2
# Creating train data in proper format

# Training subdata
train.images <- matrix(nrow = trn.size,ncol = ncol(train_images)^2)
ind.trn <- sample(1:nrow(train_images),size = trn.size)
for(i in 1:trn.size)
{
  v = as.vector(train_images[ind.trn[i],,])
  train.images[i,] = v
}
train.labels <- train_labels[ind.trn]
table(train.labels)

# Test subdata
test.images <- matrix(nrow = tst.size,ncol = ncol(test_images)^2)
ind.tst <- sample(1:nrow(test_images),size = tst.size)
# Training subdata
for(i in 1:tst.size)
{
  v = as.vector(test_images[ind.tst[i],,])
  test.images[i,] = v
}
test.labels <- test_labels[ind.tst]
table(test.labels)

# Defining the datas properly
Train.Data = data.frame("Y" = as.factor(train.labels),train.images/255)
Test.Data = data.frame("Y" = as.factor(test.labels),test.images/255)

# Running the SVM example
library(e1071)
svmfit = svm(Y~., data = Train.Data[,-2], kernel = "radial", cost = 10^10)

# Training Error
p = predict(svmfit, Train.Data[,-1])
(T = table(p ,train.labels))
1-sum(diag(T))/sum(T)

# Test Error
p = predict(svmfit, Test.Data[,-1])
(T = table(p ,test.labels))
1-sum(diag(T))/sum(T)

