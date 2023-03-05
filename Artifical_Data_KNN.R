library(lattice)
library(rpart) # used for classification trees
library(e1071) # SVM
library(MASS)
library(package = "class")
library(mvtnorm)

# Data Generation
#n = 100
#r = 1 + rnorm(n,0,0.5)
#theta = runif(n,0,2*pi)
#x1 = r*cos(theta);y1 = r*sin(theta)
#r = 0 + rnorm(n,0,0.5)
#theta = runif(n,0,2*pi)
#x2 = r*cos(theta);y2 = r*sin(theta)
#X_dat <- data.frame(X = c(x1,x2),Y = c(y1,y2),label = rep(c(1,0),each = n))
#plot(Y~X,data = X_dat,col = as.factor(label),pch = 20)
dim(X_dat)
px1 = do.breaks(range(X_dat$X),brks)
px2 = do.breaks(range(X_dat$Y),brks)


# Data 2
#d = 2
#x1 <- mvrnorm(n,mu = rep(0,d),Sigma = diag(rep(1,d)))
#x2 <- mvrnorm(n,mu = rep(2,d),Sigma = diag(rep(1,d)))
#X_dat2 <- data.frame(rbind(x1,x2),label = rep(c(1,0),each = n))
names(X_dat2) <- c("X","Y","label")
head(X_dat2)
plot(Y~X,data = X_dat2,col = as.factor(X_dat2$label),pch = 20)

####X_dat = X_dat2
X_backup = X_dat

###KNN
#Breaking into equispaced grids
brks = 100
Kn = 20
g <- with(X_dat,
          expand.grid(X = do.breaks(range(X), brks),
                      Y = do.breaks(range(Y), brks)))

g$pred.knn <- knn(train = X_dat[, c("X", "Y")],
              test = g[, c("X", "Y")],
              cl = as.factor(X_dat$label), k = Kn)




M = as.factor(g$pred.knn)
M = as.integer(M)
M = matrix(M,nrow = brks+1)

# Plotting the decision boundaries

contour(px1, px2, M, labels="", xlab="", ylab="",axes=FALSE, levels = c(1:3),method = "simple",lwd = 1.5)
points(g, pch=".", cex=1.2, col=as.factor(g$pred.knn))
points(Y ~ X, data = X_dat,
       col = as.factor(X_dat$label),
       pch = 20, cex = 1)
box()
title(main = bquote("K = " ~.(Kn)),xlab = "X1",ylab = "X2")

# Test error of the knn classifier

ord = sample(1:100,100)
train = ord[1:90]
test = ord[91:100]

pred = knn(train = X_dat[train, c("X", "Y")],
    test = g[test, c("X", "Y")],
    cl = as.factor(X_dat[train,3]), k = 30)
10-sum(diag(table(pred,X_dat[test,3])))

#LDA
fm1 <- lda(label ~ X+Y, X_dat)
g$pred.lda <- predict(fm1, newdata = g)$class

M = as.factor(g$pred.lda)
M = as.integer(M)
M = matrix(M,nrow = brks+1)

# Plotting the decision boundaries

contour(px1, px2, M, labels="", xlab="", ylab="",axes=FALSE, levels = c(1:3),method = "simple",lwd = 1.5)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=as.factor(g$pred.lda))
points(Y ~ X, data = X_dat,
       col = as.factor(X_dat$label),
       pch = 20, cex = 1)
box()
table(predict(fm1,newdata = X_dat[,c("X","Y")])$class,X_dat$label)

###QDA

fm2 <- qda(label ~ X+Y, X_dat)
g$pred.qda <- predict(fm2, newdata = g)$class

M = as.factor(g$pred.qda)
M = as.integer(M)
M = matrix(M,nrow = brks+1)

# Plotting the decision boundaries

contour(px1, px2, M, labels="", xlab="", ylab="",axes=FALSE, levels = c(1:3),method = "simple",lwd = 1.5)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=as.factor(g$pred.qda))
points(Y ~ X, data = X_dat,
       col = as.factor(X_dat$label),
       pch = 20, cex = 1)
box()
table(predict(fm1,newdata = X_dat[,c("X","Y")])$class,X_dat$label)



###SVM


fm.svm <-
  svm(label ~ X + Y, data = X_dat,
      type = "C-classification",
      kernel = "linear", 
      cost = 10^4)
g$pred.svm <- predict(fm.svm, newdata = g)

M = as.factor(g$pred.svm)
M = as.integer(M)
M = matrix(M,nrow = brks+1)

# Plotting the decision boundaries

contour(px1, px2, M, labels="", xlab="", ylab="",axes=FALSE, levels = c(1:3),method = "simple",lwd = 1.5)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=as.factor(g$pred.svm))
points(Y ~ X, data = X_dat,
       col = as.factor(X_dat$label),
       pch = 20, cex = 1)
box()
table(predict(fm.svm,newdata = X_dat[,c("X","Y")]),X_dat$label)

tune.out <- tune(svm, label ~ X + Y, data = X_dat,
                 kernel = "polynomial",
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   degree = c(1, 2, 3, 4)
                 )
)
summary(tune.out)
tune.out$best.parameters
tune.out$best.performance

## Using polynomial kernel
fm.svm <-
  svm(label ~ X + Y, data = X_dat,
      type = "C-classification",
      kernel = "polynomial", 
      cost = 10)
g$pred.svm <- predict(fm.svm, newdata = g)

M = as.factor(g$pred.svm)
M = as.integer(M)
M = matrix(M,nrow = brks+1)

# Plotting the decision boundaries

contour(px1, px2, M, labels="", xlab="", ylab="",axes=FALSE, levels = c(1:2),method = "simple",lwd = 1.5)
gd <- expand.grid(x=px1, y=px2)
points(gd, pch=".", cex=1.2, col=as.factor(g$pred.svm))
points(Y ~ X, data = X_dat,
       col = as.factor(X_dat$label),
       pch = 20, cex = 1)
box()
table(predict(fm.svm,newdata = X_dat[,c("X","Y")]),X_dat$label)

tune.out <- tune(svm, label ~ X + Y, data = X_dat,
                 kernel = "polynomial",
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000)
                 )
)
summary(tune.out)
tune.out$best.parameters
tune.out$best.performance


## Trying out on Mobile price prediction dataset

dat <- read.csv("E:\\Dekstop\\ISI_Class_Files\\Second Semester\\Multivariate Data Analysis\\Project\\Mobile_Price_train.csv")
head(dat)
dim(dat)
names(dat)
dat1 = dat[,-c(2,4,6,18,19,20)]
dat1 = dat[,-c(2,4,6,18,19,20)]
dat1$price_range <- as.factor(dat1$price_range)
train = sample(2000,1500,replace = FALSE)
dat_train = dat1[train,]
dat_test = dat1[-train,]
dim(dat_train);dim(dat_test)

##LDA
mod.lda <- lda(price_range ~ .,data = dat_train)
pred.lda <- predict(mod.lda,newdata = dat_test)$class
head(pred.lda)
table(pred.lda,dat_test$price_range)


##Estimating true risk
ind = sample(1:2000,2000,replace = FALSE)
i=1
count1 = NULL
for(i in 1:5)
{
  test = ind[((i-1)*400+1):(i*400)]
  dat_train = dat1[-test,]
  dat_test = dat1[test,]
  mod.lda <- lda(price_range ~ .,data = dat_train)
  pred.lda <- predict(mod.lda,newdata = dat_test)$class
  head(pred.lda)
  Tb = table(pred.lda,dat_test$price_range)
  count1[i] = 400-sum(diag(Tb))  
}

count1
mean(count1)

##QDA
mod.qda <- qda(price_range ~ .,data = dat_train)
pred.qda <- predict(mod.qda,newdata = dat_test)$class
head(pred.qda)
table(pred.qda,dat_test$price_range)

##Estimating true risk
ind = sample(1:2000,2000,replace = FALSE)
i=1
count2 = NULL
for(i in 1:5)
{
  test = ind[((i-1)*400+1):(i*400)]
  dat_train = dat1[-test,]
  dat_test = dat1[test,]
  mod.qda <- qda(price_range ~ .,data = dat_train)
  pred.qda <- predict(mod.qda,newdata = dat_test)$class
  Tb = table(pred.qda,dat_test$price_range)
  count2[i] = 400-sum(diag(Tb))  
}

count2
mean(count2)


##KNN
mod.knn <- knn(train = dat_train,test = dat_test,cl = dat_train$price_range,k = 20)
table(mod.knn,dat_test$price_range)


ind = sample(1:2000,2000,replace = FALSE)
i=1
count3 = NULL
for(i in 1:5)
{
  test = ind[((i-1)*400+1):(i*400)]
  dat_train = dat1[-test,]
  dat_test = dat1[test,]
  mod.knn <- knn(train = dat_train,test = dat_test,cl = dat_train$price_range,k = 20)
  Tb = table(mod.knn,dat_test$price_range)
  count3[i] = 400-sum(diag(Tb))  
}

count3
mean(count3)



##SVMs
svmfit <- svm(price_range ~ ., data = dat_train, kernel = "linear",
              cost = 100)

table(predict(svmfit,newdata = dat_test),dat_test$price_range)
dim(dat_test)

tune.out <- tune(svm, price_range ~ ., data = dat_train,
                 kernel = "linear",
                 ranges = list(
                   cost = c(0.01,1,100,1000)
                 )
)

tune.out$best.parameters


ind = sample(1:2000,2000,replace = FALSE)
i=1
count4 = NULL
for(i in 1:5)
{
  test = ind[((i-1)*400+1):(i*400)]
  dat_train = dat1[-test,]
  dat_test = dat1[test,]
  svmfit <- svm(price_range ~ ., data = dat_train, kernel = "linear",
                cost = 100)
  Tb = table(predict(svmfit,newdata = dat_test),dat_test$price_range)
  count4[i] = 400-sum(diag(Tb))  
}

count4
mean(count4)

count1;count2;count3;count4

plot(1:5,count1,type = "o",col = "red",pch = 20,cex = 1,xlab = "Sample",ylab = "No of Misclassifications",main = "Comparison of Different Classifiers",ylim = c(0,50))
lines(1:5,count2,type = "o",col = "blue",pch = 19,cex = 0.5)
lines(1:5,count3,type = "o",col = "green",pch = 18,cex = 1.2)
lines(1:5,count4,type = "o",col = "darkorchid",pch = 17,cex = 1.4)
legend("topright",horiz = TRUE,legend = c("LDA","QDA","KNN","SVM"),col = c("red","blue","green","darkorchid"),pch = c(20,19,18,17))
