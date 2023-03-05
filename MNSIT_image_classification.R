library("keras")
mnist <- dataset_mnist()

x_train <- mnist$train$x
g_train <- mnist$train$y
x_test <- mnist$test$x
g_test <- mnist$test$y

ind1 = sample(1:nrow(x_train),size = 5000)
ind2 = sample(1:nrow(x_test),size = 500)

x_trn = x_train[ind1,,]  
x_tst = x_test[ind2,,]

x_trn_mat = matrix(x_trn, nrow = nrow(x_trn))
x_tst_mat = matrix(x_tst, nrow = nrow(x_tst))


library(nnet)

fm_nnet <- nnet(x = x_trn_mat, y = g_train[ind1],
                size = 15,
                entropy = TRUE,
                MaxNWts = 5000, decay = 5e-04,
                maxit = 200)

