title: "Kinematic Features Final Project"
author: "Sartaj Saadman"

#####################################loading the libraries required################################
library(tidyr)
library(dplyr)
library(MASS)
library(class)
library(caret)
library(e1071)

##########################################loading the labeled data##################################
trn.dat=read.csv("/Users/Sartaj/Desktop/MASTERS/602/Final/New folder/labeled.csv", stringsAsFactors =T)[, -1]
trn.dat <- trn.dat[ , -which(names(trn.dat) %in% c("RelativeDurationofPrimary","RelativeSizeofPrimary"))]

index.var=apply(trn.dat[, 1:4], 1, paste, collapse = ":")
uni.vars=unique(index.var)

trn.dat.means=NULL
inde.vars=NULL
for (i in uni.vars){
  trn.dat.i=trn.dat[i==index.var, ]
  trn.mean.i=colMeans(trn.dat.i[,-(1:5)])
  inde.vars=rbind(inde.vars, trn.dat.i[1,(1:4)])
  trn.dat.means=rbind(trn.dat.means, trn.mean.i)
}

#Train and test
fulldata <- cbind(inde.vars,trn.dat.means)
sample.row <- sample(nrow(fulldata),round(nrow(fulldata)*.90)) 
train.data <- fulldata[sample.row,]
test.data <- fulldata[-sample.row,]
train.y <- train.data[, 1:4]
train.x <- train.data[, -(1:4)]
test.y <- test.data[, 1:4]
test.x <- test.data[, -(1:4)]

#full dataset
classes.joint.full=apply(inde.vars[,1:3], 1, paste, collapse = ":")
classes.joint.full <- as.data.frame(classes.joint.full)
fulldata.joint <- cbind(classes.joint.full,fulldata[,4:ncol(fulldata)])
fulldata.joint <- as.data.frame(fulldata.joint)
fulldata.joint$Trial <- NULL
colnames(fulldata.joint)[1] <- "Y"

fulldata.group <- fulldata[,-c(2,3,4)]
fulldata.sub <- fulldata[,-c(1,3,4)]
fulldata.con <- fulldata[,-c(1,2,4)]

#train dataset
traindata.joint <- apply(train.y[,1:3], 1, paste, collapse = ":")
traindata.joint <- as.data.frame(traindata.joint)
traindata.joint <- cbind(traindata.joint, train.x)
colnames(traindata.joint)[1] <- "Y"

traindata.group <- train.data[,-c(2,3,4)]
traindata.sub <- train.data[,-c(1,3,4)]
traindata.con <- train.data[,-c(1,2,4)]

#test dataset
testdata.joint <- apply(test.y[,1:3], 1, paste, collapse = ":")
testdata.joint <- as.data.frame(testdata.joint)
testdata.joint <- cbind(testdata.joint, test.x)
colnames(testdata.joint)[1] <- "Y"

testdata.group <- test.data[,-c(2,3,4)]
testdata.sub <- test.data[,-c(1,3,4)]
testdata.con <- test.data[,-c(1,2,4)]


####################################loading the unlabeled example data#############################
unlabeled.examp=read.csv("/Users/Sartaj/Desktop/MASTERS/602/Final/New folder/unlab.example.trial.csv",
                         stringsAsFactors = F)[, -1]
unlab.dat.means=NULL
uni.vars1=unique(unlabeled.examp$Trial)
for (i in uni.vars1){
  unlab.dat.i=unlabeled.examp[i==unlabeled.examp$Trial, ]
  unlab.mean.i=colMeans(unlab.dat.i[,-(1:2)])
  unlab.dat.means=rbind(unlab.dat.means, unlab.mean.i)
}
unlab.dat.means=unlab.dat.means[, -(13:14)]


############################################LDA with PCA##############################################

#Joint model
set.seed(702)
trctrl <- trainControl(method = "cv", number = 10)
LDA_Model_Joint <- train(Y ~., 
                         data = traindata.joint, 
                         method = "lda",
                         trControl=trctrl,
                         preProcess=c('pca','center','scale'))


#Group model
LDA_Model_Group <- train(Group ~., 
                         data = traindata.group, 
                         method = "lda",
                         trControl=trctrl,
                         preProcess=c('pca','center','scale'))

#Subject model
LDA_Model_Subject <- train(Subject ~., 
                           data = traindata.sub, 
                           method = "lda",
                           trControl=trctrl,
                           preProcess=c('pca','center','scale'))
#condition model
LDA_Model_Condition <- train(Condition ~., 
                             data = traindata.con, 
                             method = "lda",
                             trControl=trctrl,
                             preProcess=c('pca','center','scale'))

#Model Accuracy
LDA_Model_Joint$results
LDA_Model_Group$results
LDA_Model_Subject$results
LDA_Model_Condition$results

#prediction accuracy for joint
LDA_Predict <- predict(LDA_Model_Joint, newdata = testdata.joint)
LDA_Matches <- cbind(as.character(LDA_Predict),as.character(testdata.joint$Y))
LDA_Matches <- as.data.frame(LDA_Matches)
LDA_Matches$TF <- ifelse(as.character(LDA_Matches$V1) == as.character(LDA_Matches$V2),1,0)
sum(LDA_Matches$TF/nrow(LDA_Matches))
#prediction accuracy of group
LDA_Predict2 <- predict(LDA_Model_Group, newdata = testdata.group)
LDA_Matches2 <- LDA_Predict2 == testdata.group$Group
cf_matrix_lda <- confusionMatrix(LDA_Predict2, testdata.group$Group)
cf_matrix_lda 
#prediction accuracy of subject
LDA_Predict3 <- predict(LDA_Model_Subject, newdata = testdata.sub)
LDA_Matches3 <- LDA_Predict3 == testdata.sub$Subject
cf_matrix_lda2 <- confusionMatrix(LDA_Predict3, testdata.sub$Subject)
cf_matrix_lda2 
#prediction accuracy of condition
LDA_Predict4 <- predict(LDA_Model_Condition, newdata = testdata.con)
LDA_Matches4 <- LDA_Predict4 == testdata.con$Condition
cf_matrix_lda3 <- confusionMatrix(LDA_Predict4, testdata.con$Condition)
cf_matrix_lda3

##prediction of Unlabeled example
Joint.predict.LDA <- predict(LDA_Model_Joint, newdata = unlab.dat.means)
Group.predict.LDA <- predict(LDA_Model_Group, newdata = unlab.dat.means)
Subject.predict.LDA <- predict(LDA_Model_Subject, newdata = unlab.dat.means)
Condition.predict.LDA <- predict(LDA_Model_Condition, newdata = unlab.dat.means)



##########################################LDA without PCA#########################################

#Joint model
set.seed(702)
trctrl <- trainControl(method = "cv", number = 10)
LDA_Model_Joint2 <- train(Y ~., 
                          data = traindata.joint, 
                          method = "lda",
                          trControl=trctrl,
                          preProcess = c('scale'))


#Group model
LDA_Model_Group2 <- train(Group ~., 
                          data = traindata.group, 
                          method = "lda",
                          trControl=trctrl,
                          preProcess = c('scale'))

#Subject model
LDA_Model_Subject2 <- train(Subject ~., 
                            data = traindata.sub, 
                            method = "lda",
                            trControl=trctrl,
                            preProcess = c('scale'))
#condition model
LDA_Model_Condition2 <- train(Condition ~., 
                              data = traindata.con, 
                              method = "lda",
                              trControl=trctrl,
                              preProcess = c('scale'))

#Model Accuracy
LDA_Model_Joint2$results
LDA_Model_Group2$results
LDA_Model_Subject2$results
LDA_Model_Condition2$results

#prediction accuracy for joint
LDA_Predict <- predict(LDA_Model_Joint2, newdata = testdata.joint)
LDA_Matches <- cbind(as.character(LDA_Predict),as.character(testdata.joint$Y))
LDA_Matches <- as.data.frame(LDA_Matches)
LDA_Matches$TF <- ifelse(as.character(LDA_Matches$V1) == as.character(LDA_Matches$V2),1,0)
sum(LDA_Matches$TF/nrow(LDA_Matches))
#prediction accuracy of group
LDA_Predict2 <- predict(LDA_Model_Group2, newdata = testdata.group)
LDA_Matches2 <- LDA_Predict2 == testdata.group$Group
cf_matrix_lda <- confusionMatrix(LDA_Predict2, testdata.group$Group)
cf_matrix_lda 
#prediction accuracy of subject
LDA_Predict3 <- predict(LDA_Model_Subject2, newdata = testdata.sub)
LDA_Matches3 <- LDA_Predict3 == testdata.sub$Subject
cf_matrix_lda2 <- confusionMatrix(LDA_Predict3, testdata.sub$Subject)
cf_matrix_lda2 
#prediction accuracy of condition
LDA_Predict4 <- predict(LDA_Model_Condition2, newdata = testdata.con)
LDA_Matches4 <- LDA_Predict4 == testdata.con$Condition
cf_matrix_lda3 <- confusionMatrix(LDA_Predict4, testdata.con$Condition)
cf_matrix_lda3

##prediction of Unlabeled example
Joint.predict.LDA <- predict(LDA_Model_Joint2, newdata = unlab.dat.means)
Group.predict.LDA <- predict(LDA_Model_Group2, newdata = unlab.dat.means)
Subject.predict.LDA <- predict(LDA_Model_Subject2, newdata = unlab.dat.means)
Condition.predict.LDA <- predict(LDA_Model_Condition2, newdata = unlab.dat.means)



###########################################KNN with pca###########################################

#Joint model
set.seed(702)
grid_knn <- expand.grid(k = c(1:10))
trctrl <- trainControl(method = "cv", number = 10)
KNN_Model_Joint <- train(Y ~., 
                         data = traindata.joint, 
                         method = "knn",
                         trControl=trctrl,
                         tuneGrid = grid_knn,
                         preProcess=c('pca','center','scale'))


#Group model
KNN_Model_Group <- train(Group ~., 
                         data = traindata.group, 
                         method = "knn",
                         trControl=trctrl,
                         tuneGrid = grid_knn,
                         preProcess=c('pca','center','scale'))


#Subject model
KNN_Model_Subject <- train(Subject ~., 
                           data = traindata.sub, 
                           method = "knn",
                           trControl=trctrl,
                           tuneGrid = grid_knn,
                           preProcess=c('pca','center','scale'))


#Condition model
KNN_Model_Condition <- train(Condition ~., 
                             data = traindata.con, 
                             method = "knn",
                             trControl=trctrl,
                             tuneGrid = grid_knn,
                             preProcess=c('pca','center','scale'))


plot(KNN_Model_Joint, main = "Joint")
plot(KNN_Model_Group, main = "Group")
plot(KNN_Model_Subject, main = "Subject")
plot(KNN_Model_Condition, main = "Condition")

#Model Accuracy
KNN_Model_Joint$results
KNN_Model_Group$results
KNN_Model_Subject$results
KNN_Model_Condition$results

#prediction accuracy for joint
KNN_Predict <- predict(KNN_Model_Joint, newdata = testdata.joint)
KNN_Matches <- cbind(as.character(KNN_Predict),as.character(testdata.joint$Y))
KNN_Matches <- as.data.frame(KNN_Matches)
KNN_Matches$TF <- ifelse(as.character(KNN_Matches$V1) == as.character(KNN_Matches$V2),1,0)
sum(KNN_Matches$TF/nrow(KNN_Matches))
#prediction accuracy of group
KNN_Predict2 <- predict(KNN_Model_Group, newdata = testdata.group)
KNN_Matches2 <- KNN_Predict2 == testdata.group$Y
cf_matrix_knn <- confusionMatrix(KNN_Predict2, testdata.group$Group)
cf_matrix_knn 
#prediction accuracy of subject
KNN_Predict3 <- predict(KNN_Model_Subject, newdata = testdata.sub)
KNN_Matches3 <- KNN_Predict3 == testdata.sub$Subject
cf_matrix_knn2 <- confusionMatrix(KNN_Predict3, testdata.sub$Subject)
cf_matrix_knn2 
#prediction accuracy of condition
KNN_Predict4 <- predict(KNN_Model_Condition, newdata = testdata.con)
KNN_Matches4 <- KNN_Predict4 == testdata.con$Condition
cf_matrix_knn3 <- confusionMatrix(KNN_Predict4, testdata.con$Condition)
cf_matrix_knn3

#Prediction of unlabeled example
Joint.predict.KNN <- predict(KNN_Model_Joint, newdata = unlab.dat.means)
Group.predict.KNN <- predict(KNN_Model_Group, newdata = unlab.dat.means)
Subject.predict.KNN <- predict(KNN_Model_Subject, newdata = unlab.dat.means)
Condition.predict.KNN <- predict(KNN_Model_Condition, newdata = unlab.dat.means)



#########################################KNN without pca###########################################

#Joint model
set.seed(702)
grid_knn <- expand.grid(k = c(1:10))
trctrl <- trainControl(method = "cv", number = 10)
KNN_Model_Joint2 <- train(Y ~., 
                          data = traindata.joint, 
                          method = "knn",
                          trControl=trctrl,
                          tuneGrid = grid_knn,
                          preProcess = c('scale'))
plot(KNN_Model_Joint2, main = "Joint")

#Group model
KNN_Model_Group2 <- train(Group ~., 
                          data = traindata.group, 
                          method = "knn",
                          trControl=trctrl,
                          tuneGrid = grid_knn,
                          preProcess = c('scale'))
plot(KNN_Model_Group2, main = "Group")

#Subject model
KNN_Model_Subject2 <- train(Subject ~., 
                            data = traindata.sub, 
                            method = "knn",
                            trControl=trctrl,
                            tuneGrid = grid_knn,
                            preProcess = c('scale'))
plot(KNN_Model_Subject2, main = "Subject")

#Condition model
KNN_Model_Condition2 <- train(Condition ~., 
                              data = traindata.con, 
                              method = "knn",
                              trControl=trctrl,
                              tuneGrid = grid_knn,
                              preProcess = c('scale'))
plot(KNN_Model_Condition2, main = "Condition")

#Model Accuracy
KNN_Model_Joint2$results
KNN_Model_Group2$results
KNN_Model_Subject2$results
KNN_Model_Condition2$results

#prediction accuracy for joint
KNN_Predict <- predict(KNN_Model_Joint2, newdata = testdata.joint)
KNN_Matches <- cbind(as.character(KNN_Predict),as.character(testdata.joint$Y))
KNN_Matches <- as.data.frame(KNN_Matches)
KNN_Matches$TF <- ifelse(as.character(KNN_Matches$V1) == as.character(KNN_Matches$V2),1,0)
sum(KNN_Matches$TF/nrow(KNN_Matches))
#prediction accuracy of group
KNN_Predict2 <- predict(KNN_Model_Group2, newdata = testdata.group)
KNN_Matches2 <- KNN_Predict2 == testdata.group$Y
cf_matrix_knn <- confusionMatrix(KNN_Predict2, testdata.group$Group)
cf_matrix_knn 
#prediction accuracy of subject
KNN_Predict3 <- predict(KNN_Model_Subject2, newdata = testdata.sub)
KNN_Matches3 <- KNN_Predict3 == testdata.sub$Subject
cf_matrix_knn2 <- confusionMatrix(KNN_Predict3, testdata.sub$Subject)
cf_matrix_knn2 
#prediction accuracy of condition
KNN_Predict4 <- predict(KNN_Model_Condition2, newdata = testdata.con)
KNN_Matches4 <- KNN_Predict4 == testdata.con$Condition
cf_matrix_knn3 <- confusionMatrix(KNN_Predict4, testdata.con$Condition)
cf_matrix_knn3

#Prediction of unlabeled example
Joint.predict.KNN <- predict(KNN_Model_Joint2, newdata = unlab.dat.means)
Group.predict.KNN <- predict(KNN_Model_Group2, newdata = unlab.dat.means)
Subject.predict.KNN <- predict(KNN_Model_Subject2, newdata = unlab.dat.means)
Condition.predict.KNN <- predict(KNN_Model_Condition2, newdata = unlab.dat.means)












