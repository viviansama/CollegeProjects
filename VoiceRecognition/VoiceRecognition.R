library(factoextra)
library(ggplot2)
library(scatterplot3d)
library(tidyverse)
library(ROCR)
library(caret)
library(doParallel)
library(ellipse)
library(plyr)
library(expss)
library(labelled)
library(e1071)
library(class)
library(ROCR)
library(profvis)
library(psych)
library(nFactors)
library(GPArotation)
library(corrplot)

#####################################################################################################
#####################################################################################################
# Load dataset
voiceDB <- read.csv("voice.csv", header = TRUE, sep = ",")
str(voiceDB)
voiceDB$label <- as.factor(voiceDB$label)
is.factor(voiceDB$label)
sum(is.na(voiceDB)) #No missing data
summary(voiceDB)
#####################################################################################################
#Bar Plot
counts <- table(voiceDB$label)
barplot(counts, main = "Gender Distribution",
        xlab = "Genres", col = c("red", "darkblue"))

#Numerical Genre Distribution
table(voiceDB$label)
#####################################################################################################
#How does each numerical variable vary across the labels?
voiceDB %>% na.omit() %>%
  gather(type, value, 1:20) %>%
  ggplot(aes(x = value, fill = label)) + geom_density(alpha = 0.6) +
  facet_wrap(~type, scales = "free") + theme(axis.text.x = element_text(angle = 50, vjust = 1)) +
  labs(title = "Density Plots of Data across Variables")
#####################################################################################################
#Correlation Matrix
corrplot(cor(voiceDB[, 1:20]), order = "AOE", type = "full", method = "color", addCoef.col = "grey", tl.cex = 0.9)
#####################################################################################################
#PRE-PROCESSING
#Mix the entire dataset
set.seed(9850)
gp <- runif(nrow(voiceDB))
voiceDB <- voiceDB[order(gp),] #Mixing
head(voiceDB, 50)
#####################################################################################################
# Re-Scale the numerical features
summary(voiceDB[, c(1:20)]) #Checking the Min and Max
# Auxiliary function for normalizing
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
voiceDB$meanfreq <- normalize(voiceDB$meanfreq)
voiceDB$sd <- normalize(voiceDB$sd)
voiceDB$median <- normalize(voiceDB$median)
voiceDB$Q25 <- normalize(voiceDB$Q25)
voiceDB$Q75 <- normalize(voiceDB$Q75)
voiceDB$IQR <- normalize(voiceDB$IQR)
voiceDB$skew <- normalize(voiceDB$skew)
voiceDB$kurt <- normalize(voiceDB$kurt)
voiceDB$sp.ent <- normalize(voiceDB$sp.ent)
voiceDB$sfm <- normalize(voiceDB$sfm)
voiceDB$mode <- normalize(voiceDB$mode)
voiceDB$centroid <- normalize(voiceDB$centroid)
voiceDB$meanfun <- normalize(voiceDB$meanfun)
voiceDB$minfun <- normalize(voiceDB$minfun)
voiceDB$maxfun <- normalize(voiceDB$maxfun)
voiceDB$meandom <- normalize(voiceDB$meandom)
voiceDB$mindom <- normalize(voiceDB$mindom)
voiceDB$maxdom <- normalize(voiceDB$maxdom)
voiceDB$dfrange <- normalize(voiceDB$dfrange)
voiceDB$modindx <- normalize(voiceDB$modindx)

#####################################################################################################
#####################################################################################################

#####################################################################################################
#####################################################################################################

# Run K-means and determine classification precision
accumulatedPrecision <- 0
for (i in 1:100) {
  # K-means
  k_unnorm <- kmeans(voiceDB[c(1:20)], centers = 2, nstart = 20)
  #fviz_cluster(k_unnorm, data = voiceDB[c(1:20)])
  classificationCount <- 0
  for (i in 1:length(voiceDB$label)) {
    if (voiceDB$label[i] == "female" & k_unnorm$cluster[i] == 1) {
      classificationCount <- classificationCount + 1
    }
    if (voiceDB$label[i] == "male" & k_unnorm$cluster[i] == 2) {
      classificationCount <- classificationCount + 1
    }
  }
  accumulatedPrecision <- accumulatedPrecision + (classificationCount / length(voiceDB$label))
}

accumulatedPrecision <- accumulatedPrecision / 100
accumulatedPrecision

#####################################################################################################
#####################################################################################################

#####################################################################################################
#####################################################################################################

#CREATE A TRAINING DATASET(use to learn a pattern) AND A TEST DATASET(to test how well is our model predicts)
# Run KNN and determine classification precision

##Generate a random number that is 80% of the total number of rows in dataset.
ran <- sample(1:nrow(voiceDB), 0.8 * nrow(voiceDB))

##extract training set
voiceDB_train <- voiceDB[1:2534, 1:20]
##extract testing set
voiceDB_test <- voiceDB[2535:3168, 1:20]

##extract the column 21 for train dataset because it will be used as 'cl' argument in knn function.
voiceDB_target_category <- voiceDB[1:2534, 21]

##extract the column 21 for test the dataset to measure the accuracy
voiceDB_test_category <- voiceDB[2535:3168, 21]

#select k
sqrt(3168) #56.2 select a the odd number
##run knn function
require(class)
voiceDBKNN <- knn(train = voiceDB_train, test = voiceDB_test, cl = voiceDB_target_category, k = 57)
voiceDBKNN
##create confusion matrix
tab <- table(voiceDB_test_category, voiceDBKNN)
tab
##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
accuracy <- function(x) { sum(diag(x) / (sum(rowSums(x)))) * 100 }
accuracy(tab)

#####################################################################################################
#####################################################################################################


#####################################################################################################
#####################################################################################################
# PCA ANALYSIS whit Correlation Matrix
voiceDB.pr <- prcomp(voiceDB[c(1:20)], center = TRUE, scale = TRUE)
summary(voiceDB.pr)

screeplot(voiceDB.pr, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 2.2, col = "red", lty = 5)
legend("topright", legend = c("Eigenvalue = 1"),
       col = c("red"), lty = 5, cex = 0.6)
cumpro <- cumsum(voiceDB.pr$sdev ^ 2 / sum(voiceDB.pr$sdev ^ 2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 3, col = "blue", lty = 5)
abline(h = 0.68, col = "blue", lty = 5)
legend("topleft", legend = c("Cut-off @ PC3"),
       col = c("blue"), lty = 5, cex = 0.6)

scatterplot3d(voiceDB.pr$x[, 1:3], angle = 80, xlab = "PC1 (45.2%)", ylab = "PC2 (11.87%)", zlab = "PC3(10.91%)", main = "PC1 / PC2 / PC3 - plot")

PCAresults <- voiceDB.pr$x[, 1:3]

####EigenVectors plots
biplot(voiceDB.pr)

summary(voiceDB.pr)
biplot(voiceDB.pr, xlabs = rep("", nrow(voiceDB))) #
fviz_eig(voiceDB.pr)


voiceDB.pr$rotation
voiceDB.pr$center
voiceDB.pr$scale
voiceDB.pr$n.obs
voiceDB.pr$scores
voiceDB.pr$loadings
voiceDB.pr$call
voiceDB.pr$sdev
#####################################################################################################
#####################################################################################################
##PCA whit Covariance Matrix
voiceDB.prCM <- prcomp(voiceDB[c(1:20)], scale = FALSE)
summary(voiceDB.prCM)

screeplot(voiceDB.prCM, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 2.2, col = "red", lty = 5)
legend("topright", legend = c("Eigenvalue = 1"),
       col = c("red"), lty = 5, cex = 0.6)
cumpro <- cumsum(voiceDB.prCM$sdev ^ 2 / sum(voiceDB.prCM$sdev ^ 2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 3, col = "blue", lty = 5)
abline(h = 0.68, col = "blue", lty = 5)
legend("topleft", legend = c("Cut-off @ PC3"),
       col = c("blue"), lty = 5, cex = 0.6)

scatterplot3d(voiceDB.prCM$x[, 1:3], angle = 80, xlab = "PC1 (45.2%)", ylab = "PC2 (11.87%)", zlab = "PC3(10.91%)", main = "PC1 / PC2 / PC3 - plot")

PCAresults <- voiceDB.prCM$x[, 1:3]

####EigenVectors plots
biplot(voiceDB.prCM)

summary(voiceDB.prCM)
biplot(voiceDB.prCM, xlabs = rep("", nrow(voiceDB))) #
fviz_eig(voiceDB.prCM)

voiceDB.prCM$rotation
voiceDB.pr$center
voiceDB.pr$scale
voiceDB.pr$n.obs
voiceDB.pr$scores
voiceDB.pr$loadings
voiceDB.pr$call
voiceDB.pr$sdev
#####################################################################################################
#####################################################################################################
# Recalculate k-means classification precision after space reduction
accumulatedPrecisionPCA <- 0
for (i in 1:100) {
  k <- kmeans(PCAresults, centers = 2, nstart = 20)

  #####Viz
  #fviz_cluster(k, data = PCAresults)

  s3d <- scatterplot3d(voiceDB.pr$x[, 1:3], color = k$cluster, angle = 45, xlab = "PC1 (45.2%)", ylab = "PC2 (11.87%)", zlab = "PC3(10.91%)", main = "PC1 / PC2 / PC3 - plot")

  legend(s3d$xyz.convert(7.5, 3, 4.5), legend = levels(voiceDB$label), col = c("black", "red"), pch = 16)

  #K-Means
  classificationCount <- 0
  for (i in 1:length(voiceDB$label)) {
    if (voiceDB$label[i] == "female" & k$cluster[i] == 1) {
      classificationCount <- classificationCount + 1
    }
    if (voiceDB$label[i] == "male" & k$cluster[i] == 2) {
      classificationCount <- classificationCount + 1
    }
  }
  accumulatedPrecisionPCA <- accumulatedPrecisionPCA + (classificationCount / length(voiceDB$label))
}
accumulatedPrecisionPCA <- accumulatedPrecisionPCA / 100
accumulatedPrecisionPCA
#####################################################################################################
#####################################################################################################

#####################################################################################################
#####################################################################################################
# Recalculate KNN classification precision after space reduction
# Run KNN and determine classification precision

##Generate a random number that is 80% of the total number of rows in dataset.
ran <- sample(1:nrow(PCAresults), 0.8 * nrow(PCAresults))

##extract training set
voiceDB_train <- PCAresults[1:2534, 1:3]
##extract testing set
voiceDB_test <- PCAresults[2535:3168, 1:3]

##extract the column 21 for train dataset because it will be used as 'cl' argument in knn function.
voiceDB_target_category <- voiceDB[1:2534, 21]

##extract the column 21 for test the dataset to measure the accuracy
voiceDB_test_category <- voiceDB[2535:3168, 21]

#select k
sqrt(3168) #56.2 select a the odd number
##run knn function
require(class)
voiceDBKNNPCA <- knn(train = voiceDB_train, test = voiceDB_test, cl = voiceDB_target_category, k = 57)
voiceDBKNNPCA
##create confusion matrix
tab <- table(voiceDB_test_category, voiceDBKNNPCA)
tab
##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
accuracy <- function(x) { sum(diag(x) / (sum(rowSums(x)))) * 100 }
accuracy(tab)

#####################################################################################################
