# Required Libraries
install.packages("pROC")

library(pROC)


# K-means Model
roc_obj_kmeans <- roc(test$diagnosis, as.numeric(pre_kmeans))
auc_kmeans <- auc(roc_obj_kmeans)

# Neural Network Model
prob_nnet <- predict(model_nnet, test, type="prob")
roc_obj_nnet <- roc(test$diagnosis, prob_nnet[,2])
auc_nnet <- auc(roc_obj_nnet)

# Visualize ROC curves
roc_obj_kmeans <- roc(test$diagnosis, as.numeric(pre_kmeans))
roc_obj_nnet <- roc(test$diagnosis, prob_nnet[,2])

# Plot the ROC curves
plot(roc_obj_kmeans, col = "blue", main = "ROC Curves")
lines(roc_obj_nnet, col = "red")
legend("bottomright", legend = c("K-means", "Neural Network"), col = c("blue", "red"), lwd = 2)
abline(a=0, b=1)

# Add AUC values to the legend
legend("topleft", legend = c(paste("K-means (AUC=", auc_kmeans, ")"), paste("Neural Network (AUC=", auc_nnet, ")")), col = c("blue", "red"), lwd = 2)

# Print AUC values to the console
print(paste0("K-means AUC: ", auc_kmeans))
print(paste0("Neural Network AUC: ", auc_nnet))

# Confronto tra modelli
cv.values = resamples(list(kmeans = pre_kmeans, neural_network = prob_nnet[,2]), method = "cv", folds = 10)

summary(cv.values)

dotplot(cv.values, metric = "ROC") 
bwplot(cv.values, layout = c(3, 1)) 
splom(cv.values,metric="ROC")
cv.values$timings

