### 4) DATASET SPLIT & PREPARE FOR ML
# Preparo dataset originale
set.seed(123)
smp_size = floor(0.70 * nrow(data))
train_ind = sample(seq_len(nrow(data)), size = smp_size)
train = data[train_ind, -c(1)]
test = data[-train_ind, -c(1)]

# controllo bilanciamento di train e test set
prop.table(table(train$diagnosis))*100
prop.table(table(test$diagnosis))*100

# dataset ridotto
train2 = data2[train_ind, -c(1)]
test2 = data2[-train_ind, -c(1)]




############################ K MEANS ##############
predict.kmeans <- function(newdata, object){
  centers <- object$centers
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}


#### K-MEANS - Dataset Originale
learn_kmeans = kmeans(train[,-c(1)], centers=2)
learn_kmeans = kmeans(train[,-c(1)], centers=2)
pre_kmeans = predict.kmeans(test[,-c(1)],learn_kmeans)
pre_kmeans = factor(ifelse(pre_kmeans == 1,"B","M"))

# Silhoutte dei 2 modelli K-Means
kms = silhouette(learn_kmeans$cluster, dist(train[,-c(1)]))
plot(kms) 

# matrice di dissimilarità
dissplot(dist(train[,-c(1)]), labels=learn_kmeans$cluster,options=list(main="Dissimilarity Matrix"))


# matrice confusione globale & plot cluster
cm_kmeans = confusionMatrix(pre_kmeans, test$diagnosis)
cm_kmeans
learn_kmeans$cluster = ifelse(learn_kmeans$cluster == 1,"B","M")
fviz_cluster(learn_kmeans, data = train[,-c(1,2)])

# matrice confusione per "Malignant"
cm_kmeans_m = confusionMatrix(pre_kmeans, test$diagnosis, mode = "prec_recall", positive = "M")
cm_kmeans_m

# matrice confusione per "Benign"
cm_kmeans_b = confusionMatrix(pre_kmeans, test$diagnosis, mode = "prec_recall", positive = "B")
cm_kmeans_b

# Precision Malignant
precision_m = cm_kmeans_m[["byClass"]] [["Precision"]]
# Precision Benign
precision_b = cm_kmeans_b[["byClass"]] [["Precision"]]
# Precision Macro Average
precision_macro_average = mean(c(precision_m, precision_b))

precision_m
precision_b
precision_macro_average



# Recall Malignant
recall_m = cm_kmeans_m[["byClass"]] [["Recall"]]
# Recall Benign
recall_b = cm_kmeans_b[["byClass"]] [["Recall"]]
# Recall Macro Average
recall_macro_average = mean(c(recall_m, recall_b))

recall_m
recall_b
recall_macro_average


# F1 Malignant
f1_m = cm_kmeans_m[["byClass"]] [["F1"]]
# F1 Benign
f1_b = cm_kmeans_b[["byClass"]] [["F1"]]
# F1 Macro Average
f1_macro_average = mean(c(f1_m, f1_b))

f1_m
f1_b
f1_macro_average







#### K-MEANS - Dataset Ridotto 
set.seed(42)
learn_kmeans = kmeans(train2[,-c(1)], centers=2)
pre_kmeans = predict.kmeans(test2[,-c(1)],learn_kmeans)
pre_kmeans = factor(ifelse(pre_kmeans == 1,"B","M"))

# Silhoutte dei 2 modelli K-Means 
kms = silhouette(learn_kmeans$cluster, dist(train2[,-c(1)]))
plot(kms)

# Matrice di dissimilarità
# se l'incrocio è chiaro allora non sono simili, se scuro allora simili
dissplot(dist(train[,-c(1)]), labels=learn_kmeans$cluster,options=list(main="Dissimilarity Matrix"))

# matrice confusione globale & plot cluster
cm_kmeans_ridotto = confusionMatrix(pre_kmeans, test$diagnosis)
cm_kmeans_ridotto
learn_kmeans$cluster <- ifelse(learn_kmeans$cluster == 1,"B","M")
fviz_cluster(learn_kmeans, data = train[,-c(1,2)])#si vede che c'è un po' di overlapping

# matrice confusione per "Malignant"
cm_kmeans_m = confusionMatrix(pre_kmeans, test$diagnosis, mode = "prec_recall", positive = "M")
cm_kmeans_m

# matrice confusione per "Benign"
cm_kmeans_b = confusionMatrix(pre_kmeans, test$diagnosis, mode = "prec_recall", positive = "B")
cm_kmeans_b

# Precision Malignant
precision_m = cm_kmeans_m[["byClass"]] [["Precision"]]
# Precision Benign
precision_b = cm_kmeans_b[["byClass"]] [["Precision"]]
# Precision Macro Average
precision_macro_average = mean(c(precision_m, precision_b))

precision_m
precision_b
precision_macro_average



# Recall Malignant
recall_m = cm_kmeans_m[["byClass"]] [["Recall"]]
# Recall Benign
recall_b = cm_kmeans_b[["byClass"]] [["Recall"]]
# Recall Macro Average
recall_macro_average = mean(c(recall_m, recall_b))

recall_m
recall_b
recall_macro_average



# F1 Malignant
f1_m = cm_kmeans_m[["byClass"]] [["F1"]]
# F1 Benign
f1_b = cm_kmeans_b[["byClass"]] [["F1"]]
# F1 Macro Average
f1_macro_average = mean(c(f1_m, f1_b))

f1_m
f1_b
f1_macro_average


#### Confronto 2 modelli K-Means

col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,2))
fourfoldplot(cm_kmeans$table, color = col, conf.level = 0, margin = 1, main=paste("K-Means (",round(cm_kmeans$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_kmeans_ridotto$table, color = col, conf.level = 0, margin = 1, main=paste("K-Means - Ridotto (",round(cm_kmeans_ridotto$overall[1]*100),"%)",sep=""))
dev.off()














############################ RETI NEURALI ##############
# FitControl
fitControl = trainControl(method="cv", #cross validation
                          number = 10, #10 fold
                          preProcOptions = list(thresh = 0.65), #soglia varianza PCA
                          classProbs = TRUE,
                          savePredictions = "final",
                          summaryFunction = twoClassSummary)



## RETE NEURALE - Dataset Originale
model_nnet <- train(diagnosis~.,
                    data[,-(1)],
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    type ="Classification",
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)

# matrice confusione globale
cm_nnet = confusionMatrix(model_nnet$pred$pred, as.factor(model_nnet$pred$obs))
cm_nnet
plotnet(model_nnet)

cm_nnet_m <- confusionMatrix(model_nnet$pred$pred, as.factor(model_nnet$pred$obs), mode = "prec_recall", positive = "M")
cm_nnet_m

cm_nnet_b <- confusionMatrix(model_nnet$pred$pred, as.factor(model_nnet$pred$obs), mode = "prec_recall", positive = "B")
cm_nnet_b

# Precision Malignant
precision_m = cm_nnet_m[["byClass"]] [["Precision"]]
# Precision Benign
precision_b = cm_nnet_b[["byClass"]] [["Precision"]]
# Precision Macro Average
precision_macro_average = mean(c(precision_m, precision_b))

precision_m
precision_b
precision_macro_average



# Recall Malignant
recall_m = cm_nnet_m[["byClass"]] [["Recall"]]
# Recall Benign
recall_b = cm_nnet_b[["byClass"]] [["Recall"]]
# Recall Macro Average
recall_macro_average = mean(c(recall_m, recall_b))

recall_m
recall_b
recall_macro_average


# F1 Malignant
f1_m = cm_nnet_m[["byClass"]] [["F1"]]
# F1 Benign
f1_b = cm_nnet_b[["byClass"]] [["F1"]]
# F1 Macro Average
f1_macro_average = mean(c(f1_m, f1_b))

f1_m
f1_b
f1_macro_average





## RETE NEURALE - Dataset Ridotto
model_nnet_ridotto <- train(diagnosis~.,
                            data2[,-(1)],
                            method="nnet",
                            metric="ROC",
                            preProcess=c('center', 'scale'),
                            type ="Classification",
                            trace=FALSE,
                            tuneLength=10,
                            trControl=fitControl)

# matrice confusione globale
plotnet(model_nnet_ridotto)
cm_nnet_ridotto = confusionMatrix(model_nnet_ridotto$pred$pred, as.factor(model_nnet_ridotto$pred$obs))
cm_nnet_ridotto

cm_nnet_ridotto_m <- confusionMatrix(model_nnet_ridotto$pred$pred, as.factor(model_nnet_ridotto$pred$obs), mode = "prec_recall", positive = "M")
cm_nnet_ridotto_m

cm_nnet_ridotto_b <- confusionMatrix(model_nnet_ridotto$pred$pred, as.factor(model_nnet_ridotto$pred$obs), mode = "prec_recall", positive = "B")
cm_nnet_ridotto_b

# Precision Malignant
precision_m = cm_nnet_ridotto_m[["byClass"]] [["Precision"]]
# Precision Benign
precision_b = cm_nnet_ridotto_b[["byClass"]] [["Precision"]]
# Precision Macro Average
precision_macro_average = mean(c(precision_m, precision_b))

precision_m
precision_b
precision_macro_average



# Recall Malignant
recall_m = cm_nnet_ridotto_m[["byClass"]] [["Recall"]]
# Recall Benign
recall_b = cm_nnet_ridotto_b[["byClass"]] [["Recall"]]
# Recall Macro Average
recall_macro_average = mean(c(recall_m, recall_b))

recall_m
recall_b
recall_macro_average


# F1 Malignant
f1_m = cm_nnet_ridotto_m[["byClass"]] [["F1"]]
# F1 Benign
f1_b = cm_nnet_ridotto_b[["byClass"]] [["F1"]]
# F1 Macro Average
f1_macro_average = mean(c(f1_m, f1_b))

f1_m
f1_b
f1_macro_average





## RETE NEURALE - Dataset PCA
model_pca_nnet <- train(diagnosis~.,
                        data[,-(1)],
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        type ="Classification",
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

plotnet(model_pca_nnet)

# matrice confusione globale
cm_pca_nnet = confusionMatrix(model_pca_nnet$pred$pred, as.factor(model_pca_nnet$pred$obs))
cm_pca_nnet

# matrice confusione per "Malignant"
cm_pca_nnet_m <- confusionMatrix(model_pca_nnet$pred$pred, as.factor(model_pca_nnet$pred$obs), mode = "prec_recall", positive = "M")
cm_pca_nnet_m

# matrice confusione per "Benign"
cm_pca_nnet_b <- confusionMatrix(model_pca_nnet$pred$pred, as.factor(model_pca_nnet$pred$obs), mode = "prec_recall", positive = "B")
cm_pca_nnet_b

# Precision Malignant
precision_m = cm_pca_nnet_m[["byClass"]] [["Precision"]]
# Precision Benign
precision_b = cm_pca_nnet_b[["byClass"]] [["Precision"]]
# Precision Macro Average
precision_macro_average = mean(c(precision_m, precision_b))

precision_m
precision_b
precision_macro_average



# Recall Malignant
recall_m = cm_pca_nnet_m[["byClass"]] [["Recall"]]
# Recall Benign
recall_b = cm_pca_nnet_b[["byClass"]] [["Recall"]]
# Recall Macro Average
recall_macro_average = mean(c(recall_m, recall_b))

recall_m
recall_b
recall_macro_average


# F1 Malignant
f1_m = cm_pca_nnet_m[["byClass"]] [["F1"]]
# F1 Benign
f1_b = cm_pca_nnet_b[["byClass"]] [["F1"]]
# F1 Macro Average
f1_macro_average = mean(c(f1_m, f1_b))

f1_m
f1_b
f1_macro_average




######################## CONFRONTO MODELLI MIGLIORI ###################
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,3))
fourfoldplot(cm_nnet$table, color = col, conf.level = 0, margin = 1, main=paste("Neural Network (",round(cm_nnet$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_nnet_ridotto$table, color = col, conf.level = 0, margin = 1, main=paste("Neural Network Ridotto (",round(cm_nnet_ridotto$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_pca_nnet$table, color = col, conf.level = 0, margin = 1, main=paste("Neural Network PCA (",round(cm_pca_nnet$overall[1]*100),"%)",sep=""))

# Tempi rete1
model_nnet$times
# Tempi rete2
model_nnet_ridotto$times
# Tempi rete3
model_pca_nnet$times





######################## CONFRONTO RETE MIGLIORE & KMEANS ###################
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,2))
fourfoldplot(cm_kmeans$table, color = col, conf.level = 0, margin = 1, main=paste("K-Means (",round(cm_kmeans$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_pca_nnet$table, color = col, conf.level = 0, margin = 1, main=paste("Neural Network (",round(cm_pca_nnet$overall[1]*100),"%)",sep=""))#


