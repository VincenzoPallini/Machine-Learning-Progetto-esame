#### 3) PCA - Principal Component Analysis 

# PCA applicata a tutti gli attributi
pca_res = prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
summary(pca_res)

fviz_eig(pca_res, addlabels=TRUE, ylim=c(0,60), geom = c("bar", "line"), barfill = "pink", barcolor="grey",linecolor = "red", ncp=10)+
  labs(title = "Cancer All Variances - PCA",
       x = "Principal Components", y = "% of variances")


# Analisi degli autovalori
eig.val <- get_eigenvalue(pca_res)
eig.val
## Le prime due componenti spiegano il 63% della varianza. 
## con 10 componenti raggiungiamo il 95% della varianza e con 17 componenti il 99%


# Estraggo le variabili PCA 
all_var <- get_pca_var(pca_res)
all_var


# Correlation circle
set.seed(218)
res.all <- kmeans(all_var$coord, centers = 6, nstart = 25)
grp <- as.factor(res.all$cluster)

# tutte le variabili
fviz_pca_var(pca_res, col.var = grp, 
             palette = "jco",
             legend.title = "Cluster")



## BIPLOT
# tutte le variabili
fviz_pca_biplot(pca_res, col.ind = data$diagnosis, col="black",
                palette = "jco", geom = "point", repel=TRUE,
                legend.title="Diagnosis", addEllipses = TRUE)



# Scatter plot delle nostre istanze nelle componenti 1 e 2
pca_df <- as.data.frame(pca_res$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$diagnosis)) + geom_point(alpha=0.5)


## Qualità di rappresentazione della PCA
# Correlazionie tra le variabili e la PCA
corrplot(all_var$cos2, is.corr=FALSE)    
corrplot(all_var$contrib, is.corr=FALSE)


