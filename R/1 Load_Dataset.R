#### 1) Caricamento Dataset
# readr
install.packages("readr")

# dplyr
install.packages("dplyr")

# ggplot2
install.packages("ggplot2")

# corrplot
install.packages("corrplot")

# gridExtra
install.packages("gridExtra")

# pROC
install.packages("pROC")

# MASS
install.packages("MASS")

# caTools
install.packages("caTools")

# caret
install.packages("caret")

# caretEnsemble
install.packages("caretEnsemble")

# reshape2
install.packages("reshape2")

# factoextra
install.packages("factoextra")

# psych
install.packages("psych")

# GGally
install.packages("GGally")

# PerformanceAnalytics
install.packages("PerformanceAnalytics")

# cluster
install.packages("cluster")

# seriation
install.packages("seriation")

# NeuralNetTools
install.packages("NeuralNetTools")

library(needs)
needs(readr,
      dplyr,
      ggplot2,
      corrplot,
      gridExtra,
      pROC,
      MASS,
      caTools,
      caret,
      caretEnsemble,
      reshape2,
      factoextra,
      psych,
      GGally,
      PerformanceAnalytics,
      cluster,
      seriation,
      NeuralNetTools)
#registerDoMC(cores = 3)

data = read.csv("data.csv")

# Analisi veloce degli attributi presenti e della loro tipologia 
str(data)
head(data)
sapply(data, class)

# Factoring dell'attributo target "diagnosis"
data$diagnosis = factor(data$diagnosis)
sapply(data, class)

# Rimozione dell'ultima colonna la quale presenta valori nulli 
data[,33] = NULL 

# Per ogni attributo è presente 
# 1) Mean: valore medio
# 2) SE: errore standard
# 3) Worst: valore peggiore/più alto
sapply(data, class)

