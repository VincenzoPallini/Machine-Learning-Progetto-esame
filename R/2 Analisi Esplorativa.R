##### 2) ANALISI ESPLORATIVA
# Analisi in dettaglio per ogni attributo
summary(data)
str(data)

# Controllo di eventuali valori nulli nelle istanze
apply(data, 2, function (data) sum(is.na(data)))

# Esaminiamo il bilanciamento dei dati
table(data$diagnosis)
prop.table(table(data$diagnosis))
pie(table(data$diagnosis), labels = c("Benign", "Malignant"), border = "black")



# Variabili rispetto al target
# Attributi con valori medi
df.m = melt(data[,-c(1,13:32)], id.var = "diagnosis")
p = ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Caratteristiche Mean") + ylab("")+ guides(fill=guide_legend(title="Group"))
p


# Attributi Errore Standard
df.m = melt(data[,-c(1,3:12,23:32)], id.var = "diagnosis")
p = ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Caratteristiche SE") + ylab("")+ guides(fill=guide_legend(title="Group"))
p


# Attributi Worst
df.m = melt(data[,c(2,23:32)], id.var = "diagnosis")
p = ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Caratteristiche Worst") + ylab("")+ guides(fill=guide_legend(title="Group"))
p
# si evidenzia che in generale, le diagnosi maligne hanno punteggi più alti in tutti gli attributi




##### Analisi Correlazione
#Prima vista correlazione tra variabile mean, se, worst
ggpairs(data[,c(2,3,13,23)], aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Mean vs SE vs Worst")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))


# Prima vista di una matrice di correlazione tra tutte le variabili
matrice_correlazione = cor(data[,3:ncol(data)])
corrplot(matrice_correlazione, order="hclust", tl.cex=1, addrect = 8) #C'è una grande correlazione tra alcuni attributi


# Troviamo gli attributi che sono altamente correlati (idealmente > 0,90)
highlyCorrelated  = findCorrelation(matrice_correlazione, cutoff=0.9)
print(highlyCorrelated ) #stampa indici degli attributi altamente correlati

# Correlazione tra le variabili altamente correlate
corrdata = data[, -c(1,2)]
corrplot (cor (corrdata[, highlyCorrelated]), method="number", order = "hclust")









#### Analisi in dettaglio delle correlazione con degli Scatter Plots 
# I triangoli inferiori forniscono grafici a dispersione e i triangoli superiori forniscono valori di correlazione.
# Vediamo che ci sono correlazioni estremamente elevate tra alcune delle variabili.

ggpairs(data[,c(3:12,2)], aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Cancer Mean")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

ggpairs(data[,c(13:22,2)], aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Cancer SE")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

ggpairs(data[,c(23:32,2)], aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Cancer Worst")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))



# libreria ggcorr: ci permette di vedere meglio le correlazioni
ggcorr(data[,c(3:12)], name = "corr", label = TRUE)+
  theme(legend.position="none")+
  labs(title="Cancer Mean")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))


ggcorr(data[,c(13:22)], name = "corr", label = TRUE)+
  theme(legend.position="none")+
  labs(title="Cancer SE")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))


ggcorr(data[,c(23:32)], name = "corr", label = TRUE)+
  theme(legend.position="none")+
  labs(title="Cancer Worst")+
  theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))



#### CONCLUSIONI ANALISI CORRELAZIONI
# dopo aver visto le correlazioni si è deciso di creare un'altro dataset
# rimuovendo le variabili altamente correlate
highlyCorrelated = highlyCorrelated + 2 
data2 = data[-(highlyCorrelated)]

# numero di colonne dopo la rimozione delle variabili correlate
ncol(data2)








