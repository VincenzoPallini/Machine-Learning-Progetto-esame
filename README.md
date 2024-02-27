# Machine-Learning-Progetto-esame

**Introduzione**

Il cancro al seno rappresenta una delle principali cause di mortalità tra le donne a livello globale. La diagnosi precoce gioca un ruolo cruciale nel migliorare le possibilità di trattamento e sopravvivenza. Tuttavia, i metodi diagnostici tradizionali possono essere limitati nella loro capacità di rilevazione precoce. L'impiego di tecniche di Machine Learning (ML) nella diagnosi del cancro al seno promette di aumentare l'efficienza e l'accuratezza della diagnosi, rendendola più accessibile e meno costosa.

**Tecnologie Utilizzate**

1. **Python**: Linguaggio di programmazione scelto per lo sviluppo del progetto, grazie alla sua vasta libreria di strumenti per il data science e il machine learning.
2. **NumPy e Pandas**: Librerie per la manipolazione e l'analisi dei dati. Utilizzate per la preparazione e l'esplorazione del dataset.
3. **Matplotlib e Seaborn**: Librerie per la visualizzazione dei dati. Impiegate per generare grafici e visualizzazioni utili all'analisi esplorativa.
4. **Scikit-learn**: Libreria per il machine learning. Utilizzata per implementare algoritmi di classificazione, come Decision Tree e Reti Neurali, e per la valutazione dei modelli attraverso metriche come matrici di confusione, accuratezza, precision, recall e F1-score.

**Sviluppo del Progetto**

1. **Preparazione del Dataset**: Il dataset utilizzato, Breast Cancer Wisconsin Diagnostic (WDBC), è stato analizzato per verificare la presenza di valori nulli e per effettuare operazioni di refactoring, come la rimozione di colonne non utili all'analisi.

2. **Analisi Esplorativa**: Attraverso l'analisi di variabili, correlazioni e la creazione di boxplot, è stata esaminata la distribuzione delle caratteristiche delle cellule tumorali, distinguendo tra tumori benigni e maligni.

3. **Riduzione della Dimensionalità**: La Principal Component Analysis (PCA) è stata applicata per ridurre il numero di variabili mantenendo la maggior parte dell'informazione utile.

4. **Modellazione**: Sono stati selezionati e addestrati due modelli predittivi, Decision Tree e Reti Neurali, utilizzando due versioni del dataset ottenute dalla PCA.

5. **Valutazione dei Modelli**: Gli esperimenti hanno incluso la valutazione delle performance dei modelli attraverso metriche specifiche, curva ROC e la 10-fold cross-validation per testare la loro capacità di generalizzazione.

6. **Conclusioni**: L'analisi dei risultati ha dimostrato l'efficacia delle tecniche di ML nella diagnosi del cancro al seno, evidenziando i punti di forza e le aree di miglioramento dei modelli utilizzati.

