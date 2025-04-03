# Machine Learning Project: Breast Cancer Prediction

## Project Goal

The aim of this project was to apply and evaluate different Machine Learning techniques for classifying breast tumors as benign (B) or malignant (M). The objective was to build effective predictive models based on the characteristics of tumor cells extracted from digitized images.

**`Machine_Learning_esame.pdf`**: The main project report illustrating the methodology, experiments (focusing on Python version), results, and analysis. (*Note: This report file is in Italian*)

## Dataset

The **Breast Cancer Wisconsin Diagnostic (WDBC)** dataset, publicly available from the UCI Machine Learning Repository, was used. This dataset contains 569 samples, each described by 30 numerical features representing measures such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension of the cell nucleus. For each feature, the mean, standard error (SE), and worst/largest value are provided. The target variable is the diagnosis (Malignant/Benign).

## Methodology

1.  **Data Preparation:**
    * Loading the dataset (Python: `pandas`; R: `readr`).
    * Checking and handling null values: an entirely null column and the non-essential ID column were identified and removed.
    * Preliminary analysis of data types.

2.  **Exploratory Data Analysis (EDA):**
    * Analysis of the target variable ('diagnosis') distribution (Python: `matplotlib`/`seaborn`; R: base R `pie`).
    * Visualization of feature distributions (mean, SE, worst) against the diagnosis using boxplots (Python: `seaborn`; R: `ggplot2`).
    * Analysis of correlations between features using correlation matrices and heatmaps (Python: `seaborn`; R: `corrplot`, `GGally`, `PerformanceAnalytics`). High correlation was observed among several variables. A reduced dataset was created in R by removing highly correlated features (cutoff > 0.9).
    * Detailed correlation analysis using scatter plots (Python: `seaborn`/`matplotlib`; R: `GGally`).

3.  **Dimensionality Reduction (PCA):**
    * Principal Component Analysis (PCA) was applied after standardizing the data (Python: `sklearn.preprocessing.StandardScaler`, `sklearn.decomposition.PCA`; R: `stats::prcomp`, `factoextra`).
    * Analysis of explained variance showed that the first few principal components capture a significant portion of the total variance.
    * Two distinct datasets based on PCA (6 and 9 components) were created (primarily used in the Python implementation).

## Implemented Models

### Python Implementation (`scikit-learn`, `Keras`)

1.  **Decision Tree (`scikit-learn`):**
    * Chosen due to the good data separability observed after PCA and the binary nature of the target.
    * Post-pruning using the `ccp_alpha` parameter was applied via `sklearn.tree.DecisionTreeClassifier` to control tree complexity and prevent overfitting, selecting the optimal alpha based on test set accuracy.
    * Trained and evaluated on the 6-feature and 9-feature PCA datasets.

2.  **Neural Network (`Keras`):**
    * Chosen for its ability to model complex, non-linear relationships and its tolerance to noise.
    * A Multi-Layer Perceptron (MLP) architecture was implemented using `keras.models.Sequential` with `Dense` layers (hidden layers: 16 and 8 neurons). ReLU activation was used for hidden layers, and Sigmoid for the output layer (binary classification).
    * Optimized using Adam optimizer and `binary_crossentropy` loss function. Training involved backpropagation.
    * Two networks were trained, one for the 6-feature PCA dataset and one for the 9-feature dataset.

### R Implementation:

1.  **K-Means Clustering:**
    * K-Means clustering (with k=2) was implemented using R's `kmeans` function to partition the data based on features.
    * Model performance was assessed using silhouette analysis (`cluster::silhouette`) and visualization (`factoextra::fviz_cluster`, `seriation::dissplot`). Predictions on the test set used a custom `predict.kmeans` function. Confusion matrices were generated using `caret::confusionMatrix`.

2.  **Neural Network:**
    * A Neural Network model was trained and evaluated using the `caret` package's `train` function with `method="nnet"`. This typically interfaces with the `nnet` package, implementing a single-hidden-layer MLP.
    * Trained on the original, reduced (high correlation removed), and PCA-transformed datasets. Performance was evaluated using confusion matrices and ROC metrics within the `caret` framework.

## Technologies Used

* **Programming Languages:** Python, R
* **Python Libraries:**
    * `pandas` for data manipulation.
    * `numpy` for numerical computations.
    * `matplotlib` & `seaborn` for data visualization.
    * `scikit-learn` (`sklearn`) for data preprocessing (StandardScaler, train_test_split), PCA, Decision Trees, and model evaluation (metrics, model_selection).
    * `Keras` (with TensorFlow backend) for Neural Network implementation.
    * `pydotplus` for Decision Tree visualization.
* **R Libraries:**
    * `needs`, `readr`, `dplyr`, `ggplot2`, `corrplot`, `gridExtra`, `pROC`, `MASS`, `caTools`, `caret`, `caretEnsemble`, `reshape2`, `factoextra`, `psych`, `GGally`, `PerformanceAnalytics`, `cluster`, `seriation`, `NeuralNetTools`.

## Results & Evaluation

* **Metrics:** Models were evaluated using confusion matrices, Accuracy, Precision, Recall, F1-Score, ROC curves, and Area Under Curve (AUC).
* **Performance (Main focus on Python models as per report):**
    * Both Python model types (Decision Tree, Neural Network) showed good overall performance.
    * Neural Networks (Keras) generally achieved slightly higher accuracy and AUC compared to Decision Trees (sklearn) on the tested PCA datasets. The 6-feature NN reached ~96.5% accuracy and an AUC of 0.99, while the corresponding DT achieved ~91.6% accuracy and an AUC of 0.92.
    * Decision Trees exhibited significantly lower training times.
* **Cross-Validation:** 10-fold cross-validation was performed (shown for Python models) to assess robustness and generalization capability.
* **R Model Insights:** The R implementation provided comparisons between K-Means and Neural Networks (`nnet`), also exploring performance differences on original vs. reduced vs. PCA datasets.

