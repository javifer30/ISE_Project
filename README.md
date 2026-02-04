# Performance Bug Classification in Deep Learning Frameworks

## Project Overview
This project aims to automatically classify bug reports from various Deep Learning frameworks (TensorFlow, PyTorch, Keras, etc.) as either **performance bug-related** or **non-performance bug-related**. 

The repository implements and compares two approaches:
1.  **Baseline:** A Naive Bayes classifier using TF-IDF vectorization.
2.  **Solution:** A Support Vector Machine (SVM) classifier with hyperparameter tuning and enhanced preprocessing.

Statistical analysis is performed to evaluate whether the Solution significantly outperforms the Baseline.

## Repository Structure

```text
ISE_Project/
├── datasets/                   # CSV files containing bug reports (Title, Body, Class)
│   ├── caffe.csv
│   ├── keras.csv
│   ├── pytorch.csv
│   ├── tensorflow.csv
│   └── incubator-mxnet.csv
├── Baseline_Code.py            # Script for the Naive Bayes Baseline model
├── Solution_Javier.py          # Script for the SVM Solution model
├── StatisticalAnalysis_Javier.py # Script for comparing results (Plots & Wilcoxon test)
├── manual.pdf                  # User manual
├── replication.pdf             # Replication guide
└── requirements.pdf            # Project requirements
```

## Requirements
The project requires **Python 3.x** and the following libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `nltk`
*   `matplotlib`
*   `scipy`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn nltk matplotlib scipy
```

## Usage

### 1. Prepare Data
Ensure the dataset CSV file for your target project (e.g., `tensorflow.csv`) is accessible to the scripts. 
*Note: The scripts currently look for the CSV file in the same directory. You may need to move the desired file from `datasets/` to the root folder or modify the `path` variable in the scripts.*

### 2. Run Baseline
Open `Baseline_Code.py` and ensure the `project` variable matches your target dataset (default is `'tensorflow'`).
```python
project = 'tensorflow'  # Change to 'keras', 'pytorch', etc. as needed
```
Run the script:
```bash
python Baseline_Code.py
```
This generates `{project}_NB.csv` containing performance metrics.

### 3. Run Solution
Open `Solution_Javier.py`, check the `project` variable, and run:
```bash
python Solution_Javier.py
```
This generates `{project}_SVM.csv`.

### 4. Statistical Analysis
After running both the baseline and solution for the same project:
1.  Open `StatisticalAnalysis_Javier.py`.
2.  Set the `project` variable to match the one you simulated.
3.  Run the analysis:
    ```bash
    python StatisticalAnalysis_Javier.py
    ```
This will produce:
*   **Pareto Front Plot:** Visualizing F1 vs AUC trade-offs.
*   **Mean Metrics Plot:** Comparing average performance.
*   **P-values Plot:** Results of the Wilcoxon signed-rank test for significance.

## Methodology

### Data Preprocessing
Both models apply standard NLP preprocessing steps:
*   **HTML Removal:** Strips HTML tags.
*   **Emoji Removal:** Filters out unicode emojis.
*   **Stopword Removal:** Removes common English stopwords (NLTK).
*   **Text Cleaning:** Lowercasing and removing special characters.
*   **Punctuation Removal:** *Implemented in the Solution (SVM) only* to further reduce noise.

### Model Configuration
*   **Baseline (Naive Bayes):** 
    *   Uses `GaussianNB`.
    *   Evaluated over 30 random splits (80% train / 20% test).
*   **Solution (SVM):** 
    *   Uses `SVC` with RBF kernel and `class_weight='balanced'`.
    *   Hyperparameters (`C`, `gamma`) tuned via GridSearch (5-fold CV).
    *   Evaluated over 30 random splits (70% train / 30% test).

### Metrics
The models are evaluated using:
*   Accuracy
*   Precision (Macro)
*   Recall (Macro)
*   F1 Score (Macro)
*   AUC (Area Under Curve)