# Glioma Survival Prediction Analysis

This project analyzes clinical and molecular data from glioma patients to predict survival time using various machine learning methods.

## Project Overview

The analysis focuses on predicting survival time (days to death) in glioma patients using clinical and molecular features. The project implements multiple regression models and compares their performance in predicting patient survival.

## Data

The analysis uses two main datasets:
- `ClinicaGliomasMayo2025.xlsx`: Clinical data containing patient demographics and molecular markers
- `xCell_gene_tpm_Mayo2025.xlsx`: Gene expression data (currently commented out in the analysis)

### Key Features Used
- Clinical features (age, gender, race)
- Molecular markers (Chr_7_gain_Chr_10_loss, ATRX_status, EGFR_CNV, etc.)
- Tumor descriptors

## Analysis Methods

The project implements and compares several regression methods:
1. Linear Regression (`lm`)
2. Random Forest (`rf`)
3. k-Nearest Neighbors (`knn`)
4. Support Vector Machine with RBF kernel (`svmRadial`)
5. Gradient Boosting (`gbm`)

Each method is evaluated using:
- Leave-One-Out Cross-Validation (LOOCV)
- Root Mean Square Error (RMSE)
- Correlation between predicted and actual values

## Results

Results are saved in the `results` folder as PDF files:
- `predicted_vs_actual_survival_[method].pdf`: Scatter plots comparing predicted vs actual survival times for each method
- Each plot includes correlation coefficient and RMSE metrics

## Requirements

Required R packages:
```R
install.packages(c("readxl", "caret", "randomForest", "gbm", "kernlab", "e1071"))
```

## Usage

1. Ensure all required packages are installed
2. Place data files in the `data` directory
3. Run the analysis:
```R
Rscript run_experiments.R
```

## Directory Structure

```
.
├── data/
│   ├── ClinicaGliomasMayo2025.xlsx
│   └── xCell_gene_tpm_Mayo2025.xlsx
├── results/
│   └── predicted_vs_actual_survival_*.pdf
├── run_experiments.R
└── README.md
```

## Future Work

Potential improvements and extensions:
- Incorporate gene expression data
- Add more advanced feature selection methods
- Implement additional machine learning models
- Add survival analysis methods
- Include model interpretation and feature importance analysis

## License

[Add your license information here]

## Contact

[Add your contact information here]