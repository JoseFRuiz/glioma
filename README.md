# Glioma Survival Prediction Analysis

This project analyzes glioma survival prediction using xCell gene expression data and multiple machine learning models.

## Project Structure

```
glioma/
├── data/
│   ├── ClinicaGliomasMayo2025.xlsx          # Clinical data
│   └── xCell_gene_tpm_Mayo2025.xlsx         # Gene expression data
├── results/                                  # Generated plots and results
├── run_experiments.R                         # Main analysis script
├── draft.r                                   # Draft analysis
└── README.md                                 # This file
```

## Code Organization Improvements

The `run_experiments.R` script has been reorganized with the following improvements:

### 1. **Modular Structure**
- **Configuration Section**: Centralized constants and settings
- **Data Loading Functions**: Modular functions for loading and preprocessing data
- **Model Training Functions**: Reusable functions for model training and evaluation
- **Variable Importance Analysis**: Functions to analyze feature relevance
- **Main Pipeline**: Orchestrates the entire analysis workflow

### 2. **Eliminated Redundancy**
- Removed unused clinical features analysis code
- Consolidated duplicate method definitions
- Unified variable naming conventions
- Removed redundant data preparation steps

### 3. **Key Functions**

#### Data Processing
- `load_clinical_data()`: Loads clinical data from Excel
- `load_xcell_data()`: Loads and reshapes xCell gene expression data
- `prepare_xcell_dataset()`: Merges and prepares final dataset

#### Model Training
- `setup_train_control()`: Configures training parameters
- `train_and_evaluate_model()`: Trains and evaluates individual models
- `create_scatter_plot()`: Generates performance visualization
- `print_results_summary()`: Displays model performance summary

#### Variable Importance Analysis
- `analyze_variable_importance()`: Extracts and visualizes feature importance for tree-based models (RF, GBM)
- `create_importance_comparison()`: Compares variable importance across different models

#### Main Pipeline
- `main_analysis()`: Orchestrates the complete analysis workflow

### 4. **Benefits of Reorganization**
- **Maintainability**: Functions are self-contained and reusable
- **Readability**: Clear separation of concerns with logical sections
- **Extensibility**: Easy to add new models or modify existing ones
- **Debugging**: Isolated functions make troubleshooting easier
- **Documentation**: Each function has clear purpose and parameters

## Variable Importance Analysis

The script now includes comprehensive variable importance analysis for tree-based models:

### **Supported Models**
- **Random Forest (rf)**: Uses `varImp()` from the `randomForest` package
- **Gradient Boosting (gbm)**: Uses `varImp()` from the `gbm` package

### **Outputs Generated**
1. **Individual Model Plots**: `variable_importance_rf.pdf` and `variable_importance_gbm.pdf`
2. **Comparison Analysis**: 
   - `variable_importance_comparison.pdf` - Visual comparison of top 30 variables
   - `variable_importance_comparison.csv` - Complete importance scores for all variables
   - `top_50_important_variables.csv` - Top 50 most important variables
3. **Console Output**: Top 20 most important variables for each model

### **Key Insights**
- Identifies which xCell features (cell types) are most predictive of survival
- Compares importance rankings across different tree-based models
- Provides average importance scores for robust feature selection
- Enables biological interpretation of cell type contributions to survival prediction

## Scale Comparison Analysis

The script now addresses scale mismatch issues and provides multiple approaches:

### **Problem Solved**
- **Original Issue**: Models trained on log-transformed data showed good correlation but poor scale matching when back-transformed
- **Solution**: Multiple evaluation approaches to compare performance, including bias correction

### **Approaches Implemented**

#### **1. Log-Transformed Approach (Original)**
- Trains models on log-transformed survival data
- Evaluates performance in both log-scale and original scale
- Provides two sets of plots for each model

#### **2. Original Scale Approach (Alternative)**
- Trains models directly on original survival data (no log transformation)
- Compares performance with log-transformed approach
- Helps identify which approach works better for your data

#### **3. Bias Correction (New)**
- **Problem**: When exponentiating log-transformed predictions, E[exp(X)] ≠ exp(E[X])
- **Solution**: Applies bias correction using residual variance: `exp(pred + var/2)`
- **Result**: Much better scale matching in original units

### **Outputs Generated**
1. **Dual-Scale Plots**: Each model now generates two plots:
   - `predicted_vs_actual_log_scale_[method].pdf` - Log-scale evaluation
   - `predicted_vs_actual_original_scale_[method].pdf` - Original scale evaluation with bias correction

2. **Scale Comparison**: 
   - `scale_comparison_rf.pdf` - Side-by-side comparison showing bias correction effect

3. **Enhanced Metrics**: Performance reported in both scales:
   - Log-scale: Correlation and RMSE in log(days)
   - Original scale: Correlation and RMSE in actual days (both raw and bias-corrected)

### **Bias Correction Details**
The bias correction addresses the fundamental issue with log-transformed regression:
- **Raw back-transformation**: `exp(prediction)` - often underestimates true values
- **Bias-corrected**: `exp(prediction + residual_variance/2)` - accounts for the fact that E[exp(X)] = exp(μ + σ²/2) for log-normal distributions

### **Benefits**
- **Accurate Evaluation**: See how models perform in both transformed and original scales
- **Better Interpretation**: Understand if log transformation helps or hurts performance
- **Robust Comparison**: Compare different modeling approaches systematically
- **Biological Relevance**: Original scale metrics are more interpretable for clinical applications
- **Fixed Scale Mismatch**: Bias correction provides much better predictions in original units

## Usage

To run the analysis:

```r
source("run_experiments.R")
```

The script will:
1. Load clinical and gene expression data
2. Prepare the dataset for modeling
3. Train multiple machine learning models (lm, rf, knn, svmRadial, gbm)
4. Generate performance plots in the `results/` directory
5. Print a summary of model performance

## Dependencies

Required R packages:
- `readxl`: For reading Excel files
- `caret`: For machine learning workflows
- `randomForest`: For random forest models
- `gbm`: For gradient boosting models
- `kernlab`: For SVM models
- `e1071`: For additional ML utilities
- `dplyr`: For data manipulation

## Results

The analysis generates:
- Performance plots for each model in `results/`
- Console output with correlation and RMSE metrics
- Log-transformed survival predictions using xCell features

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

[Add your contact information here]