# =============================================================================
# Glioma Survival Prediction - Data Preparation Script (FIXED)
# =============================================================================

# Install required packages (uncomment if needed)
# install.packages(c("readxl", "caret", "dplyr"), dependencies = TRUE)

# Load required libraries
library(readxl)
library(caret)
library(dplyr)

# =============================================================================
# Configuration
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results")
}

# Set random seed for reproducibility
set.seed(123)

# =============================================================================
# Data Loading Functions
# =============================================================================

load_clinical_data <- function(file_path = "data/ClinicaGliomasMayo2025.xlsx") {
  # Load clinical data from Excel file
  cat("Loading clinical data from:", file_path, "\n")
  read_excel(file_path)
}

load_xcell_data <- function(file_path = "data/xCell_gene_tpm_Mayo2025.xlsx") {
  # Load and reshape xCell gene expression data
  cat("Loading xCell data from:", file_path, "\n")
  xcell_data <- read_excel(file_path)
  
  # Reshape: make TCGACode a column and CellType as column names
  xcell_t <- as.data.frame(t(xcell_data[,-1]))
  CellTypeNames <- xcell_data$CellType
  colnames(xcell_t) <- CellTypeNames
  xcell_t$TCGACode <- colnames(xcell_data)[-1]
  xcell_t <- xcell_t %>% relocate(TCGACode)
  
  cat("Reshaped xCell data with", ncol(xcell_t)-1, "cell types\n")
  
  return(list(data = xcell_t, cell_types = CellTypeNames))
}

prepare_dataset <- function(clinica_data, xcell_data, cell_types) {
  # Prepare dataset for modeling
  cat("Merging clinical and xCell data...\n")
  
  # Merge clinical and xCell data
  merged_data <- left_join(
    clinica_data[, c("TCGACode", "days_to_death.demographic")], 
    xcell_data, 
    by = "TCGACode"
  )
  
  # Prepare data: use only CellTypeNames as predictors
  xcell_features <- as.character(cell_types)
  data_xcell <- merged_data[, c("days_to_death.demographic", xcell_features)]
  data_xcell <- na.omit(data_xcell)  # Remove rows with missing values
  
  cat("Final dataset prepared with", nrow(data_xcell), "samples and", ncol(data_xcell)-1, "features\n")
  
  return(data_xcell)
}

# =============================================================================
# Save Full Dataset (FIXED approach)
# =============================================================================

save_full_dataset <- function(data, output_dir = "results") {
  # Save the full dataset (no splitting)
  cat("Saving full dataset...\n")
  
  # Save full dataset
  full_data_file <- paste0(output_dir, "/full_data.csv")
  write.csv(data, full_data_file, row.names = FALSE)
  
  cat("Full dataset saved to:", full_data_file, "(", nrow(data), "samples)\n")
  
  # Print dataset statistics
  cat("\nDataset Summary:\n")
  cat("================\n")
  cat("Total samples:", nrow(data), "\n")
  cat("Number of features:", ncol(data) - 1, "\n")
  
  # Print target variable statistics
  cat("\nTarget Variable Statistics:\n")
  cat("==========================\n")
  cat("Mean days to death:", round(mean(data$days_to_death.demographic), 1), "\n")
  cat("Median days to death:", round(median(data$days_to_death.demographic), 1), "\n")
  cat("Min days to death:", round(min(data$days_to_death.demographic), 1), "\n")
  cat("Max days to death:", round(max(data$days_to_death.demographic), 1), "\n")
  cat("Standard deviation:", round(sd(data$days_to_death.demographic), 1), "\n")
  
  return(list(
    full_data = data,
    full_data_file = full_data_file
  ))
}

# =============================================================================
# Main Data Preparation Pipeline
# =============================================================================

main_data_preparation <- function() {
  # Main data preparation pipeline
  
  cat("Glioma Survival Prediction - Data Preparation (FIXED)\n")
  cat("====================================================\n\n")
  
  # Load data
  cat("Step 1: Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  
  # Prepare dataset
  cat("\nStep 2: Preparing dataset...\n")
  data_xcell <- prepare_dataset(
    clinica_data, 
    xcell_data_list$data, 
    xcell_data_list$cell_types
  )
  
  # Save full dataset (no splitting)
  cat("\nStep 3: Saving full dataset...\n")
  save_results <- save_full_dataset(data_xcell)
  
  # Print final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("DATA PREPARATION COMPLETE (FIXED)\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- full_data.csv (complete dataset)\n")
  cat("\nNote: This approach saves the full dataset to match the original methodology.\n")
  cat("The train/test split will be handled by caret during model training.\n")
  
  return(save_results)
}

# =============================================================================
# Run Data Preparation
# =============================================================================

if (!interactive()) {
  # Run the data preparation if script is executed directly
  results <- main_data_preparation()
} 