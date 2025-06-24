# =============================================================================
# Glioma Survival Prediction - Data Preparation Script
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
# Data Splitting Function
# =============================================================================

split_and_save_data <- function(data, train_ratio = 0.7, output_dir = "results") {
  # Split data into training and test sets
  cat("Splitting data into training (", train_ratio*100, "%) and test (", (1-train_ratio)*100, "%) sets...\n")
  
  # Create train/test split indices
  train_indices <- createDataPartition(data$days_to_death.demographic, p = train_ratio, list = FALSE)
  
  # Split the data
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Save training and test sets
  train_file <- paste0(output_dir, "/train_data.csv")
  test_file <- paste0(output_dir, "/test_data.csv")
  
  write.csv(train_data, train_file, row.names = FALSE)
  write.csv(test_data, test_file, row.names = FALSE)
  
  cat("Training set saved to:", train_file, "(", nrow(train_data), "samples)\n")
  cat("Test set saved to:", test_file, "(", nrow(test_data), "samples)\n")
  
  # Save split indices for reference
  indices_file <- paste0(output_dir, "/data_split_indices.csv")
  split_info <- data.frame(
    sample_id = 1:nrow(data),
    is_training = 1:nrow(data) %in% train_indices
  )
  write.csv(split_info, indices_file, row.names = FALSE)
  
  cat("Split indices saved to:", indices_file, "\n")
  
  # Print summary statistics
  cat("\nData Split Summary:\n")
  cat("==================\n")
  cat("Total samples:", nrow(data), "\n")
  cat("Training samples:", nrow(train_data), "\n")
  cat("Test samples:", nrow(test_data), "\n")
  cat("Training ratio:", round(nrow(train_data)/nrow(data), 3), "\n")
  
  # Print target variable statistics
  cat("\nTarget Variable Statistics:\n")
  cat("==========================\n")
  cat("Training set - Mean days to death:", round(mean(train_data$days_to_death.demographic), 1), "\n")
  cat("Training set - Median days to death:", round(median(train_data$days_to_death.demographic), 1), "\n")
  cat("Test set - Mean days to death:", round(mean(test_data$days_to_death.demographic), 1), "\n")
  cat("Test set - Median days to death:", round(median(test_data$days_to_death.demographic), 1), "\n")
  
  return(list(
    train_data = train_data,
    test_data = test_data,
    train_indices = train_indices,
    train_file = train_file,
    test_file = test_file
  ))
}

# =============================================================================
# Main Data Preparation Pipeline
# =============================================================================

main_data_preparation <- function() {
  # Main data preparation pipeline
  
  cat("Glioma Survival Prediction - Data Preparation\n")
  cat("============================================\n\n")
  
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
  
  # Split and save data
  cat("\nStep 3: Splitting and saving data...\n")
  split_results <- split_and_save_data(data_xcell)
  
  # Print final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("DATA PREPARATION COMPLETE\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- train_data.csv (training set)\n")
  cat("- test_data.csv (test set)\n")
  cat("- data_split_indices.csv (split information)\n")
  
  return(split_results)
}

# =============================================================================
# Run Data Preparation
# =============================================================================

if (!interactive()) {
  # Run the data preparation if script is executed directly
  results <- main_data_preparation()
} 