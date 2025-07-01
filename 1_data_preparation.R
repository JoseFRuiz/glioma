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
  dir.create("results", recursive = TRUE)
}

# Set random seed for reproducibility
set.seed(123)

# =============================================================================
# Data Loading Functions
# =============================================================================

load_clinical_data <- function(file_path = "data/ClinicaGliomasMayo2025.xlsx") {
  cat("Loading clinical data from:", file_path, "\n")
  data <- read_excel(file_path)
  
  # Clean TCGACode
  if (!"TCGACode" %in% names(data)) stop("Missing 'TCGACode' column in clinical data.")
  data$TCGACode <- trimws(as.character(data$TCGACode))
  
  # Check required outcome column
  if (!"days_to_death.demographic" %in% names(data)) {
    stop("Column 'days_to_death.demographic' not found in clinical data.")
  }
  
  return(data)
}

load_xcell_data <- function(file_path = "data/xCell_gene_tpm_Mayo2025.xlsx") {
  cat("Loading xCell data from:", file_path, "\n")
  xcell_data <- read_excel(file_path)
  
  # Check expected structure
  if (!"CellType" %in% names(xcell_data)) {
    stop("Missing 'CellType' column in xCell data.")
  }
  
  # Transpose data: columns are TCGACode samples
  xcell_t <- as.data.frame(t(xcell_data[,-1]))
  CellTypeNames <- xcell_data$CellType
  colnames(xcell_t) <- CellTypeNames
  xcell_t$TCGACode <- colnames(xcell_data)[-1]
  xcell_t$TCGACode <- trimws(as.character(xcell_t$TCGACode))
  xcell_t <- xcell_t %>% relocate(TCGACode)
  
  cat("Reshaped xCell data with", ncol(xcell_t) - 1, "cell types\n")
  return(list(data = xcell_t, cell_types = CellTypeNames))
}

prepare_dataset <- function(clinica_data, xcell_data, cell_types) {
  cat("Merging clinical and xCell data...\n")
  
  merged_data <- left_join(
    clinica_data[, c("TCGACode", "days_to_death.demographic")],
    xcell_data,
    by = "TCGACode"
  )
  
  xcell_features <- as.character(cell_types)
  
  if (!all(xcell_features %in% colnames(merged_data))) {
    warning("Some expected xCell features were not found in merged data.")
    xcell_features <- intersect(xcell_features, colnames(merged_data))
  }
  
  data_xcell <- merged_data[, c("days_to_death.demographic", xcell_features)]
  data_xcell <- na.omit(data_xcell)
  
  if (nrow(data_xcell) == 0) {
    stop("No data left after removing missing values.")
  }
  
  cat("Final dataset prepared with", nrow(data_xcell), "samples and", ncol(data_xcell) - 1, "features\n")
  return(data_xcell)
}

# =============================================================================
# Data Splitting and Saving
# =============================================================================

split_and_save_data <- function(data, train_ratio = 0.7, output_dir = "results") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  cat("Splitting data into training (", train_ratio * 100, "%) and test (", (1 - train_ratio) * 100, "%) sets...\n")
  train_indices <- createDataPartition(data$days_to_death.demographic, p = train_ratio, list = FALSE)
  
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Save datasets
  train_file <- file.path(output_dir, "train_data.csv")
  test_file <- file.path(output_dir, "test_data.csv")
  write.csv(train_data, train_file, row.names = FALSE)
  write.csv(test_data, test_file, row.names = FALSE)
  
  cat("Training set saved to:", train_file, "(", nrow(train_data), "samples)\n")
  cat("Test set saved to:", test_file, "(", nrow(test_data), "samples)\n")
  
  # Save split metadata
  indices_file <- file.path(output_dir, "data_split_indices.csv")
  split_info <- data.frame(
    sample_index = 1:nrow(data),
    is_training = 1:nrow(data) %in% train_indices
  )
  write.csv(split_info, indices_file, row.names = FALSE)
  cat("Split indices saved to:", indices_file, "\n")
  
  # Print basic stats
  cat("\nData Split Summary:\n")
  cat("==================\n")
  cat("Total samples:", nrow(data), "\n")
  cat("Training samples:", nrow(train_data), "\n")
  cat("Test samples:", nrow(test_data), "\n")
  cat("Training ratio:", round(nrow(train_data) / nrow(data), 3), "\n")
  
  # Target variable stats
  cat("\nTarget Variable Statistics:\n")
  cat("==========================\n")
  cat("Training set - Mean:", round(mean(train_data$days_to_death.demographic), 1), "\n")
  cat("Training set - Median:", round(median(train_data$days_to_death.demographic), 1), "\n")
  cat("Test set - Mean:", round(mean(test_data$days_to_death.demographic), 1), "\n")
  cat("Test set - Median:", round(median(test_data$days_to_death.demographic), 1), "\n")
  
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
  cat("Glioma Survival Prediction - Data Preparation\n")
  cat("============================================\n\n")
  
  # Step 1: Load data
  cat("Step 1: Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  
  # Step 2: Prepare dataset
  cat("\nStep 2: Preparing dataset...\n")
  data_xcell <- prepare_dataset(
    clinica_data,
    xcell_data_list$data,
    xcell_data_list$cell_types
  )
  
  # Step 3: Split and save data
  cat("\nStep 3: Splitting and saving data...\n")
  split_results <- split_and_save_data(data_xcell)

  # =============================
  # Add interaction features to train and test sets
  # =============================
  cat("\nAdding interaction features to train and test sets...\n")
  train_data <- split_results$train_data
  test_data <- split_results$test_data

  # Compute correlations on the full dataset (excluding target)
  features <- data_xcell[, -1]
  target <- data_xcell$days_to_death.demographic
  feature_cors <- sapply(features, function(x) cor(x, target, use = "complete.obs"))
  feature_cors_abs <- abs(feature_cors)
  top_features <- names(sort(feature_cors_abs, decreasing = TRUE)[1:3])

  # Create interaction features (all pairwise for top 3)
  interaction_names <- c()
  for (i in 1:(length(top_features)-1)) {
    for (j in (i+1):length(top_features)) {
      feat1 <- top_features[i]
      feat2 <- top_features[j]
      interaction_name <- paste0(feat1, "_x_", feat2)
      train_data[[interaction_name]] <- train_data[[feat1]] * train_data[[feat2]]
      test_data[[interaction_name]] <- test_data[[feat1]] * test_data[[feat2]]
      interaction_names <- c(interaction_names, interaction_name)
    }
  }
  cat("Added interaction features:", paste(interaction_names, collapse=", "), "\n")

  # Save interaction feature names for downstream scripts
  writeLines(interaction_names, "results/interaction_features.txt")

  # Save updated train and test sets with interaction features
  write.csv(train_data, split_results$train_file, row.names = FALSE)
  write.csv(test_data, split_results$test_file, row.names = FALSE)
  cat("Updated training and test sets saved with interaction features.\n")

  # Final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("DATA PREPARATION COMPLETE\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- train_data.csv (training set)\n")
  cat("- test_data.csv (test set)\n")
  cat("- data_split_indices.csv (split information)\n")
  cat("- interaction_features.txt (interaction feature names)\n")
  cat("\nNote: Data has been explicitly split to avoid bias in model training.\n")
  
  return(split_results)
}

# =============================================================================
# Run Data Preparation
# =============================================================================

if (!interactive()) {
  results <- main_data_preparation()
}