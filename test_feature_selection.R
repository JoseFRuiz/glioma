# =============================================================================
# Test Feature Selection: Remove Negative Importance Variables
# =============================================================================

# Load required libraries
library(readxl)
library(caret)
library(randomForest)
library(dplyr)

# Source the main analysis functions
source("rf_classifier.R")

# Function to load and prepare data (copied from main script)
load_clinical_data <- function(file_path = "data/ClinicaGliomasMayo2025.xlsx") {
  read_excel(file_path)
}

load_xcell_data <- function(file_path = "data/xCell_gene_tpm_Mayo2025.xlsx") {
  xcell_data <- read_excel(file_path)
  xcell_t <- as.data.frame(t(xcell_data[,-1]))
  CellTypeNames <- xcell_data$CellType
  colnames(xcell_t) <- CellTypeNames
  xcell_t$TCGACode <- colnames(xcell_data)[-1]
  xcell_t <- xcell_t %>% relocate(TCGACode)
  return(list(data = xcell_t, cell_types = CellTypeNames))
}

prepare_dataset <- function(clinica_data, xcell_data, cell_types) {
  merged_data <- left_join(
    clinica_data[, c("TCGACode", "days_to_death.demographic")], 
    xcell_data, 
    by = "TCGACode"
  )
  xcell_features <- as.character(cell_types)
  data_xcell <- merged_data[, c("days_to_death.demographic", xcell_features)]
  data_xcell <- na.omit(data_xcell)
  return(data_xcell)
}

# Function to train model with specific features
train_model_with_features <- function(data, features_to_use, ctrl) {
  set.seed(123)
  
  # Select only the specified features
  selected_data <- data[, c("days_to_death.demographic", features_to_use)]
  
  # Train model
  model <- train(
    days_to_death.demographic ~ .,
    data = selected_data,
    method = "rf",
    trControl = ctrl,
    importance = TRUE
  )
  
  # Get test set indices
  train_indices <- model$control$index[[1]]
  test_indices <- setdiff(1:nrow(selected_data), train_indices)
  
  # Get predictions for test set
  test_predictions <- predict(model, newdata = selected_data[test_indices,])
  test_actual <- selected_data$days_to_death.demographic[test_indices]
  
  # Calculate performance metrics
  correlation <- cor(test_actual, test_predictions)
  rmse <- sqrt(mean((test_actual - test_predictions)^2))
  mae <- mean(abs(test_actual - test_predictions))
  r_squared <- cor(test_actual, test_predictions)^2
  
  return(list(
    model = model,
    predictions = test_predictions,
    actual = test_actual,
    correlation = correlation,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    n_features = length(features_to_use)
  ))
}

# Main comparison function
compare_models <- function() {
  cat("Feature Selection Test: Removing Negative Importance Variables\n")
  cat("=============================================================\n\n")
  
  # Load data
  cat("Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  
  # Prepare full dataset
  cat("Preparing dataset...\n")
  data_xcell <- prepare_dataset(
    clinica_data, 
    xcell_data_list$data, 
    xcell_data_list$cell_types
  )
  
  cat("Dataset prepared with", nrow(data_xcell), "samples and", ncol(data_xcell)-1, "features\n\n")
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # 1. Train model with ALL features (baseline)
  cat("1. Training model with ALL features...\n")
  all_features <- colnames(data_xcell)[-1]  # Exclude target variable
  baseline_results <- train_model_with_features(data_xcell, all_features, ctrl)
  
  cat("   Baseline Results:\n")
  cat("   - Features:", baseline_results$n_features, "\n")
  cat("   - Correlation:", round(baseline_results$correlation, 3), "\n")
  cat("   - RMSE:", round(baseline_results$rmse, 0), "days\n")
  cat("   - MAE:", round(baseline_results$mae, 0), "days\n")
  cat("   - R²:", round(baseline_results$r_squared, 3), "\n\n")
  
  # 2. Get importance scores from baseline model
  cat("2. Analyzing variable importance...\n")
  importance_data <- varImp(baseline_results$model$finalModel)
  imp_df <- data.frame(
    Variable = rownames(importance_data),
    Importance = importance_data[, 1]
  )
  imp_df <- imp_df[order(-imp_df$Importance), ]
  
  # Separate positive and negative importance variables
  positive_vars <- imp_df$Variable[imp_df$Importance > 0]
  negative_vars <- imp_df$Variable[imp_df$Importance <= 0]
  
  cat("   Positive importance variables:", length(positive_vars), "\n")
  cat("   Negative importance variables:", length(negative_vars), "\n\n")
  
  # 3. Train model with ONLY positive importance features
  cat("3. Training model with POSITIVE importance features only...\n")
  positive_results <- train_model_with_features(data_xcell, positive_vars, ctrl)
  
  cat("   Positive Features Results:\n")
  cat("   - Features:", positive_results$n_features, "\n")
  cat("   - Correlation:", round(positive_results$correlation, 3), "\n")
  cat("   - RMSE:", round(positive_results$rmse, 0), "days\n")
  cat("   - MAE:", round(positive_results$mae, 0), "days\n")
  cat("   - R²:", round(positive_results$r_squared, 3), "\n\n")
  
  # 4. Compare results
  cat("4. Performance Comparison:\n")
  cat("   =======================\n")
  
  # Calculate improvements
  cor_improvement <- positive_results$correlation - baseline_results$correlation
  rmse_improvement <- baseline_results$rmse - positive_results$rmse
  mae_improvement <- baseline_results$mae - positive_results$mae
  r2_improvement <- positive_results$r_squared - baseline_results$r_squared
  
  cat("   Metric          | All Features | Positive Only | Improvement\n")
  cat("   ----------------|-------------|---------------|-------------\n")
  cat(sprintf("   Correlation      | %11.3f | %13.3f | %+11.3f\n", 
              baseline_results$correlation, positive_results$correlation, cor_improvement))
  cat(sprintf("   RMSE (days)      | %11.0f | %13.0f | %+11.0f\n", 
              baseline_results$rmse, positive_results$rmse, rmse_improvement))
  cat(sprintf("   MAE (days)       | %11.0f | %13.0f | %+11.0f\n", 
              baseline_results$mae, positive_results$mae, mae_improvement))
  cat(sprintf("   R²               | %11.3f | %13.3f | %+11.3f\n", 
              baseline_results$r_squared, positive_results$r_squared, r2_improvement))
  cat(sprintf("   Features         | %11d | %13d | %+11d\n", 
              baseline_results$n_features, positive_results$n_features, 
              positive_results$n_features - baseline_results$n_features))
  
  # 5. Save results
  results_summary <- data.frame(
    Model = c("All Features", "Positive Features Only"),
    N_Features = c(baseline_results$n_features, positive_results$n_features),
    Correlation = c(baseline_results$correlation, positive_results$correlation),
    RMSE = c(baseline_results$rmse, positive_results$rmse),
    MAE = c(baseline_results$mae, positive_results$mae),
    R_squared = c(baseline_results$r_squared, positive_results$r_squared)
  )
  
  write.csv(results_summary, "results/feature_selection_comparison.csv", row.names = FALSE)
  write.csv(data.frame(Variable = positive_vars), "results/selected_positive_features.csv", row.names = FALSE)
  write.csv(data.frame(Variable = negative_vars), "results/removed_negative_features.csv", row.names = FALSE)
  
  cat("\n5. Results saved to 'results' directory:\n")
  cat("   - feature_selection_comparison.csv\n")
  cat("   - selected_positive_features.csv\n")
  cat("   - removed_negative_features.csv\n\n")
  
  # 6. Conclusion
  cat("6. Conclusion:\n")
  cat("   ============\n")
  if (positive_results$correlation > baseline_results$correlation) {
    cat("   ✓ Removing negative importance variables IMPROVED model performance!\n")
    cat("   ✓ The model with only positive features is better.\n")
  } else {
    cat("   ✗ Removing negative importance variables did not improve performance.\n")
    cat("   ✗ Consider keeping all features or trying different selection methods.\n")
  }
  
  return(list(
    baseline = baseline_results,
    positive_only = positive_results,
    positive_vars = positive_vars,
    negative_vars = negative_vars
  ))
}

# Run the comparison
if (!interactive()) {
  results <- compare_models()
} 