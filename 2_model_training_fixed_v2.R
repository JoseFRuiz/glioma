# =============================================================================
# Glioma Survival Prediction - Model Training Script (FIXED V2)
# =============================================================================

# Install required packages (uncomment if needed)
# install.packages(c("caret", "randomForest", "dplyr"), dependencies = TRUE)

# Load required libraries
library(caret)
library(randomForest)
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
# Data Loading Function
# =============================================================================

load_full_data <- function(file_path = "results/full_data.csv") {
  # Load full dataset from CSV file
  cat("Loading full dataset from:", file_path, "\n")
  full_data <- read.csv(file_path)
  
  cat("Full dataset loaded with", nrow(full_data), "samples and", ncol(full_data)-1, "features\n")
  
  return(full_data)
}

# =============================================================================
# Model Training Functions (EXACTLY like original rf_classifier.R)
# =============================================================================

setup_train_control <- function(method = "boot", number = 1, p = 0.7) {
  # Set up training control parameters (exactly like original)
  trainControl(
    method = method,
    number = number,
    p = p,
    savePredictions = TRUE,
    returnResamp = "all"
  )
}

train_rf_model <- function(data, ctrl) {
  # Train Random Forest model and evaluate performance (exactly like original)
  cat("Training Random Forest model on full dataset...\n")
  
  # Train model on full dataset (same as original)
  model <- train(
    days_to_death.demographic ~ .,
    data = data,
    method = "rf",
    trControl = ctrl,
    importance = TRUE  # Enable variable importance
  )
  
  cat("Model training completed\n")
  
  return(model)
}

calculate_calibration_and_predictions <- function(model, data) {
  # Calculate calibration parameters and test predictions (exactly like original)
  cat("Calculating calibration parameters and test predictions...\n")
  
  # Get test set indices (same as original)
  train_indices <- model$control$index[[1]]
  test_indices <- setdiff(1:nrow(data), train_indices)
  
  cat("Training samples:", length(train_indices), "\n")
  cat("Test samples:", length(test_indices), "\n")
  
  # Get predictions for training data (same as original)
  train_predictions <- predict(model, newdata = data[train_indices,])
  train_actual <- data$days_to_death.demographic[train_indices]
  
  # Fit linear model: actual = alpha * prediction + beta (same as original)
  calib_lm <- lm(train_actual ~ train_predictions)
  
  # Extract coefficients
  alpha <- calib_lm$coefficients["train_predictions"]
  beta <- calib_lm$coefficients["(Intercept)"]
  
  # Calculate R-squared of calibration
  calib_r_squared <- summary(calib_lm)$r.squared
  
  # Apply calibration to test predictions (same as original)
  test_predictions <- predict(model, newdata = data[test_indices,])
  test_predictions <- alpha * test_predictions + beta
  test_actual <- data$days_to_death.demographic[test_indices]
  
  # Calculate performance metrics (same as original)
  correlation <- cor(test_actual, test_predictions)
  rmse <- sqrt(mean((test_actual - test_predictions)^2))
  mae <- mean(abs(test_actual - test_predictions))
  r_squared <- cor(test_actual, test_predictions)^2
  
  cat("Calibration parameters calculated:\n")
  cat("Alpha (slope):", round(alpha, 4), "\n")
  cat("Beta (intercept):", round(beta, 4), "\n")
  cat("Calibration R-squared:", round(calib_r_squared, 4), "\n")
  
  cat("Test performance:\n")
  cat("Correlation:", round(correlation, 3), "\n")
  cat("RMSE:", round(rmse, 0), "days\n")
  cat("MAE:", round(mae, 0), "days\n")
  cat("R-squared:", round(r_squared, 3), "\n")
  
  return(list(
    model = model,
    predictions = test_predictions,
    actual = test_actual,
    correlation = correlation,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    test_indices = test_indices,
    alpha = alpha,
    beta = beta,
    calibration_r_squared = calib_r_squared,
    calibration_model = calib_lm
  ))
}

save_model_and_parameters <- function(results, output_dir = "results") {
  # Save the trained model and calibration parameters
  
  # Save the Random Forest model
  model_file <- paste0(output_dir, "/rf_model.rds")
  saveRDS(results$model, model_file)
  cat("Model saved to:", model_file, "\n")
  
  # Save calibration parameters
  calib_file <- paste0(output_dir, "/calibration_parameters.rds")
  calibration_params <- list(
    alpha = results$alpha,
    beta = results$beta,
    calibration_r_squared = results$calibration_r_squared,
    calibration_model = results$calibration_model
  )
  saveRDS(calibration_params, calib_file)
  cat("Calibration parameters saved to:", calib_file, "\n")
  
  # Save calibration parameters as CSV for easy access
  calib_csv <- paste0(output_dir, "/calibration_parameters.csv")
  calib_df <- data.frame(
    Parameter = c("alpha", "beta", "calibration_r_squared"),
    Value = c(results$alpha, results$beta, results$calibration_r_squared)
  )
  write.csv(calib_df, calib_csv, row.names = FALSE)
  cat("Calibration parameters saved to:", calib_csv, "\n")
  
  # Save test predictions and actual values
  test_results_file <- paste0(output_dir, "/test_predictions.csv")
  test_df <- data.frame(
    Actual = results$actual,
    Predicted = results$predictions,
    Residuals = results$actual - results$predictions
  )
  write.csv(test_df, test_results_file, row.names = FALSE)
  cat("Test predictions saved to:", test_results_file, "\n")
  
  # Save model information
  model_info_file <- paste0(output_dir, "/model_info.txt")
  sink(model_info_file)
  cat("Random Forest Model Information\n")
  cat("==============================\n")
  cat("Training date:", Sys.Date(), "\n")
  cat("Number of trees:", results$model$finalModel$ntree, "\n")
  cat("Number of variables tried at each split:", results$model$finalModel$mtry, "\n")
  cat("Total samples:", nrow(results$model$trainingData), "\n")
  cat("Training samples:", length(results$model$control$index[[1]]), "\n")
  cat("Test samples:", length(results$test_indices), "\n")
  cat("Number of features:", ncol(results$model$trainingData) - 1, "\n")
  cat("\nCalibration Parameters:\n")
  cat("Alpha (slope):", results$alpha, "\n")
  cat("Beta (intercept):", results$beta, "\n")
  cat("Calibration R-squared:", results$calibration_r_squared, "\n")
  cat("\nTest Performance:\n")
  cat("Correlation:", results$correlation, "\n")
  cat("RMSE:", results$rmse, "days\n")
  cat("MAE:", results$mae, "days\n")
  cat("R-squared:", results$r_squared, "\n")
  sink()
  
  cat("Model information saved to:", model_info_file, "\n")
}

print_training_summary <- function(results) {
  # Print summary of training results
  cat("\nModel Training Summary:\n")
  cat("======================\n")
  cat("Model type: Random Forest\n")
  cat("Number of trees:", results$model$finalModel$ntree, "\n")
  cat("Number of variables tried at each split:", results$model$finalModel$mtry, "\n")
  cat("Total samples:", nrow(results$model$trainingData), "\n")
  cat("Training samples:", length(results$model$control$index[[1]]), "\n")
  cat("Test samples:", length(results$test_indices), "\n")
  cat("Number of features:", ncol(results$model$trainingData) - 1, "\n")
  
  cat("\nCalibration Parameters:\n")
  cat("Alpha (slope):", round(results$alpha, 4), "\n")
  cat("Beta (intercept):", round(results$beta, 4), "\n")
  cat("Calibration R-squared:", round(results$calibration_r_squared, 4), "\n")
  
  cat("\nTest Performance:\n")
  cat("Correlation:", round(results$correlation, 3), "\n")
  cat("RMSE:", round(results$rmse, 0), "days\n")
  cat("MAE:", round(results$mae, 0), "days\n")
  cat("R-squared:", round(results$r_squared, 3), "\n")
}

# =============================================================================
# Main Training Pipeline
# =============================================================================

main_model_training <- function() {
  # Main model training pipeline
  
  cat("Glioma Survival Prediction - Model Training (FIXED V2)\n")
  cat("=====================================================\n\n")
  
  # Load full data
  cat("Step 1: Loading full dataset...\n")
  full_data <- load_full_data()
  
  # Set up training control
  cat("\nStep 2: Setting up training control...\n")
  ctrl <- setup_train_control()
  
  # Train Random Forest model
  cat("\nStep 3: Training Random Forest model...\n")
  model <- train_rf_model(full_data, ctrl)
  
  # Calculate calibration parameters and test predictions
  cat("\nStep 4: Calculating calibration and test predictions...\n")
  results <- calculate_calibration_and_predictions(model, full_data)
  
  # Save model and parameters
  cat("\nStep 5: Saving model and parameters...\n")
  save_model_and_parameters(results)
  
  # Print training summary
  print_training_summary(results)
  
  # Print final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("MODEL TRAINING COMPLETE (FIXED V2)\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- rf_model.rds (trained model)\n")
  cat("- calibration_parameters.rds (calibration parameters)\n")
  cat("- calibration_parameters.csv (calibration parameters)\n")
  cat("- test_predictions.csv (test predictions and actual values)\n")
  cat("- model_info.txt (model information)\n")
  
  return(results)
}

# =============================================================================
# Run Model Training
# =============================================================================

if (!interactive()) {
  # Run the model training if script is executed directly
  results <- main_model_training()
} 