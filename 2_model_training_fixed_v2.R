# =============================================================================
# Glioma Survival Prediction - Model Training Script (Revised)
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
output_dir <- "results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Set random seed for reproducibility
set.seed(123)

# =============================================================================
# Data Loading Functions
# =============================================================================

load_training_data <- function(file_path = file.path(output_dir, "train_data.csv")) {
  cat("Loading training data from:", file_path, "\n")
  train_data <- read.csv(file_path)
  
  if (!"days_to_death.demographic" %in% names(train_data)) {
    stop("Column 'days_to_death.demographic' not found in training data.")
  }
  
  cat("Training data loaded with", nrow(train_data), "samples and", ncol(train_data) - 1, "features\n")
  return(train_data)
}

load_test_data <- function(file_path = file.path(output_dir, "test_data.csv")) {
  cat("Loading test data from:", file_path, "\n")
  test_data <- read.csv(file_path)
  
  if (!"days_to_death.demographic" %in% names(test_data)) {
    stop("Column 'days_to_death.demographic' not found in test data.")
  }
  
  cat("Test data loaded with", nrow(test_data), "samples and", ncol(test_data) - 1, "features\n")
  return(test_data)
}

# =============================================================================
# Model Training Functions
# =============================================================================

setup_train_control <- function(method = "cv", number = 5) {
  trainControl(
    method = method,
    number = number,
    savePredictions = "final",
    returnResamp = "all"
  )
}

train_rf_model <- function(train_data, ctrl) {
  cat("Training Random Forest model on training data...\n")
  
  # Remove zero-variance predictors
  nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
  if (any(nzv$zeroVar)) {
    cat("Removing", sum(nzv$zeroVar), "predictors with zero variance...\n")
    train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  }
  
  model <- train(
    days_to_death.demographic ~ .,
    data = train_data,
    method = "rf",
    trControl = ctrl,
    importance = TRUE
  )
  
  cat("Model training completed\n")
  return(model)
}

calculate_calibration_and_predictions <- function(model, train_data, test_data) {
  cat("Calculating calibration parameters and test predictions...\n")
  
  # Use model$pred for true out-of-fold predictions if available
  if (!is.null(model$pred)) {
    # Use final model tuning parameters only
    best_tune <- model$bestTune
    pred_data <- model$pred
    for (param in names(best_tune)) {
      pred_data <- pred_data[pred_data[[param]] == best_tune[[param]], ]
    }
    cv_predictions <- pred_data$pred
    cv_actual <- pred_data$obs
  } else {
    cv_predictions <- predict(model, newdata = train_data)
    cv_actual <- train_data$days_to_death.demographic
    warning("Out-of-fold predictions not found; using fitted predictions instead.")
  }
  
  # Calibration model
  calib_lm <- lm(cv_actual ~ cv_predictions)
  alpha <- coef(calib_lm)["cv_predictions"]
  beta <- coef(calib_lm)["(Intercept)"]
  calib_r_squared <- summary(calib_lm)$r.squared
  
  # Predict on test set and apply calibration
  test_predictions <- predict(model, newdata = test_data)
  test_predictions <- alpha * test_predictions + beta
  test_actual <- test_data$days_to_death.demographic
  
  # Performance metrics
  correlation <- cor(test_actual, test_predictions)
  rmse <- sqrt(mean((test_actual - test_predictions)^2))
  mae <- mean(abs(test_actual - test_predictions))
  r_squared <- correlation^2
  
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
    alpha = alpha,
    beta = beta,
    calibration_r_squared = calib_r_squared,
    calibration_model = calib_lm
  ))
}

save_model_and_parameters <- function(results, output_dir = "results") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Save model
  model_file <- file.path(output_dir, "rf_model.rds")
  saveRDS(results$model, model_file)
  cat("Model saved to:", model_file, "\n")
  
  # Save calibration parameters
  calib_file <- file.path(output_dir, "calibration_parameters.rds")
  saveRDS(list(
    alpha = results$alpha,
    beta = results$beta,
    calibration_r_squared = results$calibration_r_squared,
    calibration_model = results$calibration_model
  ), calib_file)
  cat("Calibration parameters saved to:", calib_file, "\n")
  
  calib_csv <- file.path(output_dir, "calibration_parameters.csv")
  write.csv(data.frame(
    Parameter = c("alpha", "beta", "calibration_r_squared"),
    Value = c(results$alpha, results$beta, results$calibration_r_squared)
  ), calib_csv, row.names = FALSE)
  cat("Calibration parameters saved to:", calib_csv, "\n")
  
  # Save test predictions
  test_results_file <- file.path(output_dir, "test_predictions.csv")
  write.csv(data.frame(
    Actual = results$actual,
    Predicted = results$predictions,
    Residuals = results$actual - results$predictions
  ), test_results_file, row.names = FALSE)
  cat("Test predictions saved to:", test_results_file, "\n")
  
  # Save model summary
  model_info_file <- file.path(output_dir, "model_info.txt")
  sink(model_info_file)
  cat("Random Forest Model Information\n")
  cat("==============================\n")
  cat("Training date:", Sys.Date(), "\n")
  cat("Number of trees:", results$model$finalModel$ntree, "\n")
  cat("Variables per split (mtry):", results$model$finalModel$mtry, "\n")
  cat("Training samples:", nrow(results$model$trainingData), "\n")
  cat("Test samples:", length(results$actual), "\n")
  cat("Features used:", ncol(results$model$trainingData) - 1, "\n")
  cat("\nCalibration Parameters:\n")
  cat("Alpha:", results$alpha, "\n")
  cat("Beta:", results$beta, "\n")
  cat("Calibration R-squared:", results$calibration_r_squared, "\n")
  cat("\nTest Performance:\n")
  cat("Correlation:", results$correlation, "\n")
  cat("RMSE:", results$rmse, "\n")
  cat("MAE:", results$mae, "\n")
  cat("R-squared:", results$r_squared, "\n")
  sink()
  cat("Model information saved to:", model_info_file, "\n")
}

print_training_summary <- function(results) {
  cat("\nModel Training Summary:\n")
  cat("=======================\n")
  cat("Model: Random Forest\n")
  cat("Trees:", results$model$finalModel$ntree, "\n")
  cat("mtry:", results$model$finalModel$mtry, "\n")
  cat("Training samples:", nrow(results$model$trainingData), "\n")
  cat("Test samples:", length(results$actual), "\n")
  cat("Features:", ncol(results$model$trainingData) - 1, "\n")
  
  cat("\nCalibration:\n")
  cat("Alpha:", round(results$alpha, 4), "\n")
  cat("Beta:", round(results$beta, 4), "\n")
  cat("Calibration R²:", round(results$calibration_r_squared, 4), "\n")
  
  cat("\nTest Performance:\n")
  cat("Correlation:", round(results$correlation, 3), "\n")
  cat("RMSE:", round(results$rmse, 0), "days\n")
  cat("MAE:", round(results$mae, 0), "days\n")
  cat("R²:", round(results$r_squared, 3), "\n")
}

# =============================================================================
# Main Training Pipeline
# =============================================================================

main_model_training <- function() {
  cat("Glioma Survival Prediction - Model Training\n")
  cat("==========================================\n\n")
  
  # Step 1: Load data
  train_data <- load_training_data()
  test_data <- load_test_data()
  
  # Step 2: Train model
  ctrl <- setup_train_control()
  model <- train_rf_model(train_data, ctrl)
  
  # Step 3: Calibration and evaluation
  results <- calculate_calibration_and_predictions(model, train_data, test_data)
  
  # Step 4: Save results
  save_model_and_parameters(results, output_dir)
  
  # Step 5: Summary
  print_training_summary(results)
  
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat("MODEL TRAINING COMPLETE\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  cat("Results saved to:", output_dir, "\n")
}

# =============================================================================
# Run Model Training
# =============================================================================

if (!interactive()) {
  results <- main_model_training()
}
