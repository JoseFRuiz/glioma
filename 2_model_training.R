# =============================================================================
# Glioma Survival Prediction - Model Training Script
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

load_training_data <- function(file_path = "results/train_data.csv") {
  # Load training data from CSV file
  cat("Loading training data from:", file_path, "\n")
  train_data <- read.csv(file_path)
  
  cat("Training data loaded with", nrow(train_data), "samples and", ncol(train_data)-1, "features\n")
  
  return(train_data)
}

# =============================================================================
# Model Training Functions
# =============================================================================

setup_train_control <- function(method = "boot", number = 1, p = 0.7) {
  # Set up training control parameters
  trainControl(
    method = method,
    number = number,
    p = p,
    savePredictions = TRUE,
    returnResamp = "all"
  )
}

train_rf_model <- function(train_data, ctrl) {
  # Train Random Forest model
  cat("Training Random Forest model...\n")
  
  # Train model on training data
  model <- train(
    days_to_death.demographic ~ .,
    data = train_data,
    method = "rf",
    trControl = ctrl,
    importance = TRUE  # Enable variable importance
  )
  
  cat("Model training completed\n")
  
  return(model)
}

calculate_calibration_parameters <- function(model, train_data) {
  # Calculate alpha and beta parameters for linear calibration
  cat("Calculating calibration parameters...\n")
  
  # Get training indices from the model
  train_indices <- model$control$index[[1]]
  
  # Get predictions on training data
  train_predictions <- predict(model, newdata = train_data[train_indices,])
  train_actual <- train_data$days_to_death.demographic[train_indices]
  
  # Fit linear model: actual = alpha * prediction + beta
  calib_lm <- lm(train_actual ~ train_predictions)
  
  # Extract coefficients
  alpha <- calib_lm$coefficients["train_predictions"]
  beta <- calib_lm$coefficients["(Intercept)"]
  
  # Calculate R-squared of calibration
  calib_r_squared <- summary(calib_lm)$r.squared
  
  cat("Calibration parameters calculated:\n")
  cat("Alpha (slope):", round(alpha, 4), "\n")
  cat("Beta (intercept):", round(beta, 4), "\n")
  cat("Calibration R-squared:", round(calib_r_squared, 4), "\n")
  
  return(list(
    alpha = alpha,
    beta = beta,
    calibration_r_squared = calib_r_squared,
    calibration_model = calib_lm
  ))
}

save_model_and_parameters <- function(model, calibration_params, output_dir = "results") {
  # Save the trained model and calibration parameters
  
  # Save the Random Forest model
  model_file <- paste0(output_dir, "/rf_model.rds")
  saveRDS(model, model_file)
  cat("Model saved to:", model_file, "\n")
  
  # Save calibration parameters
  calib_file <- paste0(output_dir, "/calibration_parameters.rds")
  saveRDS(calibration_params, calib_file)
  cat("Calibration parameters saved to:", calib_file, "\n")
  
  # Save calibration parameters as CSV for easy access
  calib_csv <- paste0(output_dir, "/calibration_parameters.csv")
  calib_df <- data.frame(
    Parameter = c("alpha", "beta", "calibration_r_squared"),
    Value = c(calibration_params$alpha, calibration_params$beta, calibration_params$calibration_r_squared)
  )
  write.csv(calib_df, calib_csv, row.names = FALSE)
  cat("Calibration parameters saved to:", calib_csv, "\n")
  
  # Save model information
  model_info_file <- paste0(output_dir, "/model_info.txt")
  sink(model_info_file)
  cat("Random Forest Model Information\n")
  cat("==============================\n")
  cat("Training date:", Sys.Date(), "\n")
  cat("Number of trees:", model$finalModel$ntree, "\n")
  cat("Number of variables tried at each split:", model$finalModel$mtry, "\n")
  cat("Training samples:", length(model$control$index[[1]]), "\n")
  cat("Number of features:", ncol(model$trainingData) - 1, "\n")
  cat("\nCalibration Parameters:\n")
  cat("Alpha (slope):", calibration_params$alpha, "\n")
  cat("Beta (intercept):", calibration_params$beta, "\n")
  cat("Calibration R-squared:", calibration_params$calibration_r_squared, "\n")
  sink()
  
  cat("Model information saved to:", model_info_file, "\n")
}

print_training_summary <- function(model, calibration_params, train_data) {
  # Print summary of training results
  cat("\nModel Training Summary:\n")
  cat("======================\n")
  cat("Model type: Random Forest\n")
  cat("Number of trees:", model$finalModel$ntree, "\n")
  cat("Number of variables tried at each split:", model$finalModel$mtry, "\n")
  cat("Training samples:", length(model$control$index[[1]]), "\n")
  cat("Number of features:", ncol(train_data) - 1, "\n")
  
  cat("\nCalibration Parameters:\n")
  cat("Alpha (slope):", round(calibration_params$alpha, 4), "\n")
  cat("Beta (intercept):", round(calibration_params$beta, 4), "\n")
  cat("Calibration R-squared:", round(calibration_params$calibration_r_squared, 4), "\n")
  
  # Print target variable statistics
  cat("\nTraining Data Statistics:\n")
  cat("Mean days to death:", round(mean(train_data$days_to_death.demographic), 1), "\n")
  cat("Median days to death:", round(median(train_data$days_to_death.demographic), 1), "\n")
  cat("Min days to death:", round(min(train_data$days_to_death.demographic), 1), "\n")
  cat("Max days to death:", round(max(train_data$days_to_death.demographic), 1), "\n")
}

# =============================================================================
# Main Training Pipeline
# =============================================================================

main_model_training <- function() {
  # Main model training pipeline
  
  cat("Glioma Survival Prediction - Model Training\n")
  cat("==========================================\n\n")
  
  # Load training data
  cat("Step 1: Loading training data...\n")
  train_data <- load_training_data()
  
  # Set up training control
  cat("\nStep 2: Setting up training control...\n")
  ctrl <- setup_train_control()
  
  # Train Random Forest model
  cat("\nStep 3: Training Random Forest model...\n")
  model <- train_rf_model(train_data, ctrl)
  
  # Calculate calibration parameters
  cat("\nStep 4: Calculating calibration parameters...\n")
  calibration_params <- calculate_calibration_parameters(model, train_data)
  
  # Save model and parameters
  cat("\nStep 5: Saving model and parameters...\n")
  save_model_and_parameters(model, calibration_params)
  
  # Print training summary
  print_training_summary(model, calibration_params, train_data)
  
  # Print final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("MODEL TRAINING COMPLETE\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- rf_model.rds (trained model)\n")
  cat("- calibration_parameters.rds (calibration parameters)\n")
  cat("- calibration_parameters.csv (calibration parameters)\n")
  cat("- model_info.txt (model information)\n")
  
  return(list(
    model = model,
    calibration_params = calibration_params,
    train_data = train_data
  ))
}

# =============================================================================
# Run Model Training
# =============================================================================

if (!interactive()) {
  # Run the model training if script is executed directly
  results <- main_model_training()
} 