# =============================================================================
# Debug Script - Compare Three-Script vs Original Approach
# =============================================================================

# Load required libraries
library(readxl)
library(caret)
library(randomForest)
library(dplyr)

# Set random seed for reproducibility
set.seed(123)

# =============================================================================
# Data Loading Functions (same as original)
# =============================================================================

load_clinical_data <- function(file_path = "data/ClinicaGliomasMayo2025.xlsx") {
  read_excel(file_path)
}

load_xcell_data <- function(file_path = "data/xCell_gene_tpm_Mayo2025.xlsx") {
  xcell_data <- read_excel(file_path)
  
  # Reshape: make TCGACode a column and CellType as column names
  xcell_t <- as.data.frame(t(xcell_data[,-1]))
  CellTypeNames <- xcell_data$CellType
  colnames(xcell_t) <- CellTypeNames
  xcell_t$TCGACode <- colnames(xcell_data)[-1]
  xcell_t <- xcell_t %>% relocate(TCGACode)
  
  return(list(data = xcell_t, cell_types = CellTypeNames))
}

prepare_dataset <- function(clinica_data, xcell_data, cell_types) {
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
  
  return(data_xcell)
}

# =============================================================================
# Original Approach (from rf_classifier.R)
# =============================================================================

run_original_approach <- function(data_xcell) {
  cat("Running ORIGINAL approach (from rf_classifier.R)...\n")
  
  # Set up training control (same as original)
  ctrl_original <- trainControl(
    method = "boot",
    number = 1,
    p = 0.7,
    savePredictions = TRUE,
    returnResamp = "all"
  )
  
  # Train model on full dataset
  model_original <- train(
    days_to_death.demographic ~ .,
    data = data_xcell,
    method = "rf",
    trControl = ctrl_original,
    importance = TRUE
  )
  
  # Get test set indices
  train_indices_original <- model_original$control$index[[1]]
  test_indices_original <- setdiff(1:nrow(data_xcell), train_indices_original)
  
  # Get predictions for test set (same as original)
  train_predictions_original <- predict(model_original, newdata = data_xcell[train_indices_original,])
  train_actual_original <- data_xcell$days_to_death.demographic[train_indices_original]
  
  # Fit linear model: actual = alpha * prediction + beta
  calib_lm_original <- lm(train_actual_original ~ train_predictions_original)
  
  # Extract coefficients
  alpha_original <- calib_lm_original$coefficients["train_predictions_original"]
  beta_original <- calib_lm_original$coefficients["(Intercept)"]
  
  # Apply calibration to test predictions
  test_predictions_original <- predict(model_original, newdata = data_xcell[test_indices_original,])
  test_predictions_original <- alpha_original * test_predictions_original + beta_original
  test_actual_original <- data_xcell$days_to_death.demographic[test_indices_original]
  
  # Calculate performance metrics
  correlation_original <- cor(test_actual_original, test_predictions_original)
  rmse_original <- sqrt(mean((test_actual_original - test_predictions_original)^2))
  mae_original <- mean(abs(test_actual_original - test_predictions_original))
  r_squared_original <- cor(test_actual_original, test_predictions_original)^2
  
  cat("Original approach results:\n")
  cat("Correlation:", round(correlation_original, 3), "\n")
  cat("RMSE:", round(rmse_original, 0), "days\n")
  cat("MAE:", round(mae_original, 0), "days\n")
  cat("R-squared:", round(r_squared_original, 3), "\n")
  cat("Alpha:", round(alpha_original, 4), "\n")
  cat("Beta:", round(beta_original, 4), "\n")
  cat("Training samples:", length(train_indices_original), "\n")
  cat("Test samples:", length(test_indices_original), "\n\n")
  
  return(list(
    model = model_original,
    predictions = test_predictions_original,
    actual = test_actual_original,
    correlation = correlation_original,
    rmse = rmse_original,
    mae = mae_original,
    r_squared = r_squared_original,
    alpha = alpha_original,
    beta = beta_original,
    train_indices = train_indices_original,
    test_indices = test_indices_original
  ))
}

# =============================================================================
# Three-Script Approach (simulated)
# =============================================================================

run_three_script_approach <- function(data_xcell) {
  cat("Running THREE-SCRIPT approach...\n")
  
  # Step 1: Create train/test split (like script 1)
  train_indices_three <- createDataPartition(data_xcell$days_to_death.demographic, p = 0.7, list = FALSE)
  train_data_three <- data_xcell[train_indices_three, ]
  test_data_three <- data_xcell[-train_indices_three, ]
  
  cat("Three-script split:\n")
  cat("Training samples:", nrow(train_data_three), "\n")
  cat("Test samples:", nrow(test_data_three), "\n")
  
  # Step 2: Train model on training data only (like script 2)
  ctrl_three <- trainControl(
    method = "boot",
    number = 1,
    p = 0.7,
    savePredictions = TRUE,
    returnResamp = "all"
  )
  
  # Train model on training data only
  model_three <- train(
    days_to_death.demographic ~ .,
    data = train_data_three,
    method = "rf",
    trControl = ctrl_three,
    importance = TRUE
  )
  
  # Get internal train/test split from caret
  internal_train_indices <- model_three$control$index[[1]]
  internal_test_indices <- setdiff(1:nrow(train_data_three), internal_train_indices)
  
  cat("Internal caret split within training data:\n")
  cat("Internal training samples:", length(internal_train_indices), "\n")
  cat("Internal test samples:", length(internal_test_indices), "\n")
  
  # Calculate calibration parameters (like script 2)
  train_predictions_three <- predict(model_three, newdata = train_data_three[internal_train_indices,])
  train_actual_three <- train_data_three$days_to_death.demographic[internal_train_indices]
  
  calib_lm_three <- lm(train_actual_three ~ train_predictions_three)
  alpha_three <- calib_lm_three$coefficients["train_predictions_three"]
  beta_three <- calib_lm_three$coefficients["(Intercept)"]
  
  # Step 3: Make predictions on test set (like script 3)
  test_predictions_three <- predict(model_three, newdata = test_data_three)
  test_predictions_three <- alpha_three * test_predictions_three + beta_three
  test_actual_three <- test_data_three$days_to_death.demographic
  
  # Calculate performance metrics
  correlation_three <- cor(test_actual_three, test_predictions_three)
  rmse_three <- sqrt(mean((test_actual_three - test_predictions_three)^2))
  mae_three <- mean(abs(test_actual_three - test_predictions_three))
  r_squared_three <- cor(test_actual_three, test_predictions_three)^2
  
  cat("Three-script approach results:\n")
  cat("Correlation:", round(correlation_three, 3), "\n")
  cat("RMSE:", round(rmse_three, 0), "days\n")
  cat("MAE:", round(mae_three, 0), "days\n")
  cat("R-squared:", round(r_squared_three, 3), "\n")
  cat("Alpha:", round(alpha_three, 4), "\n")
  cat("Beta:", round(beta_three, 4), "\n\n")
  
  return(list(
    model = model_three,
    predictions = test_predictions_three,
    actual = test_actual_three,
    correlation = correlation_three,
    rmse = rmse_three,
    mae = mae_three,
    r_squared = r_squared_three,
    alpha = alpha_three,
    beta = beta_three,
    train_indices = train_indices_three,
    test_indices = setdiff(1:nrow(data_xcell), train_indices_three)
  ))
}

# =============================================================================
# Fixed Three-Script Approach
# =============================================================================

run_fixed_three_script_approach <- function(data_xcell) {
  cat("Running FIXED THREE-SCRIPT approach...\n")
  
  # Step 1: Create train/test split (like script 1)
  train_indices_fixed <- createDataPartition(data_xcell$days_to_death.demographic, p = 0.7, list = FALSE)
  train_data_fixed <- data_xcell[train_indices_fixed, ]
  test_data_fixed <- data_xcell[-train_indices_fixed, ]
  
  cat("Fixed three-script split:\n")
  cat("Training samples:", nrow(train_data_fixed), "\n")
  cat("Test samples:", nrow(test_data_fixed), "\n")
  
  # Step 2: Train model on training data only with NO internal resampling
  ctrl_fixed <- trainControl(
    method = "none",  # No resampling - train on all training data
    savePredictions = FALSE
  )
  
  # Train model on all training data
  model_fixed <- train(
    days_to_death.demographic ~ .,
    data = train_data_fixed,
    method = "rf",
    trControl = ctrl_fixed,
    importance = TRUE
  )
  
  # Calculate calibration parameters using all training data
  train_predictions_fixed <- predict(model_fixed, newdata = train_data_fixed)
  train_actual_fixed <- train_data_fixed$days_to_death.demographic
  
  calib_lm_fixed <- lm(train_actual_fixed ~ train_predictions_fixed)
  alpha_fixed <- calib_lm_fixed$coefficients["train_predictions_fixed"]
  beta_fixed <- calib_lm_fixed$coefficients["(Intercept)"]
  
  # Step 3: Make predictions on test set
  test_predictions_fixed <- predict(model_fixed, newdata = test_data_fixed)
  test_predictions_fixed <- alpha_fixed * test_predictions_fixed + beta_fixed
  test_actual_fixed <- test_data_fixed$days_to_death.demographic
  
  # Calculate performance metrics
  correlation_fixed <- cor(test_actual_fixed, test_predictions_fixed)
  rmse_fixed <- sqrt(mean((test_actual_fixed - test_predictions_fixed)^2))
  mae_fixed <- mean(abs(test_actual_fixed - test_predictions_fixed))
  r_squared_fixed <- cor(test_actual_fixed, test_predictions_fixed)^2
  
  cat("Fixed three-script approach results:\n")
  cat("Correlation:", round(correlation_fixed, 3), "\n")
  cat("RMSE:", round(rmse_fixed, 0), "days\n")
  cat("MAE:", round(mae_fixed, 0), "days\n")
  cat("R-squared:", round(r_squared_fixed, 3), "\n")
  cat("Alpha:", round(alpha_fixed, 4), "\n")
  cat("Beta:", round(beta_fixed, 4), "\n\n")
  
  return(list(
    model = model_fixed,
    predictions = test_predictions_fixed,
    actual = test_actual_fixed,
    correlation = correlation_fixed,
    rmse = rmse_fixed,
    mae = mae_fixed,
    r_squared = r_squared_fixed,
    alpha = alpha_fixed,
    beta = beta_fixed,
    train_indices = train_indices_fixed,
    test_indices = setdiff(1:nrow(data_xcell), train_indices_fixed)
  ))
}

# =============================================================================
# Main Debug Function
# =============================================================================

main_debug <- function() {
  cat("DEBUG: Comparing Three-Script vs Original Approach\n")
  cat("==================================================\n\n")
  
  # Load and prepare data
  cat("Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  data_xcell <- prepare_dataset(clinica_data, xcell_data_list$data, xcell_data_list$cell_types)
  
  cat("Dataset prepared with", nrow(data_xcell), "samples and", ncol(data_xcell)-1, "features\n\n")
  
  # Run all three approaches
  results_original <- run_original_approach(data_xcell)
  results_three <- run_three_script_approach(data_xcell)
  results_fixed <- run_fixed_three_script_approach(data_xcell)
  
  # Compare results
  cat("COMPARISON SUMMARY:\n")
  cat("==================\n")
  cat("Original approach:\n")
  cat("  Correlation:", round(results_original$correlation, 3), "\n")
  cat("  RMSE:", round(results_original$rmse, 0), "days\n")
  cat("  R-squared:", round(results_original$r_squared, 3), "\n\n")
  
  cat("Three-script approach (current):\n")
  cat("  Correlation:", round(results_three$correlation, 3), "\n")
  cat("  RMSE:", round(results_three$rmse, 0), "days\n")
  cat("  R-squared:", round(results_three$r_squared, 3), "\n\n")
  
  cat("Fixed three-script approach:\n")
  cat("  Correlation:", round(results_fixed$correlation, 3), "\n")
  cat("  RMSE:", round(results_fixed$rmse, 0), "days\n")
  cat("  R-squared:", round(results_fixed$r_squared, 3), "\n\n")
  
  # Check if train/test splits overlap
  cat("Train/Test Split Analysis:\n")
  cat("=========================\n")
  cat("Original approach test samples:", length(results_original$test_indices), "\n")
  cat("Three-script approach test samples:", length(results_three$test_indices), "\n")
  cat("Fixed three-script approach test samples:", length(results_fixed$test_indices), "\n")
  
  # Check overlap between original and three-script test sets
  overlap_original_three <- length(intersect(results_original$test_indices, results_three$test_indices))
  overlap_original_fixed <- length(intersect(results_original$test_indices, results_fixed$test_indices))
  
  cat("Overlap between original and three-script test sets:", overlap_original_three, "\n")
  cat("Overlap between original and fixed three-script test sets:", overlap_original_fixed, "\n")
  
  return(list(
    original = results_original,
    three_script = results_three,
    fixed_three_script = results_fixed
  ))
}

# Run debug
results <- main_debug() 