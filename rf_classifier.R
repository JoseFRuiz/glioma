# =============================================================================
# Glioma Survival Prediction - Random Forest Classifier
# =============================================================================

# Install required packages (uncomment if needed)
# install.packages(c("readxl", "caret", "randomForest", "dplyr"), dependencies = TRUE)

# Load required libraries
library(readxl)
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

# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================

load_clinical_data <- function(file_path = "data/ClinicaGliomasMayo2025.xlsx") {
  # Load clinical data from Excel file
  read_excel(file_path)
}

load_xcell_data <- function(file_path = "data/xCell_gene_tpm_Mayo2025.xlsx") {
  # Load and reshape xCell gene expression data
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
  # Prepare dataset for modeling (original scale, no transformations)
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
# Random Forest Training and Evaluation
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

train_rf_model <- function(data, ctrl) {
  # Train Random Forest model and evaluate performance
  set.seed(123)
  
  # Train model on original scale
  model <- train(
    days_to_death.demographic ~ .,
    data = data,
    method = "rf",
    trControl = ctrl,
    importance = TRUE  # Enable variable importance
  )
  
  # Get test set indices
  train_indices <- model$control$index[[1]]
  test_indices <- setdiff(1:nrow(data), train_indices)
  
  # Get predictions for test set
  test_predictions <- predict(model, newdata = data[test_indices,])
  test_actual <- data$days_to_death.demographic[test_indices]
  
  # Calculate performance metrics
  correlation <- cor(test_actual, test_predictions)
  rmse <- sqrt(mean((test_actual - test_predictions)^2))
  
  # Calculate additional metrics
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
    test_indices = test_indices
  ))
}

analyze_variable_importance <- function(model, output_dir = "results") {
  # Analyze variable importance from Random Forest
  
  # Extract variable importance
  importance_data <- varImp(model$finalModel)
  
  # Handle different importance data structures
  if (is.data.frame(importance_data)) {
    # Direct data frame
    imp_df <- data.frame(
      Variable = rownames(importance_data),
      Importance = importance_data[, 1]
    )
  } else if (is.list(importance_data) && "importance" %in% names(importance_data)) {
    # List with importance element
    imp_df <- data.frame(
      Variable = rownames(importance_data$importance),
      Importance = importance_data$importance[, 1]
    )
  } else {
    # Try to convert to data frame
    imp_df <- as.data.frame(importance_data)
    imp_df$Variable <- rownames(imp_df)
    imp_df <- imp_df[, c("Variable", names(imp_df)[1])]
    names(imp_df)[2] <- "Importance"
  }
  
  # Ensure Importance is numeric and handle NA values
  imp_df$Importance <- as.numeric(imp_df$Importance)
  imp_df <- imp_df[!is.na(imp_df$Importance), ]
  
  # Sort by importance (descending)
  imp_df <- imp_df[order(-imp_df$Importance), ]
  
  # Create variable importance plot
  pdf(paste0(output_dir, "/rf_variable_importance.pdf"), 
      width = 12, height = max(8, nrow(imp_df) * 0.3))
  
  par(mar = c(5, 12, 4, 2))
  barplot(imp_df$Importance, 
          names.arg = imp_df$Variable,
          horiz = TRUE,
          las = 2,
          col = "steelblue",
          main = "Random Forest Variable Importance",
          xlab = "Importance Score",
          cex.names = 0.8)
  
  dev.off()
  
  # Print top 20 most important variables
  cat("\nTop 20 Most Important Variables (Random Forest):\n")
  cat("================================================\n")
  top_20 <- head(imp_df, 20)
  for (i in 1:nrow(top_20)) {
    cat(sprintf("%2d. %-30s: %8.3f\n", i, top_20$Variable[i], top_20$Importance[i]))
  }
  
  # Save importance data to CSV
  write.csv(imp_df, paste0(output_dir, "/rf_variable_importance.csv"), row.names = FALSE)
  
  # Save top 50 variables to separate CSV
  top_50 <- head(imp_df, 50)
  write.csv(top_50, paste0(output_dir, "/rf_top_50_variables.csv"), row.names = FALSE)
  
  return(imp_df)
}

create_scatter_plot <- function(results, output_dir = "results") {
  # Create and save scatter plot of predicted vs actual values
  pdf(paste0(output_dir, "/rf_predicted_vs_actual.pdf"))
  par(mar = c(5, 4, 4, 6))
  
  plot(results$actual, results$predictions,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Random Forest: Predicted vs Actual Survival",
       pch = 19, col = "blue")
  
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning - use quantiles to avoid edge clipping
  x_pos <- quantile(results$actual, 0.1)
  y_pos <- quantile(results$predictions, 0.9)
  
  text(x_pos, y_pos,
       paste("Correlation:", round(results$correlation, 3), 
             "\nRMSE:", round(results$rmse, 0), "days",
             "\nMAE:", round(results$mae, 0), "days",
             "\nR²:", round(results$r_squared, 3)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.9, col = "darkblue")
  
  dev.off()
}

print_results_summary <- function(results) {
  # Print summary of Random Forest performance
  cat("\nRandom Forest Performance Summary:\n")
  cat("==================================\n")
  cat("Correlation:", round(results$correlation, 3), "\n")
  cat("RMSE:", round(results$rmse, 0), "days\n")
  cat("MAE:", round(results$mae, 0), "days\n")
  cat("R-squared:", round(results$r_squared, 3), "\n")
  
  # Print model info
  cat("\nModel Information:\n")
  cat("==================\n")
  cat("Number of trees:", results$model$finalModel$ntree, "\n")
  cat("Number of variables tried at each split:", results$model$finalModel$mtry, "\n")
  cat("Training samples:", length(results$model$control$index[[1]]), "\n")
  cat("Test samples:", length(results$test_indices), "\n")
}

# =============================================================================
# Main Analysis Pipeline
# =============================================================================

main_analysis <- function() {
  # Main Random Forest analysis pipeline
  
  cat("Random Forest Glioma Survival Prediction Analysis\n")
  cat("================================================\n\n")
  
  # Load data
  cat("Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  
  # Prepare dataset
  cat("Preparing dataset...\n")
  data_xcell <- prepare_dataset(
    clinica_data, 
    xcell_data_list$data, 
    xcell_data_list$cell_types
  )
  
  cat("Dataset prepared with", nrow(data_xcell), "samples and", ncol(data_xcell)-1, "features\n\n")
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Train Random Forest model
  cat("Training Random Forest model...\n")
  results <- train_rf_model(data_xcell, ctrl)
  
  # Print results summary
  print_results_summary(results)
  
  # Create scatter plot
  cat("\nCreating scatter plot...\n")
  create_scatter_plot(results)
  
  # Analyze variable importance
  cat("\nAnalyzing variable importance...\n")
  importance_df <- analyze_variable_importance(results$model)
  
  # Print final summary
  cat("\n" + paste0(rep("=", 50), collapse = ""), "\n")
  cat("ANALYSIS COMPLETE\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- rf_predicted_vs_actual.pdf (scatter plot)\n")
  cat("- rf_variable_importance.pdf (importance plot)\n")
  cat("- rf_variable_importance.csv (all variables)\n")
  cat("- rf_top_50_variables.csv (top 50 variables)\n")
  
  # Return results
  return(list(
    results = results,
    importance_df = importance_df,
    dataset_info = list(
      n_samples = nrow(data_xcell),
      n_features = ncol(data_xcell) - 1
    )
  ))
}

# =============================================================================
# Run Analysis
# =============================================================================

if (!interactive()) {
  # Run the analysis if script is executed directly
  results <- main_analysis()
} 