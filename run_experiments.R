# =============================================================================
# Glioma Survival Prediction Analysis (Simplified - Original Scale Only)
# =============================================================================

# Install required packages (uncomment if needed)
# install.packages(c("readxl", "caret", "randomForest", "gbm", "kernlab", "e1071", "dplyr"), dependencies = TRUE)

# Load required libraries
library(readxl)
library(caret)
library(randomForest)
library(gbm)
library(kernlab)
library(e1071)
library(dplyr)

# =============================================================================
# Configuration
# =============================================================================

# Define methods to try
METHODS <- c("lm", "rf", "knn", "svmRadial", "gbm")

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

prepare_xcell_dataset <- function(clinica_data, xcell_data, cell_types) {
  # Prepare xCell dataset for modeling (original scale, no transformations)
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
# Model Training and Evaluation Functions
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

train_and_evaluate_model <- function(data, method, ctrl) {
  # Train a model and evaluate its performance (original scale)
  set.seed(123)
  
  # Train model on original scale
  model <- train(
    days_to_death.demographic ~ .,
    data = data,
    method = method,
    trControl = ctrl
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
  
  return(list(
    model = model,
    predictions = test_predictions,
    actual = test_actual,
    correlation = correlation,
    rmse = rmse,
    test_indices = test_indices
  ))
}

analyze_variable_importance <- function(model, method, output_dir = "results") {
  # Analyze variable importance for supported models
  if (method == "rf") {
    # Extract variable importance from Random Forest
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
    pdf(paste0(output_dir, "/variable_importance_", method, ".pdf"), 
        width = 12, height = max(8, nrow(imp_df) * 0.3))
    
    par(mar = c(5, 12, 4, 2))
    barplot(imp_df$Importance, 
            names.arg = imp_df$Variable,
            horiz = TRUE,
            las = 2,
            col = "steelblue",
            main = paste("Variable Importance -", toupper(method)),
            xlab = "Importance Score",
            cex.names = 0.8)
    
    dev.off()
    
    # Print top 20 most important variables
    cat("\nTop 20 Most Important Variables (", toupper(method), "):\n")
    cat("==================================================\n")
    top_20 <- head(imp_df, 20)
    for (i in 1:nrow(top_20)) {
      cat(sprintf("%2d. %-30s: %8.3f\n", i, top_20$Variable[i], top_20$Importance[i]))
    }
    
    return(imp_df)
    
  } else if (method == "gbm") {
    # Extract variable importance from GBM
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
    pdf(paste0(output_dir, "/variable_importance_", method, ".pdf"), 
        width = 12, height = max(8, nrow(imp_df) * 0.3))
    
    par(mar = c(5, 12, 4, 2))
    barplot(imp_df$Importance, 
            names.arg = imp_df$Variable,
            horiz = TRUE,
            las = 2,
            col = "darkgreen",
            main = paste("Variable Importance -", toupper(method)),
            xlab = "Importance Score",
            cex.names = 0.8)
    
    dev.off()
    
    # Print top 20 most important variables
    cat("\nTop 20 Most Important Variables (", toupper(method), "):\n")
    cat("==================================================\n")
    top_20 <- head(imp_df, 20)
    for (i in 1:nrow(top_20)) {
      cat(sprintf("%2d. %-30s: %8.3f\n", i, top_20$Variable[i], top_20$Importance[i]))
    }
    
    return(imp_df)
    
  } else {
    cat("\nVariable importance analysis not available for method:", method, "\n")
    return(NULL)
  }
}

create_scatter_plot <- function(results, method, output_dir = "results") {
  # Create and save scatter plot of predicted vs actual values (original scale)
  pdf(paste0(output_dir, "/predicted_vs_actual_", method, ".pdf"))
  par(mar = c(5, 4, 4, 6))
  
  plot(results$actual, results$predictions,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = paste("Predicted vs Actual (", method, ")"),
       pch = 19, col = "blue")
  
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning - use quantiles to avoid edge clipping
  x_pos <- quantile(results$actual, 0.1)
  y_pos <- quantile(results$predictions, 0.9)
  
  text(x_pos, y_pos,
       paste("Corr:", round(results$correlation, 3), 
             "\nRMSE:", round(results$rmse, 0), "days"),
       pos = 4, offset = 0.5, bg = "white", cex = 0.9, col = "darkblue")
  
  dev.off()
}

print_results_summary <- function(results_list, methods) {
  # Print summary of model performance results
  cat("\nModel Performance Summary (Original Scale):\n")
  cat("===========================================\n")
  
  for (m in methods) {
    cat("\nMethod:", m, "\n")
    cat("Correlation:", round(results_list[[m]]$correlation, 3), "\n")
    cat("RMSE:", round(results_list[[m]]$rmse, 0), "days\n")
  }
}

create_importance_comparison <- function(importance_results, output_dir = "results") {
  # Create a comprehensive comparison of variable importance across models
  if (length(importance_results) == 0) {
    cat("No importance results to compare.\n")
    return(NULL)
  }
  
  # Get all unique variables
  all_variables <- unique(unlist(lapply(importance_results, function(x) x$Variable)))
  
  # Create comparison data frame
  comparison_df <- data.frame(Variable = all_variables)
  
  # Add importance scores for each method
  for (method in names(importance_results)) {
    method_imp <- importance_results[[method]]
    comparison_df[[paste0(method, "_importance")]] <- 
      method_imp$Importance[match(comparison_df$Variable, method_imp$Variable)]
  }
  
  # Calculate average importance across methods
  importance_cols <- grep("_importance$", colnames(comparison_df), value = TRUE)
  comparison_df$avg_importance <- rowMeans(comparison_df[, importance_cols], na.rm = TRUE)
  
  # Sort by average importance
  comparison_df <- comparison_df[order(-comparison_df$avg_importance), ]
  
  # Save to CSV
  write.csv(comparison_df, paste0(output_dir, "/variable_importance_comparison.csv"), 
            row.names = FALSE)
  
  # Create comparison plot for top 30 variables
  top_30 <- head(comparison_df, 30)
  
  pdf(paste0(output_dir, "/variable_importance_comparison.pdf"), 
      width = 14, height = 10)
  
  # Set up the plot
  par(mar = c(8, 12, 4, 2))
  
  # Create bar plot
  barplot(t(as.matrix(top_30[, importance_cols])),
          names.arg = top_30$Variable,
          horiz = TRUE,
          las = 2,
          col = c("steelblue", "darkgreen"),
          main = "Variable Importance Comparison (Top 30)",
          xlab = "Importance Score",
          cex.names = 0.7,
          legend.text = gsub("_importance", "", importance_cols),
          args.legend = list(x = "topright", bty = "n"))
  
  dev.off()
  
  # Print summary
  cat("\nVariable Importance Comparison Summary:\n")
  cat("=======================================\n")
  cat("Top 10 variables by average importance:\n")
  top_10 <- head(comparison_df, 10)
  for (i in 1:nrow(top_10)) {
    cat(sprintf("%2d. %-30s: %8.3f\n", i, top_10$Variable[i], top_10$avg_importance[i]))
  }
  
  # Save top 50 variables to separate CSV
  top_50 <- head(comparison_df, 50)
  write.csv(top_50, paste0(output_dir, "/top_50_important_variables.csv"), 
            row.names = FALSE)
  
  cat("\nResults saved to:\n")
  cat("- variable_importance_comparison.csv (all variables)\n")
  cat("- top_50_important_variables.csv (top 50 variables)\n")
  cat("- variable_importance_comparison.pdf (comparison plot)\n")
  
  return(comparison_df)
}

# =============================================================================
# Main Analysis Pipeline
# =============================================================================

main_analysis <- function() {
  # Main analysis pipeline (simplified - original scale only)
  
  # Load data
  cat("Loading data...\n")
  clinica_data <- load_clinical_data()
  xcell_data_list <- load_xcell_data()
  
  # Prepare dataset
  cat("Preparing dataset...\n")
  data_xcell <- prepare_xcell_dataset(
    clinica_data, 
    xcell_data_list$data, 
    xcell_data_list$cell_types
  )
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Train and evaluate models
  cat("Training and evaluating models...\n")
  model_results <- list()
  importance_results <- list()
  
  for (method in METHODS) {
    cat("Processing method:", method, "\n")
    
    # Train and evaluate model
    results <- train_and_evaluate_model(data_xcell, method, ctrl)
    model_results[[method]] <- results
    
    # Create scatter plot
    create_scatter_plot(results, method)
    
    # Analyze variable importance for tree-based models
    if (method %in% c("rf", "gbm")) {
      cat("Analyzing variable importance for", method, "...\n")
      importance_df <- analyze_variable_importance(results$model, method)
      importance_results[[method]] <- importance_df
    }
  }
  
  # Print summary
  print_results_summary(model_results, METHODS)
  
  # Create importance comparison
  create_importance_comparison(importance_results)
  
  # Return results
  return(list(
    model_results = model_results,
    importance_results = importance_results
  ))
}

# =============================================================================
# Run Analysis
# =============================================================================

if (!interactive()) {
  # Run the analysis if script is executed directly
  results <- main_analysis()
}
