# =============================================================================
# Glioma Survival Prediction Analysis
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
  # Prepare xCell dataset for modeling
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
  
  # Create log-transformed target variable
  data_xcell$log_days_to_death <- log(data_xcell$days_to_death.demographic)
  
  return(data_xcell)
}

prepare_xcell_dataset_original_scale <- function(clinica_data, xcell_data, cell_types) {
  # Prepare xCell dataset for modeling (original scale, no log transformation)
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

prepare_xcell_dataset_alternative_transforms <- function(clinica_data, xcell_data, cell_types) {
  # Prepare xCell dataset with alternative transformations
  
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
  
  # Create alternative transformations
  data_xcell$log_days_to_death <- log(data_xcell$days_to_death.demographic)
  data_xcell$sqrt_days_to_death <- sqrt(data_xcell$days_to_death.demographic)
  data_xcell$cube_root_days_to_death <- (data_xcell$days_to_death.demographic)^(1/3)
  
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

train_and_evaluate_model <- function(data, method, ctrl, target_var = "log_days_to_death") {
  # Train a model and evaluate its performance
  set.seed(123)
  
  # Train model
  model <- train(
    as.formula(paste(target_var, "~ . - days_to_death.demographic")),
    data = data,
    method = method,
    trControl = ctrl
  )
  
  # Get test set indices
  train_indices <- model$control$index[[1]]
  test_indices <- setdiff(1:nrow(data), train_indices)
  
  # Get predictions for test set
  test_predictions <- predict(model, newdata = data[test_indices,])
  test_actual <- data[[target_var]][test_indices]
  
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
    
    # Debug: print structure of importance data
    cat("Debug: importance_data structure:\n")
    print(str(importance_data))
    
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
    
    # Debug: print structure of importance data
    cat("Debug: importance_data structure:\n")
    print(str(importance_data))
    
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
  # Create and save scatter plot of predicted vs actual values
  
  # Get original scale values (before log transformation)
  test_indices <- results$test_indices
  
  # Create both log-scale and original-scale plots
  
  # 1. Log-scale plot (current approach)
  pdf(paste0(output_dir, "/predicted_vs_actual_log_scale_", method, ".pdf"))
  par(mar = c(5, 4, 4, 6))
  
  plot(results$actual, results$predictions,
       xlab = "Actual Log(Days to Death)",
       ylab = "Predicted Log(Days to Death)",
       main = paste("Predicted vs Actual (Log Scale, ", method, ")"),
       pch = 19, col = "blue")
  
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning - use quantiles to avoid edge clipping
  x_pos <- quantile(results$actual, 0.1)
  y_pos <- quantile(results$predictions, 0.9)
  
  text(x_pos, y_pos,
       paste("Corr:", round(results$correlation, 3), 
             "\nRMSE:", round(results$rmse, 2)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.9, col = "darkblue")
  
  dev.off()
  
  # 2. Original scale plot (exponentiated) with bias correction
  actual_original <- exp(results$actual)
  predicted_original_raw <- exp(results$predictions)
  
  # Apply bias correction for log-transformed predictions
  # This corrects for the fact that E[exp(X)] ≠ exp(E[X]) when X is random
  predicted_original_corrected <- bias_correct_predictions(results$predictions, results$actual)
  
  # Calculate metrics in original scale (both raw and corrected)
  correlation_original_raw <- cor(actual_original, predicted_original_raw)
  rmse_original_raw <- sqrt(mean((actual_original - predicted_original_raw)^2))
  
  correlation_original_corrected <- cor(actual_original, predicted_original_corrected)
  rmse_original_corrected <- sqrt(mean((actual_original - predicted_original_corrected)^2))
  
  # Create plot with both raw and corrected predictions
  pdf(paste0(output_dir, "/predicted_vs_actual_original_scale_", method, ".pdf"))
  par(mar = c(5, 4, 4, 6))
  
  plot(actual_original, predicted_original_corrected,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = paste("Predicted vs Actual (Original Scale, ", method, ")"),
       pch = 19, col = "blue", ylim = range(c(predicted_original_raw, predicted_original_corrected, actual_original)))
  
  # Add raw predictions as red points for comparison
  points(actual_original, predicted_original_raw, pch = 3, col = "red", cex = 0.8)
  
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning for original scale
  x_pos_orig <- quantile(actual_original, 0.1)
  y_pos_orig <- quantile(predicted_original_corrected, 0.9)
  
  text(x_pos_orig, y_pos_orig,
       paste("Corr (corrected):", round(correlation_original_corrected, 3), 
             "\nRMSE (corrected):", round(rmse_original_corrected, 0),
             "\nCorr (raw):", round(correlation_original_raw, 3),
             "\nRMSE (raw):", round(rmse_original_raw, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkblue")
  
  # Add legend
  legend("topleft", legend = c("Bias-corrected", "Raw (exp)"), 
         pch = c(19, 3), col = c("blue", "red"), cex = 0.8)
  
  dev.off()
  
  # Return additional metrics
  return(list(
    correlation_original_raw = correlation_original_raw,
    rmse_original_raw = rmse_original_raw,
    correlation_original_corrected = correlation_original_corrected,
    rmse_original_corrected = rmse_original_corrected
  ))
}

bias_correct_predictions <- function(log_predictions, log_actual) {
  # Apply bias correction for log-transformed predictions
  # Method: Use the variance of the residuals to correct the bias
  
  # Calculate residuals in log space
  residuals_log <- log_actual - log_predictions
  
  # Estimate the variance of the residuals
  residual_variance <- var(residuals_log)
  
  # Apply bias correction: exp(pred + var/2)
  corrected_predictions <- exp(log_predictions + residual_variance/2)
  
  return(corrected_predictions)
}

advanced_bias_correction <- function(log_predictions, log_actual, method = "duan") {
  # Advanced bias correction methods
  
  if (method == "duan") {
    # Duan's smearing estimator - more robust than simple variance correction
    residuals_log <- log_actual - log_predictions
    
    # Calculate the smearing factor
    n <- length(residuals_log)
    smearing_factor <- sum(exp(residuals_log)) / n
    
    # Apply Duan's correction
    corrected_predictions <- exp(log_predictions) * smearing_factor
    
  } else if (method == "cross_validation") {
    # Cross-validation based bias correction
    n <- length(log_predictions)
    corrected_predictions <- numeric(n)
    
    for (i in 1:n) {
      # Leave-one-out approach
      residuals_log <- log_actual[-i] - log_predictions[-i]  # Calculate residuals excluding current point
      train_variance <- var(residuals_log)
      corrected_predictions[i] <- exp(log_predictions[i] + train_variance/2)
    }
    
  } else if (method == "robust") {
    # Robust bias correction using median absolute deviation
    residuals_log <- log_actual - log_predictions
    
    # Use robust estimate of variance
    mad_residuals <- mad(residuals_log)
    robust_variance <- (mad_residuals / 0.6745)^2  # Convert MAD to variance
    
    corrected_predictions <- exp(log_predictions + robust_variance/2)
    
  } else {
    # Default to simple variance correction
    corrected_predictions <- bias_correct_predictions(log_predictions, log_actual)
  }
  
  return(corrected_predictions)
}

create_enhanced_scatter_plot <- function(results, method, output_dir = "results") {
  # Create enhanced scatter plot with multiple bias correction methods
  
  # Get original scale values
  actual_original <- exp(results$actual)
  predicted_original_raw <- exp(results$predictions)
  
  # Apply different bias correction methods
  predicted_duan <- advanced_bias_correction(results$predictions, results$actual, "duan")
  predicted_cv <- advanced_bias_correction(results$predictions, results$actual, "cross_validation")
  predicted_robust <- advanced_bias_correction(results$predictions, results$actual, "robust")
  predicted_simple <- bias_correct_predictions(results$predictions, results$actual)
  
  # Calculate metrics for all methods
  metrics <- list()
  methods_list <- list(
    raw = predicted_original_raw,
    simple = predicted_simple,
    duan = predicted_duan,
    cv = predicted_cv,
    robust = predicted_robust
  )
  
  for (method_name in names(methods_list)) {
    pred <- methods_list[[method_name]]
    metrics[[method_name]] <- list(
      correlation = cor(actual_original, pred),
      rmse = sqrt(mean((actual_original - pred)^2))
    )
  }
  
  # Create comprehensive comparison plot
  pdf(paste0(output_dir, "/bias_correction_comparison_", method, ".pdf"), 
      width = 14, height = 10)
  
  par(mfrow = c(2, 3))
  
  # Plot 1: Log scale
  plot(results$actual, results$predictions,
       xlab = "Actual Log(Days to Death)",
       ylab = "Predicted Log(Days to Death)",
       main = "Log Scale Performance",
       pch = 19, col = "blue")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(results$actual, 0.1), quantile(results$predictions, 0.9),
       paste("Corr:", round(results$correlation, 3), "\nRMSE:", round(results$rmse, 2)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkblue")
  
  # Plot 2: Raw exponential
  plot(actual_original, predicted_original_raw,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Raw Exponential",
       pch = 19, col = "red")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(actual_original, 0.1), quantile(predicted_original_raw, 0.9),
       paste("Corr:", round(metrics$raw$correlation, 3), "\nRMSE:", round(metrics$raw$rmse, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkred")
  
  # Plot 3: Simple bias correction
  plot(actual_original, predicted_simple,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Simple Bias Correction",
       pch = 19, col = "orange")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(actual_original, 0.1), quantile(predicted_simple, 0.9),
       paste("Corr:", round(metrics$simple$correlation, 3), "\nRMSE:", round(metrics$simple$rmse, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkorange")
  
  # Plot 4: Duan's smearing
  plot(actual_original, predicted_duan,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Duan's Smearing",
       pch = 19, col = "green")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(actual_original, 0.1), quantile(predicted_duan, 0.9),
       paste("Corr:", round(metrics$duan$correlation, 3), "\nRMSE:", round(metrics$duan$rmse, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkgreen")
  
  # Plot 5: Cross-validation
  plot(actual_original, predicted_cv,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Cross-Validation",
       pch = 19, col = "purple")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(actual_original, 0.1), quantile(predicted_cv, 0.9),
       paste("Corr:", round(metrics$cv$correlation, 3), "\nRMSE:", round(metrics$cv$rmse, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "purple4")
  
  # Plot 6: Robust
  plot(actual_original, predicted_robust,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Robust Correction",
       pch = 19, col = "brown")
  abline(0, 1, col = "red", lty = 2)
  text(quantile(actual_original, 0.1), quantile(predicted_robust, 0.9),
       paste("Corr:", round(metrics$robust$correlation, 3), "\nRMSE:", round(metrics$robust$rmse, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "brown4")
  
  dev.off()
  
  # Print comparison summary
  cat("\nBias Correction Methods Comparison (", method, "):\n")
  cat("================================================\n")
  for (method_name in names(metrics)) {
    cat(sprintf("%-20s: Corr = %.3f, RMSE = %.0f days\n", 
                method_name, 
                metrics[[method_name]]$correlation, metrics[[method_name]]$rmse))
  }
  
  # Find best method
  best_method <- names(metrics)[which.min(sapply(metrics, function(x) x$rmse))]
  cat("\nBest method by RMSE:", best_method, "\n")
  cat("Best RMSE:", round(metrics[[best_method]]$rmse, 0), "days\n")
  
  return(list(
    metrics = metrics,
    best_method = best_method,
    predictions = methods_list
  ))
}

print_results_summary <- function(results_list, methods) {
  # Print summary of model performance results
  cat("\nModel Performance Summary (xCell features):\n")
  cat("===========================================\n")
  
  for (m in methods) {
    cat("\nMethod:", m, "\n")
    cat("Log Scale:\n")
    cat("  Correlation:", round(results_list[[m]]$correlation, 3), "\n")
    cat("  RMSE:", round(results_list[[m]]$rmse, 2), "log(days)\n")
    
    # Get original scale metrics if available
    if ("correlation_original_raw" %in% names(results_list[[m]])) {
      cat("Original Scale:\n")
      cat("  Correlation (raw):", round(results_list[[m]]$correlation_original_raw, 3), "\n")
      cat("  RMSE (raw):", round(results_list[[m]]$rmse_original_raw, 0), "days\n")
      cat("  Correlation (corrected):", round(results_list[[m]]$correlation_original_corrected, 3), "\n")
      cat("  RMSE (corrected):", round(results_list[[m]]$rmse_original_corrected, 0), "days\n")
    }
    
    # Enhanced bias correction results for tree-based models
    if (m %in% c("rf", "gbm") && "enhanced_metrics" %in% names(results_list[[m]])) {
      cat("Enhanced Bias Correction:\n")
      enhanced_metrics <- results_list[[m]]$enhanced_metrics
      best_method <- results_list[[m]]$best_bias_correction
      
      for (method_name in names(enhanced_metrics)) {
        marker <- ifelse(method_name == best_method, " *", "")
        cat(sprintf("  %-15s: Corr = %.3f, RMSE = %.0f days%s\n", 
                    method_name, 
                    enhanced_metrics[[method_name]]$correlation, 
                    enhanced_metrics[[method_name]]$rmse,
                    marker))
      }
      cat("  (* = Best method by RMSE)\n")
    }
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

compare_scale_approaches <- function(clinica_data, xcell_data_list, output_dir = "results") {
  # Compare log-transformed vs original scale modeling approaches
  
  cat("\nComparing Log-Transformed vs Original Scale Approaches:\n")
  cat("======================================================\n")
  
  # Prepare datasets
  data_log <- prepare_xcell_dataset(clinica_data, xcell_data_list$data, xcell_data_list$cell_types)
  data_original <- prepare_xcell_dataset_original_scale(clinica_data, xcell_data_list$data, xcell_data_list$cell_types)
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Test with Random Forest (most robust for this comparison)
  method <- "rf"
  
  # Train on log-transformed data
  cat("Training Random Forest on log-transformed data...\n")
  results_log <- train_and_evaluate_model(data_log, method, ctrl, "log_days_to_death")
  
  # Train on original scale data
  cat("Training Random Forest on original scale data...\n")
  results_original <- train_and_evaluate_model(data_original, method, ctrl, "days_to_death.demographic")
  
  # Compare performance
  cat("\nPerformance Comparison (Random Forest):\n")
  cat("======================================\n")
  cat("Log-Transformed Approach:\n")
  cat("  Correlation:", round(results_log$correlation, 3), "\n")
  cat("  RMSE:", round(results_log$rmse, 2), "log(days)\n")
  
  cat("Original Scale Approach:\n")
  cat("  Correlation:", round(results_original$correlation, 3), "\n")
  cat("  RMSE:", round(results_original$rmse, 0), "days\n")
  
  # Create comparison plot
  pdf(paste0(output_dir, "/scale_comparison_rf.pdf"), width = 12, height = 6)
  par(mfrow = c(1, 2))
  
  # Log-transformed plot
  plot(results_log$actual, results_log$predictions,
       xlab = "Actual Log(Days to Death)",
       ylab = "Predicted Log(Days to Death)",
       main = "Log-Transformed Approach",
       pch = 19, col = "blue")
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning for log-transformed plot
  x_pos_log <- quantile(results_log$actual, 0.1)
  y_pos_log <- quantile(results_log$predictions, 0.9)
  text(x_pos_log, y_pos_log,
       paste("Corr:", round(results_log$correlation, 3), 
             "\nRMSE:", round(results_log$rmse, 2)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.9, col = "darkblue")
  
  # Original scale plot with bias correction
  actual_orig <- exp(results_log$actual)
  predicted_raw <- exp(results_log$predictions)
  predicted_corrected <- bias_correct_predictions(results_log$predictions, results_log$actual)
  
  plot(actual_orig, predicted_corrected,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Log-Transformed (Bias-Corrected)",
       pch = 19, col = "blue", ylim = range(c(predicted_raw, predicted_corrected, actual_orig)))
  
  # Add raw predictions for comparison
  points(actual_orig, predicted_raw, pch = 3, col = "red", cex = 0.8)
  
  abline(0, 1, col = "red", lty = 2)
  
  # Better text positioning for original scale plot
  x_pos_orig <- quantile(actual_orig, 0.1)
  y_pos_orig <- quantile(predicted_corrected, 0.9)
  
  # Calculate metrics
  corr_raw <- cor(actual_orig, predicted_raw)
  rmse_raw <- sqrt(mean((actual_orig - predicted_raw)^2))
  corr_corrected <- cor(actual_orig, predicted_corrected)
  rmse_corrected <- sqrt(mean((actual_orig - predicted_corrected)^2))
  
  text(x_pos_orig, y_pos_orig,
       paste("Corr (corrected):", round(corr_corrected, 3), 
             "\nRMSE (corrected):", round(rmse_corrected, 0),
             "\nCorr (raw):", round(corr_raw, 3),
             "\nRMSE (raw):", round(rmse_raw, 0)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkblue")
  
  # Add legend
  legend("topleft", legend = c("Bias-corrected", "Raw (exp)"), 
         pch = c(19, 3), col = c("blue", "red"), cex = 0.8)
  
  dev.off()
  
  cat("\nScale comparison plot saved to: scale_comparison_rf.pdf\n")
  cat("\nBias Correction Summary:\n")
  cat("Raw exponential back-transformation:\n")
  cat("  Correlation:", round(corr_raw, 3), "RMSE:", round(rmse_raw, 0), "days\n")
  cat("Bias-corrected back-transformation:\n")
  cat("  Correlation:", round(corr_corrected, 3), "RMSE:", round(rmse_corrected, 0), "days\n")
  
  return(list(
    log_approach = results_log,
    original_approach = results_original,
    bias_correction = list(
      raw = list(correlation = corr_raw, rmse = rmse_raw),
      corrected = list(correlation = corr_corrected, rmse = rmse_corrected)
    )
  ))
}

compare_transformation_approaches <- function(clinica_data, xcell_data_list, output_dir = "results") {
  # Compare different transformation approaches
  
  cat("\nComparing Different Transformation Approaches:\n")
  cat("=============================================\n")
  
  # Prepare dataset with multiple transformations
  data_transforms <- prepare_xcell_dataset_alternative_transforms(
    clinica_data, xcell_data_list$data, xcell_data_list$cell_types
  )
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Test with Random Forest
  method <- "rf"
  
  # Define transformations to test
  transformations <- list(
    log = "log_days_to_death",
    sqrt = "sqrt_days_to_death", 
    cube_root = "cube_root_days_to_death",
    original = "days_to_death.demographic"
  )
  
  results_transforms <- list()
  
  for (transform_name in names(transformations)) {
    target_var <- transformations[[transform_name]]
    cat("Testing", transform_name, "transformation...\n")
    
    # Train model
    results <- train_and_evaluate_model(data_transforms, method, ctrl, target_var)
    results_transforms[[transform_name]] <- results
  }
  
  # Create comparison plot
  pdf(paste0(output_dir, "/transformation_comparison.pdf"), width = 16, height = 12)
  par(mfrow = c(2, 2))
  
  for (transform_name in names(transformations)) {
    results <- results_transforms[[transform_name]]
    
    # Back-transform predictions to original scale for comparison
    if (transform_name == "log") {
      actual_orig <- exp(results$actual)
      pred_orig <- exp(results$predictions)
    } else if (transform_name == "sqrt") {
      actual_orig <- results$actual^2
      pred_orig <- results$predictions^2
    } else if (transform_name == "cube_root") {
      actual_orig <- results$actual^3
      pred_orig <- results$predictions^3
    } else {
      actual_orig <- results$actual
      pred_orig <- results$predictions
    }
    
    # Calculate metrics in original scale
    corr_orig <- cor(actual_orig, pred_orig)
    rmse_orig <- sqrt(mean((actual_orig - pred_orig)^2))
    
    plot(actual_orig, pred_orig,
         xlab = "Actual Days to Death",
         ylab = "Predicted Days to Death",
         main = paste(toupper(transform_name), "Transformation"),
         pch = 19, col = "blue")
    abline(0, 1, col = "red", lty = 2)
    
    text(quantile(actual_orig, 0.1), quantile(pred_orig, 0.9),
         paste("Corr:", round(corr_orig, 3), "\nRMSE:", round(rmse_orig, 0)),
         pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkblue")
  }
  
  dev.off()
  
  # Print comparison summary
  cat("\nTransformation Comparison Summary:\n")
  cat("==================================\n")
  for (transform_name in names(transformations)) {
    results <- results_transforms[[transform_name]]
    cat(sprintf("%-15s: Log-scale Corr = %.3f, RMSE = %.2f\n", 
                transform_name, results$correlation, results$rmse))
  }
  
  return(results_transforms)
}

# =============================================================================
# Advanced Model Improvements
# =============================================================================

create_feature_selected_model <- function(data, method, importance_df, top_n = 50, output_dir = "results") {
  # Create model using only top N most important features
  
  # Get top N features
  top_features <- head(importance_df$Variable, top_n)
  
  # Prepare dataset with selected features
  selected_data <- data[, c("log_days_to_death", "days_to_death.demographic", top_features)]
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Train model with selected features
  cat("Training", method, "with top", top_n, "features...\n")
  results <- train_and_evaluate_model(selected_data, method, ctrl)
  
  # Create scatter plot
  additional_metrics <- create_scatter_plot(results, method)
  
  # Add metrics to results
  results$correlation_original_raw <- additional_metrics$correlation_original_raw
  results$rmse_original_raw <- additional_metrics$rmse_original_raw
  results$correlation_original_corrected <- additional_metrics$correlation_original_corrected
  results$rmse_original_corrected <- additional_metrics$rmse_original_corrected
  
  # Enhanced bias correction
  enhanced_results <- create_enhanced_scatter_plot(results, method)
  results$best_bias_correction <- enhanced_results$best_method
  results$enhanced_metrics <- enhanced_results$metrics
  
  # Save feature list
  write.csv(data.frame(Feature = top_features), 
            paste0(output_dir, "/top_", top_n, "_features_", method, ".csv"), 
            row.names = FALSE)
  
  return(results)
}

create_ensemble_model <- function(data, output_dir = "results") {
  # Create ensemble model combining multiple algorithms
  
  cat("\nCreating Ensemble Model:\n")
  cat("=======================\n")
  
  # Set up training control
  ctrl <- setup_train_control()
  
  # Train individual models
  ensemble_models <- list()
  predictions_list <- list()
  
  for (method in METHODS) {
    cat("Training", method, "for ensemble...\n")
    results <- train_and_evaluate_model(data, method, ctrl)
    ensemble_models[[method]] <- results$model
    predictions_list[[method]] <- results$predictions
  }
  
  # Create ensemble predictions (simple average)
  test_indices <- ensemble_models[[1]]$control$index[[1]]
  test_indices <- setdiff(1:nrow(data), test_indices)
  
  # Get predictions from all models
  pred_matrix <- do.call(cbind, predictions_list)
  ensemble_predictions <- rowMeans(pred_matrix)
  
  # Get actual values
  actual_values <- data$log_days_to_death[test_indices]
  
  # Calculate ensemble metrics
  correlation <- cor(actual_values, ensemble_predictions)
  rmse <- sqrt(mean((actual_values - ensemble_predictions)^2))
  
  # Create ensemble results object
  ensemble_results <- list(
    predictions = ensemble_predictions,
    actual = actual_values,
    correlation = correlation,
    rmse = rmse,
    test_indices = test_indices,
    individual_predictions = predictions_list
  )
  
  # Create scatter plot for ensemble
  additional_metrics <- create_scatter_plot(ensemble_results, "ensemble")
  ensemble_results$correlation_original_raw <- additional_metrics$correlation_original_raw
  ensemble_results$rmse_original_raw <- additional_metrics$rmse_original_raw
  ensemble_results$correlation_original_corrected <- additional_metrics$correlation_original_corrected
  ensemble_results$rmse_original_corrected <- additional_metrics$rmse_original_corrected
  
  # Enhanced bias correction for ensemble
  enhanced_results <- create_enhanced_scatter_plot(ensemble_results, "ensemble")
  ensemble_results$best_bias_correction <- enhanced_results$best_method
  ensemble_results$enhanced_metrics <- enhanced_results$metrics
  
  # Create ensemble comparison plot
  pdf(paste0(output_dir, "/ensemble_comparison.pdf"), width = 14, height = 10)
  par(mfrow = c(2, 3))
  
  # Plot individual models vs ensemble
  for (i in 1:length(METHODS)) {
    method <- METHODS[i]
    plot(actual_values, predictions_list[[method]],
         xlab = "Actual Log(Days to Death)",
         ylab = "Predicted Log(Days to Death)",
         main = paste("Individual:", toupper(method)),
         pch = 19, col = "blue")
    abline(0, 1, col = "red", lty = 2)
    
    # Calculate individual metrics
    ind_corr <- cor(actual_values, predictions_list[[method]])
    ind_rmse <- sqrt(mean((actual_values - predictions_list[[method]])^2))
    
    text(quantile(actual_values, 0.1), quantile(predictions_list[[method]], 0.9),
         paste("Corr:", round(ind_corr, 3), "\nRMSE:", round(ind_rmse, 2)),
         pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkblue")
  }
  
  # Plot ensemble
  plot(actual_values, ensemble_predictions,
       xlab = "Actual Log(Days to Death)",
       ylab = "Predicted Log(Days to Death)",
       main = "Ensemble (Average)",
       pch = 19, col = "red")
  abline(0, 1, col = "red", lty = 2)
  
  text(quantile(actual_values, 0.1), quantile(ensemble_predictions, 0.9),
       paste("Corr:", round(correlation, 3), "\nRMSE:", round(rmse, 2)),
       pos = 4, offset = 0.5, bg = "white", cex = 0.8, col = "darkred")
  
  dev.off()
  
  # Print ensemble summary
  cat("\nEnsemble Model Summary:\n")
  cat("======================\n")
  cat("Ensemble (Average):\n")
  cat("  Correlation:", round(correlation, 3), "\n")
  cat("  RMSE:", round(rmse, 2), "log(days)\n")
  cat("  Correlation (original, corrected):", round(ensemble_results$correlation_original_corrected, 3), "\n")
  cat("  RMSE (original, corrected):", round(ensemble_results$rmse_original_corrected, 0), "days\n")
  
  return(ensemble_results)
}

optimize_bias_correction <- function(results, method, output_dir = "results") {
  # Optimize bias correction by finding the best correction factor
  
  cat("\nOptimizing Bias Correction for", method, ":\n")
  cat("==========================================\n")
  
  # Get original scale values
  actual_original <- exp(results$actual)
  predicted_log <- results$predictions
  
  # Test different correction factors
  correction_factors <- seq(0.5, 2.0, by = 0.1)
  optimization_results <- data.frame(
    factor = correction_factors,
    correlation = numeric(length(correction_factors)),
    rmse = numeric(length(correction_factors))
  )
  
  for (i in 1:length(correction_factors)) {
    factor <- correction_factors[i]
    
    # Apply correction factor
    corrected_predictions <- exp(predicted_log * factor)
    
    # Calculate metrics
    correlation <- cor(actual_original, corrected_predictions)
    rmse <- sqrt(mean((actual_original - corrected_predictions)^2))
    
    optimization_results$correlation[i] <- correlation
    optimization_results$rmse[i] <- rmse
  }
  
  # Find best factor
  best_idx <- which.min(optimization_results$rmse)
  best_factor <- optimization_results$factor[best_idx]
  best_correlation <- optimization_results$correlation[best_idx]
  best_rmse <- optimization_results$rmse[best_idx]
  
  # Create optimization plot
  pdf(paste0(output_dir, "/bias_correction_optimization_", method, ".pdf"), width = 12, height = 5)
  par(mfrow = c(1, 2))
  
  # RMSE plot
  plot(optimization_results$factor, optimization_results$rmse,
       type = "l", col = "blue", lwd = 2,
       xlab = "Correction Factor",
       ylab = "RMSE (days)",
       main = paste("RMSE vs Correction Factor (", method, ")"),
       panel.first = grid())
  points(best_factor, best_rmse, col = "red", pch = 19, cex = 1.5)
  text(best_factor, best_rmse, paste("Best:", round(best_factor, 2)), 
       pos = 3, col = "red", font = 2)
  
  # Correlation plot
  plot(optimization_results$factor, optimization_results$correlation,
       type = "l", col = "green", lwd = 2,
       xlab = "Correction Factor",
       ylab = "Correlation",
       main = paste("Correlation vs Correction Factor (", method, ")"),
       panel.first = grid())
  points(best_factor, best_correlation, col = "red", pch = 19, cex = 1.5)
  text(best_factor, best_correlation, paste("Best:", round(best_correlation, 3)), 
       pos = 1, col = "red", font = 2)
  
  dev.off()
  
  # Apply best correction
  best_corrected_predictions <- exp(predicted_log * best_factor)
  
  # Create final comparison plot
  pdf(paste0(output_dir, "/optimized_bias_correction_", method, ".pdf"), width = 10, height = 8)
  par(mfrow = c(2, 2))
  
  # Original scale (raw)
  plot(actual_original, exp(predicted_log),
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Raw Exponential",
       pch = 19, col = "red")
  abline(0, 1, col = "red", lty = 2)
  
  # Original scale (simple bias correction)
  simple_corrected <- bias_correct_predictions(predicted_log, results$actual)
  plot(actual_original, simple_corrected,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Simple Bias Correction",
       pch = 19, col = "orange")
  abline(0, 1, col = "red", lty = 2)
  
  # Original scale (optimized)
  plot(actual_original, best_corrected_predictions,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = paste("Optimized (factor =", round(best_factor, 2), ")"),
       pch = 19, col = "blue")
  abline(0, 1, col = "red", lty = 2)
  
  # Comparison of all methods
  plot(actual_original, actual_original, type = "l", col = "black", lwd = 2,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "All Methods Comparison",
       ylim = range(c(exp(predicted_log), simple_corrected, best_corrected_predictions, actual_original)))
  points(actual_original, exp(predicted_log), pch = 3, col = "red", cex = 0.8)
  points(actual_original, simple_corrected, pch = 4, col = "orange", cex = 0.8)
  points(actual_original, best_corrected_predictions, pch = 19, col = "blue", cex = 0.8)
  legend("topleft", legend = c("Perfect", "Raw", "Simple", "Optimized"), 
         lty = c(1, NA, NA, NA), pch = c(NA, 3, 4, 19), 
         col = c("black", "red", "orange", "blue"), cex = 0.8)
  
  dev.off()
  
  # Print optimization results
  cat("Optimization Results:\n")
  cat("Best correction factor:", round(best_factor, 3), "\n")
  cat("Best correlation:", round(best_correlation, 3), "\n")
  cat("Best RMSE:", round(best_rmse, 0), "days\n")
  
  return(list(
    best_factor = best_factor,
    best_correlation = best_correlation,
    best_rmse = best_rmse,
    optimization_results = optimization_results,
    best_corrected_predictions = best_corrected_predictions
  ))
}

# =============================================================================
# Main Analysis Pipeline
# =============================================================================

main_analysis <- function() {
  # Main analysis pipeline
  
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
    
    # Create scatter plot and get additional metrics
    additional_metrics <- create_scatter_plot(results, method)
    
    # Add original scale metrics to results
    results$correlation_original_raw <- additional_metrics$correlation_original_raw
    results$rmse_original_raw <- additional_metrics$rmse_original_raw
    results$correlation_original_corrected <- additional_metrics$correlation_original_corrected
    results$rmse_original_corrected <- additional_metrics$rmse_original_corrected
    
    model_results[[method]] <- results
    
    # Analyze variable importance for tree-based models
    if (method %in% c("rf", "gbm")) {
      cat("Analyzing variable importance for", method, "...\n")
      importance_df <- analyze_variable_importance(results$model, method)
      importance_results[[method]] <- importance_df
      
      # Enhanced bias correction analysis for best models
      cat("Performing enhanced bias correction analysis for", method, "...\n")
      enhanced_results <- create_enhanced_scatter_plot(results, method)
      
      # Store the best bias correction method
      results$best_bias_correction <- enhanced_results$best_method
      results$enhanced_metrics <- enhanced_results$metrics
      model_results[[method]] <- results
    }
  }
  
  # Print summary
  print_results_summary(model_results, METHODS)
  
  # Create importance comparison
  create_importance_comparison(importance_results)
  
  # Compare scale approaches
  compare_scale_approaches(clinica_data, xcell_data_list)
  
  # Compare transformation approaches
  compare_transformation_approaches(clinica_data, xcell_data_list)
  
  # Advanced improvements for best performing models
  cat("\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("ADVANCED MODEL IMPROVEMENTS\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  
  # Feature selection for Random Forest (best performing model)
  if ("rf" %in% names(importance_results)) {
    cat("\nPerforming Feature Selection Analysis:\n")
    cat("=====================================\n")
    
    # Test different numbers of top features
    feature_counts <- c(25, 50, 75, 100)
    feature_selection_results <- list()
    
    for (n_features in feature_counts) {
      cat("Testing with top", n_features, "features...\n")
      feature_results <- create_feature_selected_model(
        data_xcell, "rf", importance_results[["rf"]], n_features
      )
      feature_selection_results[[paste0("top_", n_features)]] <- feature_results
    }
    
    # Print feature selection summary
    cat("\nFeature Selection Summary:\n")
    cat("==========================\n")
    for (feature_set in names(feature_selection_results)) {
      results <- feature_selection_results[[feature_set]]
      cat(sprintf("%-15s: Corr = %.3f, RMSE = %.0f days (corrected)\n", 
                  feature_set, 
                  results$correlation_original_corrected, 
                  results$rmse_original_corrected))
    }
  }
  
  # Create ensemble model
  cat("\nCreating Ensemble Model:\n")
  cat("=======================\n")
  ensemble_results <- create_ensemble_model(data_xcell)
  
  # Optimize bias correction for best individual model
  if ("rf" %in% names(model_results)) {
    cat("\nOptimizing Bias Correction:\n")
    cat("===========================\n")
    optimization_results <- optimize_bias_correction(model_results[["rf"]], "rf")
  }
  
  # Return both model results and importance results
  return(list(
    model_results = model_results,
    importance_results = importance_results,
    feature_selection_results = if(exists("feature_selection_results")) feature_selection_results else NULL,
    ensemble_results = ensemble_results,
    optimization_results = if(exists("optimization_results")) optimization_results else NULL
  ))
}

# =============================================================================
# Run Analysis
# =============================================================================

if (!interactive()) {
  # Run the analysis if script is executed directly
  results <- main_analysis()
}
