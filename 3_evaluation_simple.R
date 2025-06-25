# =============================================================================
# Glioma Survival Prediction - Evaluation Script
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

# =============================================================================
# Load Results
# =============================================================================

load_test_results <- function(file_path = "results/test_predictions.csv") {
  # Load test predictions from CSV file
  cat("Loading test results from:", file_path, "\n")
  test_results <- read.csv(file_path)
  
  cat("Test results loaded with", nrow(test_results), "samples\n")
  
  return(test_results)
}

load_model_info <- function(file_path = "results/model_info.txt") {
  # Load model information
  cat("Loading model information from:", file_path, "\n")
  model_info <- readLines(file_path)
  
  # Print model info
  cat("\nModel Information:\n")
  cat("==================\n")
  for (line in model_info) {
    cat(line, "\n")
  }
  
  return(model_info)
}

# =============================================================================
# Evaluation Functions
# =============================================================================

calculate_performance_metrics <- function(predictions, actual) {
  # Calculate various performance metrics
  cat("Calculating performance metrics...\n")
  
  # Basic metrics
  correlation <- cor(actual, predictions)
  rmse <- sqrt(mean((actual - predictions)^2))
  mae <- mean(abs(actual - predictions))
  r_squared <- cor(actual, predictions)^2
  
  # Additional metrics
  mape <- mean(abs((actual - predictions) / actual)) * 100  # Mean Absolute Percentage Error
  
  # Calculate residuals
  residuals <- actual - predictions
  
  # Residual statistics
  mean_residual <- mean(residuals)
  median_residual <- median(residuals)
  residual_sd <- sd(residuals)
  
  # Calculate prediction intervals (assuming normal residuals)
  residual_95_ci <- quantile(residuals, c(0.025, 0.975))
  
  metrics <- list(
    correlation = correlation,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    mape = mape,
    mean_residual = mean_residual,
    median_residual = median_residual,
    residual_sd = residual_sd,
    residual_95_ci = residual_95_ci
  )
  
  return(metrics)
}

create_evaluation_plots <- function(predictions, actual, metrics, output_dir = "results") {
  # Create evaluation plots
  cat("Creating evaluation plots...\n")
  
  # 1. Predicted vs Actual scatter plot
  pdf(paste0(output_dir, "/final_predictions_vs_actual.pdf"), width = 10, height = 8)
  
  par(mfrow = c(2, 2))
  
  # Main scatter plot
  plot(actual, predictions,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = "Random Forest: Predicted vs Actual Survival",
       pch = 19, col = "blue", cex = 0.8)
  
  # Add perfect prediction line
  abline(0, 1, col = "red", lty = 2, lwd = 2)
  
  # Add metrics text
  text(quantile(actual, 0.1), quantile(predictions, 0.9),
       paste("Correlation:", round(metrics$correlation, 3),
             "\nRMSE:", round(metrics$rmse, 0), "days",
             "\nMAE:", round(metrics$mae, 0), "days",
             "\nR²:", round(metrics$r_squared, 3)),
       pos = 4, cex = 0.8, col = "darkblue")
  
  # 2. Residuals plot
  residuals <- actual - predictions
  plot(predictions, residuals,
       xlab = "Predicted Days to Death",
       ylab = "Residuals (Actual - Predicted)",
       main = "Residuals vs Predicted Values",
       pch = 19, col = "green", cex = 0.8)
  
  abline(h = 0, col = "red", lty = 2, lwd = 2)
  
  # Add residual statistics
  text(quantile(predictions, 0.1), quantile(residuals, 0.9),
       paste("Mean residual:", round(metrics$mean_residual, 1),
             "\nResidual SD:", round(metrics$residual_sd, 1)),
       pos = 4, cex = 0.8, col = "darkgreen")
  
  # 3. Residuals histogram
  hist(residuals, breaks = 20,
       xlab = "Residuals",
       ylab = "Frequency",
       main = "Distribution of Residuals",
       col = "lightblue", border = "black")
  
  # Add normal curve
  curve(dnorm(x, mean = mean(residuals), sd = sd(residuals)) * length(residuals) * diff(hist(residuals, plot = FALSE)$breaks)[1],
        add = TRUE, col = "red", lwd = 2)
  
  # 4. Q-Q plot of residuals
  qqnorm(residuals, main = "Q-Q Plot of Residuals")
  qqline(residuals, col = "red", lwd = 2)
  
  dev.off()
  
  cat("Evaluation plots saved to:", paste0(output_dir, "/final_predictions_vs_actual.pdf"), "\n")
}

save_evaluation_results <- function(predictions, actual, metrics, output_dir = "results") {
  # Save evaluation results to files
  cat("Saving evaluation results...\n")
  
  # Save predictions and actual values with additional metrics
  results_df <- data.frame(
    Actual = actual,
    Predicted = predictions,
    Residuals = actual - predictions,
    Absolute_Error = abs(actual - predictions),
    Percentage_Error = abs((actual - predictions) / actual) * 100
  )
  
  write.csv(results_df, paste0(output_dir, "/final_evaluation_results.csv"), row.names = FALSE)
  cat("Evaluation results saved to:", paste0(output_dir, "/final_evaluation_results.csv"), "\n")
  
  # Save performance metrics
  metrics_df <- data.frame(
    Metric = c("Correlation", "RMSE", "MAE", "R_squared", "MAPE", 
               "Mean_Residual", "Median_Residual", "Residual_SD",
               "Residual_95CI_Lower", "Residual_95CI_Upper"),
    Value = c(metrics$correlation, metrics$rmse, metrics$mae, metrics$r_squared, metrics$mape,
              metrics$mean_residual, metrics$median_residual, metrics$residual_sd,
              metrics$residual_95_ci[1], metrics$residual_95_ci[2])
  )
  
  write.csv(metrics_df, paste0(output_dir, "/final_performance_metrics.csv"), row.names = FALSE)
  cat("Performance metrics saved to:", paste0(output_dir, "/final_performance_metrics.csv"), "\n")
  
  # Save detailed evaluation report
  report_file <- paste0(output_dir, "/final_evaluation_report.txt")
  sink(report_file)
  cat("Glioma Survival Prediction - Final Evaluation Report\n")
  cat("===================================================\n")
  cat("Evaluation date:", Sys.Date(), "\n\n")
  
  cat("Test Set Information:\n")
  cat("Number of samples:", length(actual), "\n")
  cat("Mean actual survival:", round(mean(actual), 1), "days\n")
  cat("Median actual survival:", round(median(actual), 1), "days\n")
  cat("Min actual survival:", round(min(actual), 1), "days\n")
  cat("Max actual survival:", round(max(actual), 1), "days\n\n")
  
  cat("Performance Metrics:\n")
  cat("Correlation:", round(metrics$correlation, 4), "\n")
  cat("RMSE:", round(metrics$rmse, 1), "days\n")
  cat("MAE:", round(metrics$mae, 1), "days\n")
  cat("R-squared:", round(metrics$r_squared, 4), "\n")
  cat("MAPE:", round(metrics$mape, 2), "%\n\n")
  
  cat("Residual Analysis:\n")
  cat("Mean residual:", round(metrics$mean_residual, 1), "days\n")
  cat("Median residual:", round(metrics$median_residual, 1), "days\n")
  cat("Residual standard deviation:", round(metrics$residual_sd, 1), "days\n")
  cat("95% residual confidence interval:", round(metrics$residual_95_ci[1], 1), "to", round(metrics$residual_95_ci[2], 1), "days\n")
  
  sink()
  
  cat("Evaluation report saved to:", report_file, "\n")
}

print_evaluation_summary <- function(metrics, n_samples) {
  # Print summary of evaluation results
  cat("\nFinal Evaluation Summary:\n")
  cat("=========================\n")
  cat("Number of test samples:", n_samples, "\n")
  cat("Correlation:", round(metrics$correlation, 3), "\n")
  cat("RMSE:", round(metrics$rmse, 0), "days\n")
  cat("MAE:", round(metrics$mae, 0), "days\n")
  cat("R-squared:", round(metrics$r_squared, 3), "\n")
  cat("MAPE:", round(metrics$mape, 1), "%\n")
  
  cat("\nResidual Analysis:\n")
  cat("Mean residual:", round(metrics$mean_residual, 1), "days\n")
  cat("Residual standard deviation:", round(metrics$residual_sd, 1), "days\n")
  cat("95% residual CI:", round(metrics$residual_95_ci[1], 1), "to", round(metrics$residual_95_ci[2], 1), "days\n")
}

# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

main_evaluation <- function() {
  # Main evaluation pipeline
  
  cat("Glioma Survival Prediction - Evaluation\n")
  cat("=======================================\n\n")
  
  # Load test results
  cat("Step 1: Loading test results...\n")
  test_results <- load_test_results()
  
  # Load model info
  cat("\nStep 2: Loading model information...\n")
  model_info <- load_model_info()
  
  # Extract predictions and actual values
  predictions <- test_results$Predicted
  actual <- test_results$Actual
  
  # Calculate performance metrics
  cat("\nStep 3: Calculating performance metrics...\n")
  metrics <- calculate_performance_metrics(predictions, actual)
  
  # Create evaluation plots
  cat("\nStep 4: Creating evaluation plots...\n")
  create_evaluation_plots(predictions, actual, metrics)
  
  # Save evaluation results
  cat("\nStep 5: Saving evaluation results...\n")
  save_evaluation_results(predictions, actual, metrics)
  
  # Print evaluation summary
  print_evaluation_summary(metrics, length(actual))
  
  # Print final summary
  cat("\n", paste0(rep("=", 50), collapse = ""), "\n")
  cat("EVALUATION COMPLETE\n")
  cat(paste0(rep("=", 50), collapse = ""), "\n")
  cat("Files generated in 'results' directory:\n")
  cat("- final_predictions_vs_actual.pdf (evaluation plots)\n")
  cat("- final_evaluation_results.csv (predictions and residuals)\n")
  cat("- final_performance_metrics.csv (performance metrics)\n")
  cat("- final_evaluation_report.txt (detailed report)\n")
  
  return(list(
    predictions = predictions,
    actual = actual,
    metrics = metrics,
    test_results = test_results
  ))
}

# =============================================================================
# Run Evaluation
# =============================================================================

if (!interactive()) {
  # Run the evaluation if script is executed directly
  results <- main_evaluation()
} 