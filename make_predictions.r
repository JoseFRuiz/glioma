# =============================================================================
# Glioma Survival Prediction - Prediction Script
# =============================================================================

# Required packages
# install.packages(c("caret", "randomForest", "ggplot2", "dplyr"), dependencies = TRUE)

library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# =============================================================================
# Load Trained Model and Data
# =============================================================================

# Load the trained model
model_file <- "results/trained_rf_model.rds"

if (!file.exists(model_file)) {
  stop("Trained model not found at: ", model_file)
}

rf_model <- readRDS(model_file)
cat("Loaded trained model from:", model_file, "\n")

# Load training data
train_file <- "results/train_data.csv"

if (!file.exists(train_file)) {
  stop("Training file not found at: ", train_file)
}

train_data <- read.csv(train_file)

if (!"days_to_death.demographic" %in% names(train_data)) {
  stop("Missing target column: 'days_to_death.demographic'")
}

cat("Loaded training data with", nrow(train_data), "samples and", ncol(train_data)-1, "features\n")

# Load test data
test_file <- "results/test_data.csv"

if (!file.exists(test_file)) {
  stop("Test file not found at: ", test_file)
}

test_data <- read.csv(test_file)

if (!"days_to_death.demographic" %in% names(test_data)) {
  stop("Missing target column: 'days_to_death.demographic'")
}

cat("Loaded test data with", nrow(test_data), "samples and", ncol(test_data)-1, "features\n")

# =============================================================================
# Make Predictions on Training Set
# =============================================================================

# Remove zero-variance features from training data (same as during training)
nzv_train <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv_train$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv_train[nzv_train$zeroVar, ])]
  cat("Removed", sum(nzv_train$zeroVar), "zero-variance features from training data\n")
}

# Make predictions on training data
train_pred <- predict(rf_model, newdata = train_data)
train_actual <- train_data$days_to_death.demographic

# =============================================================================
# Make Predictions on Test Set
# =============================================================================

# Remove zero-variance features from test data (same features as training)
nzv_test <- nearZeroVar(test_data, saveMetrics = TRUE)
if (any(nzv_test$zeroVar)) {
  test_data <- test_data[, !names(test_data) %in% rownames(nzv_test[nzv_test$zeroVar, ])]
  cat("Removed", sum(nzv_test$zeroVar), "zero-variance features from test data\n")
}

# Ensure test data has same features as training data
missing_features <- setdiff(names(train_data), names(test_data))
if (length(missing_features) > 0) {
  cat("Warning: Test data missing features:", paste(missing_features, collapse = ", "), "\n")
  # Add missing features with NA values
  for (feature in missing_features) {
    test_data[[feature]] <- NA
  }
}

# Reorder test data columns to match training data
test_data <- test_data[, names(train_data)]

# Make predictions on test data
test_pred <- predict(rf_model, newdata = test_data)
test_actual <- test_data$days_to_death.demographic

# =============================================================================
# Calculate Performance Metrics
# =============================================================================

# Training set metrics
train_correlation <- cor(train_actual, train_pred)
train_rmse <- sqrt(mean((train_actual - train_pred)^2))
train_mae <- mean(abs(train_actual - train_pred))
train_r_squared <- train_correlation^2

# Test set metrics
test_correlation <- cor(test_actual, test_pred)
test_rmse <- sqrt(mean((test_actual - test_pred)^2))
test_mae <- mean(abs(test_actual - test_pred))
test_r_squared <- test_correlation^2

cat("\nPerformance on Training Set:\n")
cat("===========================\n")
cat("Correlation:", round(train_correlation, 3), "\n")
cat("RMSE:", round(train_rmse, 1), "days\n")
cat("MAE:", round(train_mae, 1), "days\n")
cat("R²:", round(train_r_squared, 3), "\n")

cat("\nPerformance on Test Set:\n")
cat("=======================\n")
cat("Correlation:", round(test_correlation, 3), "\n")
cat("RMSE:", round(test_rmse, 1), "days\n")
cat("MAE:", round(test_mae, 1), "days\n")
cat("R²:", round(test_r_squared, 3), "\n")

# =============================================================================
# Create and Save Scatter Plots
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Training set scatter plot
train_scatter_plot <- ggplot(data.frame(Actual = train_actual, Predicted = train_pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest: Predicted vs Actual (Training Set)",
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(train_actual), y = max(train_pred), hjust = 0,
           label = paste("R² =", round(train_r_squared, 3),
                         "\nRMSE =", round(train_rmse, 1),
                         "\nMAE =", round(train_mae, 1)),
           color = "black") +
  theme_minimal()

# Test set scatter plot
test_scatter_plot <- ggplot(data.frame(Actual = test_actual, Predicted = test_pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest: Predicted vs Actual (Test Set)",
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(test_actual), y = max(test_pred), hjust = 0,
           label = paste("R² =", round(test_r_squared, 3),
                         "\nRMSE =", round(test_rmse, 1),
                         "\nMAE =", round(test_mae, 1)),
           color = "black") +
  theme_minimal()

# Display the plots
print(train_scatter_plot)
print(test_scatter_plot)

# Save the plots
ggsave("results/training_scatter_plot.png", plot = train_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Training scatter plot saved to: results/training_scatter_plot.png\n")

ggsave("results/test_scatter_plot.png", plot = test_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Test scatter plot saved to: results/test_scatter_plot.png\n")

# Also save as PDF for vector graphics
ggsave("results/training_scatter_plot.pdf", plot = train_scatter_plot, 
       width = 10, height = 8, bg = "white")
cat("Training scatter plot saved to: results/training_scatter_plot.pdf\n")

ggsave("results/test_scatter_plot.pdf", plot = test_scatter_plot, 
       width = 10, height = 8, bg = "white")
cat("Test scatter plot saved to: results/test_scatter_plot.pdf\n")

# =============================================================================
# Save Predictions and Results
# =============================================================================

# Save training predictions to CSV
train_predictions_data <- data.frame(
  Actual = train_actual,
  Predicted = train_pred,
  Residuals = train_actual - train_pred
)

train_predictions_file <- "results/training_predictions.csv"
write.csv(train_predictions_data, train_predictions_file, row.names = FALSE)
cat("Training predictions saved to:", train_predictions_file, "\n")

# Save test predictions to CSV
test_predictions_data <- data.frame(
  Actual = test_actual,
  Predicted = test_pred,
  Residuals = test_actual - test_pred
)

test_predictions_file <- "results/test_predictions.csv"
write.csv(test_predictions_data, test_predictions_file, row.names = FALSE)
cat("Test predictions saved to:", test_predictions_file, "\n")

# Save performance metrics for both sets
performance_metrics <- data.frame(
  Dataset = c("Training", "Training", "Training", "Training", "Test", "Test", "Test", "Test"),
  Metric = c("Correlation", "RMSE", "MAE", "R_squared", "Correlation", "RMSE", "MAE", "R_squared"),
  Value = c(train_correlation, train_rmse, train_mae, train_r_squared, 
            test_correlation, test_rmse, test_mae, test_r_squared)
)

metrics_file <- "results/performance_metrics.csv"
write.csv(performance_metrics, metrics_file, row.names = FALSE)
cat("Performance metrics saved to:", metrics_file, "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("PREDICTION COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Files generated in 'results' directory:\n")
cat("- training_scatter_plot.png/pdf (training set scatter plot)\n")
cat("- test_scatter_plot.png/pdf (test set scatter plot)\n")
cat("- training_predictions.csv (training set predictions and residuals)\n")
cat("- test_predictions.csv (test set predictions and residuals)\n")
cat("- performance_metrics.csv (performance metrics for both sets)\n")
cat("\nPrediction analysis complete for both training and test sets.\n") 