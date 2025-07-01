# =============================================================================
# Glioma Survival Prediction - Simplified Model Prediction Script
# =============================================================================

# Required packages
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# =============================================================================
# Load Model and Data
# =============================================================================

# Load the best model
best_model_file <- "results/best_model.rds"

if (!file.exists(best_model_file)) {
  stop("Best model not found at: ", best_model_file)
}

best_model <- readRDS(best_model_file)
cat("Loaded best model from:", best_model_file, "\n")

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

# Load selected features from training
selected_features_file <- "results/selected_features.csv"
if (!file.exists(selected_features_file)) {
  stop("Selected features file not found at: ", selected_features_file)
}
selected_features <- read.csv(selected_features_file)$Feature

# Subset train and test data to these features
missing_features <- setdiff(selected_features, names(train_data))
if (length(missing_features) > 0) {
  cat("\nERROR: The following selected features are missing from train_data:\n")
  print(missing_features)
  stop("Aborting due to missing features. Please re-run 2_train_model.R to regenerate selected_features.csv.")
}
train_selected <- train_data[, c("days_to_death.demographic", selected_features)]
test_selected <- test_data[, c("days_to_death.demographic", selected_features)]

# Ensure test data has same features as training data
missing_features <- setdiff(names(train_selected), names(test_selected))
if (length(missing_features) > 0) {
  cat("Warning: Test data missing features:", paste(missing_features, collapse = ", "), "\n")
  for (feature in missing_features) {
    test_selected[[feature]] <- NA
  }
}

# Reorder test data columns to match training data
test_selected <- test_selected[, names(train_selected)]

cat("Selected", length(selected_features), "features for prediction\n")

# =============================================================================
# Make Predictions
# =============================================================================

cat("\nMaking predictions with Random Forest model...\n")

# Training predictions
train_pred <- predict(best_model, newdata = train_selected)

# Test predictions
test_pred <- predict(best_model, newdata = test_selected)

cat("Predictions completed\n")

# =============================================================================
# Calculate Performance Metrics
# =============================================================================

train_actual <- train_selected$days_to_death.demographic
test_actual <- test_selected$days_to_death.demographic

# Training metrics
train_correlation <- cor(train_actual, train_pred)
train_rmse <- sqrt(mean((train_actual - train_pred)^2))
train_mae <- mean(abs(train_actual - train_pred))
train_r_squared <- train_correlation^2

# Test metrics
test_correlation <- cor(test_actual, test_pred)
test_rmse <- sqrt(mean((test_actual - test_pred)^2))
test_mae <- mean(abs(test_actual - test_pred))
test_r_squared <- test_correlation^2

# Performance summary
performance_summary <- data.frame(
  Dataset = c("Training", "Test"),
  Correlation = c(train_correlation, test_correlation),
  RMSE = c(train_rmse, test_rmse),
  MAE = c(train_mae, test_mae),
  R2 = c(train_r_squared, test_r_squared)
)

cat("\nModel Performance Summary:\n")
cat("==========================\n")
print(performance_summary)

# =============================================================================
# Create and Save Scatter Plots
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Training set scatter plot
train_scatter_plot <- ggplot(data.frame(Actual = train_actual, Predicted = train_pred), 
                            aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest - Training Set",
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(train_actual), y = max(train_pred), hjust = 0,
           label = paste("R² =", round(train_r_squared, 3),
                         "\nRMSE =", round(train_rmse, 1),
                         "\nMAE =", round(train_mae, 1)),
           color = "black") +
  theme_minimal()

# Test set scatter plot
test_scatter_plot <- ggplot(data.frame(Actual = test_actual, Predicted = test_pred), 
                           aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest - Test Set",
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

# Also save as PDF
ggsave("results/training_scatter_plot.pdf", plot = train_scatter_plot, 
       width = 10, height = 8, bg = "white")
ggsave("results/test_scatter_plot.pdf", plot = test_scatter_plot, 
       width = 10, height = 8, bg = "white")

# =============================================================================
# Save Results
# =============================================================================

# Save performance summary
performance_file <- "results/performance_summary.csv"
write.csv(performance_summary, performance_file, row.names = FALSE)
cat("Performance summary saved to:", performance_file, "\n")

# Save predictions
best_predictions_data <- data.frame(
  Dataset = c(rep("Training", length(train_actual)), rep("Test", length(test_actual))),
  Actual = c(train_actual, test_actual),
  Predicted = c(train_pred, test_pred),
  Residuals = c(train_actual - train_pred, test_actual - test_pred)
)

best_predictions_file <- "results/best_predictions.csv"
write.csv(best_predictions_data, best_predictions_file, row.names = FALSE)
cat("Best model predictions saved to:", best_predictions_file, "\n")

# Save separate CSVs for training and test predictions
train_predictions_df <- data.frame(
  Actual = train_actual,
  Predicted = train_pred,
  Residuals = train_actual - train_pred
)
test_predictions_df <- data.frame(
  Actual = test_actual,
  Predicted = test_pred,
  Residuals = test_actual - test_pred
)

write.csv(train_predictions_df, "results/train_predictions.csv", row.names = FALSE)
cat("Training predictions saved to: results/train_predictions.csv\n")

write.csv(test_predictions_df, "results/test_predictions.csv", row.names = FALSE)
cat("Test predictions saved to: results/test_predictions.csv\n")

# Calculate feature correlations for the selected features
features <- train_selected[, -1]  # Exclude target variable
target <- train_selected$days_to_death.demographic
feature_cors <- sapply(features, function(x) cor(x, target, use = "complete.obs"))
feature_cors_abs <- abs(feature_cors)

# Save feature information
feature_info <- data.frame(
  Feature = selected_features,
  Correlation = feature_cors[selected_features],
  Abs_Correlation = feature_cors_abs[selected_features]
)
feature_info <- feature_info[order(-feature_info$Abs_Correlation), ]

feature_file <- "results/selected_features.csv"
write.csv(feature_info, feature_file, row.names = FALSE)
cat("Selected features information saved to:", feature_file, "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("PREDICTION COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Model: Random Forest\n")
cat("Training R²:", round(train_r_squared, 3), "\n")
cat("Test R²:", round(test_r_squared, 3), "\n")
cat("Training RMSE:", round(train_rmse, 1), "days\n")
cat("Test RMSE:", round(test_rmse, 1), "days\n")
cat("Overfitting score:", round(train_r_squared - test_r_squared, 3), "\n")
cat("Number of selected features:", length(selected_features), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- performance_summary.csv (model performance)\n")
cat("- best_predictions.csv (all predictions)\n")
cat("- train_predictions.csv (training set predictions)\n")
cat("- test_predictions.csv (test set predictions)\n")
cat("- selected_features.csv (selected features info)\n")
cat("- training_scatter_plot.png/pdf\n")
cat("- test_scatter_plot.png/pdf\n")
cat("\nPrediction analysis complete.\n") 