# =============================================================================
# Glioma Survival Prediction - Regularized Model Prediction Script
# =============================================================================

# Required packages
# install.packages(c("caret", "randomForest", "ggplot2", "dplyr", "glmnet", "e1071"), dependencies = TRUE)

library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)

# =============================================================================
# Load Models and Data
# =============================================================================

# Load the best regularized model
best_model_file <- "results/best_regularized_model.rds"

if (!file.exists(best_model_file)) {
  stop("Best regularized model not found at: ", best_model_file)
}

best_model <- readRDS(best_model_file)
cat("Loaded best regularized model from:", best_model_file, "\n")

# Load all models for comparison
all_models_file <- "results/all_regularized_models.rds"

if (!file.exists(all_models_file)) {
  stop("All regularized models not found at: ", all_models_file)
}

all_models <- readRDS(all_models_file)
cat("Loaded all regularized models from:", all_models_file, "\n")

# Load preprocessing parameters
preprocess_file <- "results/preprocessing_params.rds"

if (!file.exists(preprocess_file)) {
  stop("Preprocessing parameters not found at: ", preprocess_file)
}

preprocess_params <- readRDS(preprocess_file)
cat("Loaded preprocessing parameters from:", preprocess_file, "\n")

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
# Data Preprocessing
# =============================================================================

# Remove zero-variance features from training data (same as during training)
nzv_train <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv_train$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv_train[nzv_train$zeroVar, ])]
  cat("Removed", sum(nzv_train$zeroVar), "zero-variance features from training data\n")
}

# Remove zero-variance features from test data
nzv_test <- nearZeroVar(test_data, saveMetrics = TRUE)
if (any(nzv_test$zeroVar)) {
  test_data <- test_data[, !names(test_data) %in% rownames(nzv_test[nzv_test$zeroVar, ])]
  cat("Removed", sum(nzv_test$zeroVar), "zero-variance features from test data\n")
}

# Ensure test data has same features as training data
missing_features <- setdiff(names(train_data), names(test_data))
if (length(missing_features) > 0) {
  cat("Warning: Test data missing features:", paste(missing_features, collapse = ", "), "\n")
  for (feature in missing_features) {
    test_data[[feature]] <- NA
  }
}

# Reorder test data columns to match training data
test_data <- test_data[, names(train_data)]

# Apply preprocessing to both datasets
train_data_scaled <- predict(preprocess_params, train_data)
test_data_scaled <- predict(preprocess_params, test_data)

# =============================================================================
# Make Predictions with All Models
# =============================================================================

cat("\nMaking predictions with all models...\n")

# Function to make predictions based on model type
make_predictions <- function(model, data, data_scaled) {
  model_name <- names(which(sapply(all_models, function(x) identical(x, model))))
  
  if (model_name %in% c("RF_Simple", "RF_Regularized")) {
    # Random Forest models use unscaled data
    return(predict(model, newdata = data))
  } else {
    # Linear models (Ridge, Lasso, Elastic Net) use scaled data
    return(predict(model, newdata = data_scaled))
  }
}

# Make predictions for all models
predictions <- list()
for (model_name in names(all_models)) {
  model <- all_models[[model_name]]
  
  # Training predictions
  train_pred <- make_predictions(model, train_data, train_data_scaled)
  
  # Test predictions
  test_pred <- make_predictions(model, test_data, test_data_scaled)
  
  predictions[[model_name]] <- list(
    train_pred = train_pred,
    test_pred = test_pred
  )
}

# =============================================================================
# Calculate Performance Metrics for All Models
# =============================================================================

train_actual <- train_data$days_to_death.demographic
test_actual <- test_data$days_to_death.demographic

performance_summary <- data.frame()

for (model_name in names(predictions)) {
  # Training metrics
  train_pred <- predictions[[model_name]]$train_pred
  train_correlation <- cor(train_actual, train_pred)
  train_rmse <- sqrt(mean((train_actual - train_pred)^2))
  train_mae <- mean(abs(train_actual - train_pred))
  train_r_squared <- train_correlation^2
  
  # Test metrics
  test_pred <- predictions[[model_name]]$test_pred
  test_correlation <- cor(test_actual, test_pred)
  test_rmse <- sqrt(mean((test_actual - test_pred)^2))
  test_mae <- mean(abs(test_actual - test_pred))
  test_r_squared <- test_correlation^2
  
  # Add to summary
  performance_summary <- rbind(performance_summary, data.frame(
    Model = model_name,
    Train_Correlation = train_correlation,
    Train_RMSE = train_rmse,
    Train_MAE = train_mae,
    Train_R2 = train_r_squared,
    Test_Correlation = test_correlation,
    Test_RMSE = test_rmse,
    Test_MAE = test_mae,
    Test_R2 = test_r_squared,
    Overfitting_Score = train_r_squared - test_r_squared
  ))
}

# Sort by test R² (best generalization)
performance_summary <- performance_summary[order(-performance_summary$Test_R2), ]

cat("\nModel Performance Summary:\n")
cat("==========================\n")
print(performance_summary)

# =============================================================================
# Create and Save Scatter Plots for Best Model
# =============================================================================

# Get best model based on test performance
best_model_name <- performance_summary$Model[1]
best_predictions <- predictions[[best_model_name]]

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Training set scatter plot
train_scatter_plot <- ggplot(data.frame(Actual = train_actual, Predicted = best_predictions$train_pred), 
                            aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = paste("Regularized Model:", best_model_name, "- Training Set"),
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(train_actual), y = max(best_predictions$train_pred), hjust = 0,
           label = paste("R² =", round(performance_summary$Train_R2[1], 3),
                         "\nRMSE =", round(performance_summary$Train_RMSE[1], 1),
                         "\nMAE =", round(performance_summary$Train_MAE[1], 1)),
           color = "black") +
  theme_minimal()

# Test set scatter plot
test_scatter_plot <- ggplot(data.frame(Actual = test_actual, Predicted = best_predictions$test_pred), 
                           aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = paste("Regularized Model:", best_model_name, "- Test Set"),
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(test_actual), y = max(best_predictions$test_pred), hjust = 0,
           label = paste("R² =", round(performance_summary$Test_R2[1], 3),
                         "\nRMSE =", round(performance_summary$Test_RMSE[1], 1),
                         "\nMAE =", round(performance_summary$Test_MAE[1], 1)),
           color = "black") +
  theme_minimal()

# Display the plots
print(train_scatter_plot)
print(test_scatter_plot)

# Save the plots
ggsave("results/regularized_training_scatter_plot.png", plot = train_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Regularized training scatter plot saved to: results/regularized_training_scatter_plot.png\n")

ggsave("results/regularized_test_scatter_plot.png", plot = test_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Regularized test scatter plot saved to: results/regularized_test_scatter_plot.png\n")

# Also save as PDF
ggsave("results/regularized_training_scatter_plot.pdf", plot = train_scatter_plot, 
       width = 10, height = 8, bg = "white")
ggsave("results/regularized_test_scatter_plot.pdf", plot = test_scatter_plot, 
       width = 10, height = 8, bg = "white")

# =============================================================================
# Save Results
# =============================================================================

# Save performance summary
performance_file <- "results/regularized_performance_summary.csv"
write.csv(performance_summary, performance_file, row.names = FALSE)
cat("Performance summary saved to:", performance_file, "\n")

# Save best model predictions
best_predictions_data <- data.frame(
  Dataset = c(rep("Training", length(train_actual)), rep("Test", length(test_actual))),
  Actual = c(train_actual, test_actual),
  Predicted = c(best_predictions$train_pred, best_predictions$test_pred),
  Residuals = c(train_actual - best_predictions$train_pred, test_actual - best_predictions$test_pred)
)

best_predictions_file <- "results/regularized_best_predictions.csv"
write.csv(best_predictions_data, best_predictions_file, row.names = FALSE)
cat("Best model predictions saved to:", best_predictions_file, "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("REGULARIZED PREDICTION COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Best model:", best_model_name, "\n")
cat("Test R²:", round(performance_summary$Test_R2[1], 3), "\n")
cat("Test RMSE:", round(performance_summary$Test_RMSE[1], 1), "days\n")
cat("Overfitting score:", round(performance_summary$Overfitting_Score[1], 3), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- regularized_performance_summary.csv (all models performance)\n")
cat("- regularized_best_predictions.csv (best model predictions)\n")
cat("- regularized_training_scatter_plot.png/pdf\n")
cat("- regularized_test_scatter_plot.png/pdf\n")
cat("\nRegularized prediction analysis complete.\n") 