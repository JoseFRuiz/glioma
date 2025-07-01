# =============================================================================
# Glioma Survival Prediction - Improved Model Prediction Script
# =============================================================================

# Required packages
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)

# =============================================================================
# Load Improved Models and Data
# =============================================================================

# Load the best improved model
best_model_file <- "results/best_improved_model.rds"

if (!file.exists(best_model_file)) {
  stop("Best improved model not found at: ", best_model_file)
}

best_model <- readRDS(best_model_file)
cat("Loaded best improved model from:", best_model_file, "\n")

# Load all improved models for comparison
all_models_file <- "results/improved_models.rds"

if (!file.exists(all_models_file)) {
  stop("All improved models not found at: ", all_models_file)
}

all_models <- readRDS(all_models_file)
cat("Loaded all improved models from:", all_models_file, "\n")

# Load improved preprocessing parameters
preprocess_file <- "results/improved_preprocessing_params.rds"

if (!file.exists(preprocess_file)) {
  stop("Improved preprocessing parameters not found at: ", preprocess_file)
}

preprocess_params <- readRDS(preprocess_file)
cat("Loaded improved preprocessing parameters from:", preprocess_file, "\n")

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
# Apply Same Preprocessing as Training
# =============================================================================

set.seed(123)

# Apply winsorization to target variables (same as training)
q1 <- quantile(train_data$days_to_death.demographic, 0.05)
q99 <- quantile(train_data$days_to_death.demographic, 0.95)

train_data$days_to_death.demographic <- pmin(pmax(train_data$days_to_death.demographic, q1), q99)
test_data$days_to_death.demographic <- pmin(pmax(test_data$days_to_death.demographic, q1), q99)

cat("Applied winsorization to target variables\n")

# Remove zero-variance features
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  test_data <- test_data[, !names(test_data) %in% rownames(nzv[nzv$zeroVar, ])]
  cat("Removed", sum(nzv$zeroVar), "zero-variance features\n")
}

# Feature selection (same as training)
features <- train_data[, -1]
target <- train_data$days_to_death.demographic
feature_cors <- sapply(features, function(x) cor(x, target, use = "complete.obs"))
feature_cors_abs <- abs(feature_cors)

# Select features with meaningful correlations
cor_threshold <- 0.1
selected_features <- names(feature_cors_abs[feature_cors_abs > cor_threshold])

if (length(selected_features) < 5) {
  # If too few features, use top 10 by correlation
  top_features <- names(sort(feature_cors_abs, decreasing = TRUE)[1:10])
  selected_features <- top_features
}

# Create feature-engineered datasets
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

# Apply scaling for models that need it
train_scaled <- predict(preprocess_params, train_selected)
test_scaled <- predict(preprocess_params, test_selected)

# =============================================================================
# Make Predictions with All Improved Models
# =============================================================================

cat("\nMaking predictions with all improved models...\n")

# Function to make predictions based on model type
make_predictions_improved <- function(model, data, data_scaled) {
  model_name <- names(which(sapply(all_models, function(x) identical(x, model))))
  
  if (model_name == "RF_Improved") {
    # Random Forest uses unscaled data
    return(predict(model, newdata = data))
  } else {
    # Other models (Ridge, Lasso, Elastic Net, SVR) use scaled data
    return(predict(model, newdata = data_scaled))
  }
}

# Make predictions for all models
predictions <- list()
for (model_name in names(all_models)) {
  model <- all_models[[model_name]]
  
  # Training predictions
  train_pred <- make_predictions_improved(model, train_selected, train_scaled)
  
  # Test predictions
  test_pred <- make_predictions_improved(model, test_selected, test_scaled)
  
  predictions[[model_name]] <- list(
    train_pred = train_pred,
    test_pred = test_pred
  )
}

# =============================================================================
# Calculate Performance Metrics for All Models
# =============================================================================

train_actual <- train_selected$days_to_death.demographic
test_actual <- test_selected$days_to_death.demographic

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

cat("\nImproved Model Performance Summary:\n")
cat("===================================\n")
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
  labs(title = paste("Improved Model:", best_model_name, "- Training Set"),
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
  labs(title = paste("Improved Model:", best_model_name, "- Test Set"),
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
ggsave("results/improved_training_scatter_plot.png", plot = train_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Improved training scatter plot saved to: results/improved_training_scatter_plot.png\n")

ggsave("results/improved_test_scatter_plot.png", plot = test_scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Improved test scatter plot saved to: results/improved_test_scatter_plot.png\n")

# Also save as PDF
ggsave("results/improved_training_scatter_plot.pdf", plot = train_scatter_plot, 
       width = 10, height = 8, bg = "white")
ggsave("results/improved_test_scatter_plot.pdf", plot = test_scatter_plot, 
       width = 10, height = 8, bg = "white")

# =============================================================================
# Save Results
# =============================================================================

# Save performance summary
performance_file <- "results/improved_performance_summary.csv"
write.csv(performance_summary, performance_file, row.names = FALSE)
cat("Improved performance summary saved to:", performance_file, "\n")

# Save best model predictions
best_predictions_data <- data.frame(
  Dataset = c(rep("Training", length(train_actual)), rep("Test", length(test_actual))),
  Actual = c(train_actual, test_actual),
  Predicted = c(best_predictions$train_pred, best_predictions$test_pred),
  Residuals = c(train_actual - best_predictions$train_pred, test_actual - best_predictions$test_pred)
)

best_predictions_file <- "results/improved_best_predictions.csv"
write.csv(best_predictions_data, best_predictions_file, row.names = FALSE)
cat("Best improved model predictions saved to:", best_predictions_file, "\n")

# Save feature information
feature_info <- data.frame(
  Feature = selected_features,
  Correlation = feature_cors[selected_features],
  Abs_Correlation = feature_cors_abs[selected_features]
)
feature_info <- feature_info[order(-feature_info$Abs_Correlation), ]

feature_file <- "results/improved_selected_features.csv"
write.csv(feature_info, feature_file, row.names = FALSE)
cat("Selected features information saved to:", feature_file, "\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("IMPROVED PREDICTION COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Best model:", best_model_name, "\n")
cat("Test R²:", round(performance_summary$Test_R2[1], 3), "\n")
cat("Test RMSE:", round(performance_summary$Test_RMSE[1], 1), "days\n")
cat("Overfitting score:", round(performance_summary$Overfitting_Score[1], 3), "\n")
cat("Number of selected features:", length(selected_features), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- improved_performance_summary.csv (all models performance)\n")
cat("- improved_best_predictions.csv (best model predictions)\n")
cat("- improved_selected_features.csv (selected features info)\n")
cat("- improved_training_scatter_plot.png/pdf\n")
cat("- improved_test_scatter_plot.png/pdf\n")
cat("\nImproved prediction analysis complete.\n") 