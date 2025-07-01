# =============================================================================
# Glioma Survival Prediction - Model Prediction Script
# =============================================================================

# Required packages
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)

# =============================================================================
# Load Models and Data
# =============================================================================

# Load the best model
best_model_file <- "results/best_model.rds"

if (!file.exists(best_model_file)) {
  stop("Best model not found at: ", best_model_file)
}

best_model <- readRDS(best_model_file)
cat("Loaded best model from:", best_model_file, "\n")

# Load all models for comparison
all_models_file <- "results/models.rds"

if (!file.exists(all_models_file)) {
  stop("All models not found at: ", all_models_file)
}

all_models <- readRDS(all_models_file)
cat("Loaded all models from:", all_models_file, "\n")

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

# Apply scaling for models that need it

# Debugging output for column matching
cat("Columns expected by preprocess_params:\n")
print(names(preprocess_params$mean))
cat("Columns in test_selected (excluding target):\n")
print(names(test_selected)[-1])

train_scaled <- predict(preprocess_params, train_selected)
test_scaled <- predict(preprocess_params, test_selected)

# =============================================================================
# Make Predictions with All Models
# =============================================================================

# Filter models to only those whose required features are present in the data
valid_models <- list()
for (model_name in names(all_models)) {
  model <- all_models[[model_name]]
  # Try to get the features used by the model
  model_features <- tryCatch({
    if (!is.null(model$finalModel$xNames)) {
      model$finalModel$xNames
    } else {
      names(train_selected)[-1]
    }
  }, error = function(e) names(train_selected)[-1])
  # Only keep models where all features are present
  if (all(model_features %in% names(train_selected))) {
    valid_models[[model_name]] <- model
  } else {
    cat("Skipping model", model_name, "due to missing features:\n")
    missing <- setdiff(model_features, names(train_selected))
    print(missing)
  }
}

# Use valid_models instead of all_models in the prediction loop
cat("\nMaking predictions with valid models...\n")
predictions <- list()
for (model_name in names(valid_models)) {
  model <- valid_models[[model_name]]
  tryCatch({
    # Training predictions
    train_pred <- make_predictions(model, train_selected, train_scaled)
    # Test predictions
    test_pred <- make_predictions(model, test_selected, test_scaled)
    predictions[[model_name]] <- list(
      train_pred = train_pred,
      test_pred = test_pred
    )
  }, error = function(e) {
    cat("\nERROR in model:", model_name, "\n")
    print(e)
  })
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

cat("\nModel Performance Summary:\n")
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
  labs(title = paste("Model:", best_model_name, "- Training Set"),
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
  labs(title = paste("Model:", best_model_name, "- Test Set"),
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

# Save best model predictions
best_predictions_data <- data.frame(
  Dataset = c(rep("Training", length(train_actual)), rep("Test", length(test_actual))),
  Actual = c(train_actual, test_actual),
  Predicted = c(best_predictions$train_pred, best_predictions$test_pred),
  Residuals = c(train_actual - best_predictions$train_pred, test_actual - best_predictions$test_pred)
)

best_predictions_file <- "results/best_predictions.csv"
write.csv(best_predictions_data, best_predictions_file, row.names = FALSE)
cat("Best model predictions saved to:", best_predictions_file, "\n")

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
cat("Best model:", best_model_name, "\n")
cat("Test R²:", round(performance_summary$Test_R2[1], 3), "\n")
cat("Test RMSE:", round(performance_summary$Test_RMSE[1], 1), "days\n")
cat("Overfitting score:", round(performance_summary$Overfitting_Score[1], 3), "\n")
cat("Number of selected features:", length(selected_features), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- performance_summary.csv (all models performance)\n")
cat("- best_predictions.csv (best model predictions)\n")
cat("- selected_features.csv (selected features info)\n")
cat("- training_scatter_plot.png/pdf\n")
cat("- test_scatter_plot.png/pdf\n")
cat("\nPrediction analysis complete.\n") 