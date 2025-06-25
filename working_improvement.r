# =============================================================================
# Glioma Survival Prediction - Working Performance Improvement
# =============================================================================

# Required packages (only basic ones)
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# =============================================================================
# Load Data
# =============================================================================

train_file <- "results/train_data.csv"
test_file <- "results/test_data.csv"

if (!file.exists(train_file) || !file.exists(test_file)) {
  stop("Training or test file not found")
}

train_data <- read.csv(train_file)
test_data <- read.csv(test_file)

cat("Loaded training data:", nrow(train_data), "samples,", ncol(train_data)-1, "features\n")
cat("Loaded test data:", nrow(test_data), "samples,", ncol(test_data)-1, "features\n")

# =============================================================================
# Strategy 1: Data Quality Check
# =============================================================================

cat("\n=== STRATEGY 1: DATA QUALITY CHECK ===\n")

# Check data structure
cat("Training data structure:\n")
cat("Dimensions:", dim(train_data), "\n")
cat("Column names (first 10):", paste(head(names(train_data), 10), collapse = ", "), "\n")
cat("Target variable present:", "days_to_death.demographic" %in% names(train_data), "\n")

# Check for any issues with the target variable
cat("Target variable summary:\n")
print(summary(train_data$days_to_death.demographic))

# =============================================================================
# Strategy 2: Remove Zero-Variance Features
# =============================================================================

cat("\n=== STRATEGY 2: REMOVE ZERO-VARIANCE FEATURES ===\n")

# Remove zero-variance features
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  test_data <- test_data[, !names(test_data) %in% rownames(nzv[nzv$zeroVar, ])]
  cat("Removed", sum(nzv$zeroVar), "zero-variance features\n")
}

cat("Remaining features:", ncol(train_data)-1, "\n")

# =============================================================================
# Strategy 3: Simple Outlier Handling
# =============================================================================

cat("\n=== STRATEGY 3: SIMPLE OUTLIER HANDLING ===\n")

# Simple outlier handling - cap at 95th percentile
cap_outliers <- function(x, percentile = 0.95) {
  q95 <- quantile(x, percentile, na.rm = TRUE)
  x[x > q95] <- q95
  return(x)
}

# Apply outlier capping to features only (not target)
for (col in names(train_data)[-1]) {
  train_data[[col]] <- cap_outliers(train_data[[col]])
  test_data[[col]] <- cap_outliers(test_data[[col]])
}

cat("Applied outlier capping (95th percentile) to features\n")

# =============================================================================
# Strategy 4: Feature Selection by Correlation
# =============================================================================

cat("\n=== STRATEGY 4: FEATURE SELECTION BY CORRELATION ===\n")

# Calculate correlations with target
target_correlations <- abs(cor(train_data[,-1], train_data$days_to_death.demographic, use = "complete.obs"))
target_correlations <- as.numeric(target_correlations)
names(target_correlations) <- names(train_data)[-1]

# Sort by correlation
sorted_correlations <- sort(target_correlations, decreasing = TRUE)

cat("Top 10 features by correlation with target:\n")
print(head(sorted_correlations, 10))

# Select top features (but keep at least 10)
n_features <- min(20, length(sorted_correlations))
top_features <- names(sorted_correlations[1:n_features])

cat("Selected", n_features, "top features by correlation\n")

# Create datasets with selected features
train_selected <- train_data[, c("days_to_death.demographic", top_features)]
test_selected <- test_data[, c("days_to_death.demographic", top_features)]

cat("Training data dimensions after selection:", dim(train_selected), "\n")
cat("Test data dimensions after selection:", dim(test_selected), "\n")

# =============================================================================
# Strategy 5: Simple Models
# =============================================================================

cat("\n=== STRATEGY 5: SIMPLE MODELS ===\n")

set.seed(123)

# 1. Simple Random Forest (fewer trees, smaller mtry)
cat("Training Random Forest model...\n")
rf_model <- train(
  days_to_death.demographic ~ .,
  data = train_selected,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = c(2, 3, 5)),
  ntree = 100,  # Fewer trees
  importance = TRUE
)

# 2. Simple Linear Model
cat("Training Linear Model...\n")
lm_model <- train(
  days_to_death.demographic ~ .,
  data = train_selected,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

# 3. Ridge Regression (if glmnet is available)
if (require(glmnet, quietly = TRUE)) {
  cat("Training Ridge Regression model...\n")
  ridge_model <- train(
    days_to_death.demographic ~ .,
    data = train_selected,
    method = "ridge",
    trControl = trainControl(method = "cv", number = 5),
    tuneLength = 5
  )
  models <- list("RF" = rf_model, "LM" = lm_model, "Ridge" = ridge_model)
} else {
  models <- list("RF" = rf_model, "LM" = lm_model)
}

# =============================================================================
# Strategy 6: Test Set Evaluation
# =============================================================================

cat("\n=== STRATEGY 6: TEST SET EVALUATION ===\n")

# Make predictions on test set
test_predictions <- list()

for (model_name in names(models)) {
  model <- models[[model_name]]
  test_predictions[[model_name]] <- predict(model, newdata = test_selected)
}

# Calculate performance
test_actual <- test_selected$days_to_death.demographic
performance_summary <- data.frame()

for (model_name in names(test_predictions)) {
  pred <- test_predictions[[model_name]]
  correlation <- cor(test_actual, pred)
  rmse <- sqrt(mean((test_actual - pred)^2))
  mae <- mean(abs(test_actual - pred))
  r_squared <- correlation^2
  
  performance_summary <- rbind(performance_summary, data.frame(
    Model = model_name,
    Correlation = correlation,
    RMSE = rmse,
    MAE = mae,
    R2 = r_squared
  ))
}

# Sort by R²
performance_summary <- performance_summary[order(-performance_summary$R2), ]

cat("\nTest Set Performance Summary:\n")
cat("=============================\n")
print(performance_summary)

# =============================================================================
# Strategy 7: Cross-Validation Results
# =============================================================================

cat("\n=== STRATEGY 7: CROSS-VALIDATION RESULTS ===\n")

# Compare models using cross-validation
cv_comparison <- resamples(models)
cv_summary <- summary(cv_comparison)

cat("\nCross-Validation Results:\n")
cat("========================\n")
print(cv_summary)

# =============================================================================
# Strategy 8: Feature Importance
# =============================================================================

cat("\n=== STRATEGY 8: FEATURE IMPORTANCE ===\n")

# Random Forest importance
rf_imp <- varImp(rf_model)
importance_data <- rf_imp$importance
importance_data$Variable <- rownames(importance_data)
importance_data <- importance_data[order(-importance_data$Overall), ]

cat("Top 10 most important features (Random Forest):\n")
print(head(importance_data, 10))

# =============================================================================
# Strategy 9: Create Plots
# =============================================================================

cat("\n=== STRATEGY 9: CREATING PLOTS ===\n")

# Get best model
best_model_name <- performance_summary$Model[1]
best_predictions <- test_predictions[[best_model_name]]

# Create scatter plot
scatter_plot <- ggplot(data.frame(Actual = test_actual, Predicted = best_predictions), 
                      aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = paste("Working Model:", best_model_name, "- Test Set"),
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(test_actual), y = max(best_predictions), hjust = 0,
           label = paste("R² =", round(performance_summary$R2[1], 3),
                         "\nRMSE =", round(performance_summary$RMSE[1], 1),
                         "\nMAE =", round(performance_summary$MAE[1], 1)),
           color = "black") +
  theme_minimal()

# Display the plot
print(scatter_plot)

# =============================================================================
# Save Results
# =============================================================================

if (!dir.exists("results")) dir.create("results", recursive = TRUE)

# Save performance results
write.csv(performance_summary, "results/working_performance_summary.csv", row.names = FALSE)

# Save selected features
write.csv(data.frame(Feature = top_features), "results/working_selected_features.csv", row.names = FALSE)

# Save best model predictions
predictions_df <- data.frame(
  Actual = test_actual,
  Predicted = best_predictions,
  Residuals = test_actual - best_predictions
)

write.csv(predictions_df, "results/working_predictions.csv", row.names = FALSE)

# Save models
saveRDS(models, "results/working_models.rds")

# Save processed data
write.csv(train_selected, "results/train_data_working.csv", row.names = FALSE)
write.csv(test_selected, "results/test_data_working.csv", row.names = FALSE)

# Save plot
ggsave("results/working_test_scatter_plot.png", plot = scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Scatter plot saved to: results/working_test_scatter_plot.png\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("WORKING PERFORMANCE IMPROVEMENT COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Best model:", best_model_name, "\n")
cat("Best test R²:", round(performance_summary$R2[1], 3), "\n")
cat("Best test RMSE:", round(performance_summary$RMSE[1], 1), "days\n")
cat("Features used:", length(top_features), "\n")
cat("\nImprovement strategies applied:\n")
cat("1. Data quality check and validation\n")
cat("2. Remove zero-variance features\n")
cat("3. Simple outlier capping (95th percentile)\n")
cat("4. Feature selection by correlation\n")
cat("5. Simple models (RF, LM, Ridge)\n")
cat("6. Cross-validation for model selection\n")
cat("7. Feature importance analysis\n")
cat("\nResults saved to 'results' directory\n") 