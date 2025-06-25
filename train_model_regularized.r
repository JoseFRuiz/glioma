# =============================================================================
# Glioma Survival Prediction - Regularized Model Training Script
# =============================================================================

# Required packages
# install.packages(c("caret", "randomForest", "ggplot2", "dplyr", "glmnet", "e1071"), dependencies = TRUE)
# install.packages(c('glmnet', 'e1071'), repos='https://cran.rstudio.com/')

library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)

# =============================================================================
# Load Training Data
# =============================================================================

train_file <- "results/train_data.csv"

if (!file.exists(train_file)) {
  stop("Training file not found at: ", train_file)
}

train_data <- read.csv(train_file)

if (!"days_to_death.demographic" %in% names(train_data)) {
  stop("Missing target column: 'days_to_death.demographic'")
}

cat("Loaded training data with", nrow(train_data), "samples and", ncol(train_data)-1, "features\n")

# =============================================================================
# Data Preprocessing for Regularization
# =============================================================================

set.seed(123)

# Remove zero-variance features
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  cat("Removed", sum(nzv$zeroVar), "zero-variance features\n")
}

# Scale features for regularization (important for Ridge/Lasso)
preprocess_params <- preProcess(train_data[,-1], method = c("center", "scale"))
train_data_scaled <- predict(preprocess_params, train_data)

# Save preprocessing parameters for later use
saveRDS(preprocess_params, "results/preprocessing_params.rds")
cat("Preprocessing parameters saved\n")

# =============================================================================
# Regularized Random Forest Training
# =============================================================================

cat("\nTraining Regularized Random Forest Models...\n")

# 1. Standard Random Forest with reduced complexity
rf_simple <- train(
  days_to_death.demographic ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 3,  # Reduced tuning
  importance = TRUE
)

# 2. Random Forest with more regularization (fewer trees, smaller mtry)
rf_regularized <- train(
  days_to_death.demographic ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = c(2, 3, 5, 8)),  # Smaller mtry values
  ntree = 100,  # Fewer trees
  importance = TRUE
)

# 3. Ridge Regression (L2 regularization)
ridge_model <- train(
  days_to_death.demographic ~ .,
  data = train_data_scaled,
  method = "ridge",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 10,
  preProcess = NULL  # Already preprocessed
)

# 4. Lasso Regression (L1 regularization)
lasso_model <- train(
  days_to_death.demographic ~ .,
  data = train_data_scaled,
  method = "lasso",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 10,
  preProcess = NULL  # Already preprocessed
)

# 5. Elastic Net (L1 + L2 regularization)
elastic_net <- train(
  days_to_death.demographic ~ .,
  data = train_data_scaled,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 10,
  preProcess = NULL  # Already preprocessed
)

# =============================================================================
# Model Comparison
# =============================================================================

models <- list(
  "RF_Simple" = rf_simple,
  "RF_Regularized" = rf_regularized,
  "Ridge" = ridge_model,
  "Lasso" = lasso_model,
  "Elastic_Net" = elastic_net
)

# Compare models using cross-validation
model_comparison <- resamples(models)
summary_model_comparison <- summary(model_comparison)

cat("\nModel Comparison (Cross-Validation Results):\n")
cat("============================================\n")
print(summary_model_comparison)

# =============================================================================
# Save All Models
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Save all models
saveRDS(models, "results/all_regularized_models.rds")
cat("All models saved to: results/all_regularized_models.rds\n")

# Save best model based on CV performance
best_model_name <- names(which.min(summary_model_comparison$statistics$RMSE[,"Mean"]))
best_model <- models[[best_model_name]]

saveRDS(best_model, "results/best_regularized_model.rds")
cat("Best model (", best_model_name, ") saved to: results/best_regularized_model.rds\n")

# =============================================================================
# Feature Selection Analysis
# =============================================================================

cat("\nFeature Selection Analysis:\n")
cat("===========================\n")

# For Random Forest models
if (best_model_name %in% c("RF_Simple", "RF_Regularized")) {
  imp <- varImp(best_model)
  importance_data <- imp$importance
  importance_data$Variable <- rownames(importance_data)
  importance_data <- importance_data[order(-importance_data$Overall), ]
  
  # Save variable importance
  importance_file <- "results/regularized_variable_importance.csv"
  write.csv(importance_data, importance_file, row.names = FALSE)
  cat("Variable importance saved to:", importance_file, "\n")
  
  # Save importance plot
  png("results/regularized_variable_importance_plot.png", width = 10, height = 8, units = "in", res = 300)
  plot(imp, top = 20, main = paste("Top 20 Variable Importance (", best_model_name, ")"))
  dev.off()
  cat("Variable importance plot saved to: results/regularized_variable_importance_plot.png\n")
}

# For regularized linear models
if (best_model_name %in% c("Ridge", "Lasso", "Elastic_Net")) {
  # Extract coefficients
  coef_data <- coef(best_model$finalModel, s = best_model$bestTune$lambda)
  coef_df <- data.frame(
    Variable = rownames(coef_data),
    Coefficient = as.numeric(coef_data)
  )
  coef_df <- coef_df[order(-abs(coef_df$Coefficient)), ]
  
  # Save coefficients
  coef_file <- "results/regularized_coefficients.csv"
  write.csv(coef_df, coef_file, row.names = FALSE)
  cat("Model coefficients saved to:", coef_file, "\n")
  
  # Plot coefficients
  png("results/regularized_coefficients_plot.png", width = 12, height = 8, units = "in", res = 300)
  barplot(coef_df$Coefficient[1:20], names.arg = coef_df$Variable[1:20], 
          main = paste("Top 20 Coefficients (", best_model_name, ")"),
          las = 2, cex.names = 0.8)
  dev.off()
  cat("Coefficients plot saved to: results/regularized_coefficients_plot.png\n")
}

# =============================================================================
# Save Model Information
# =============================================================================

model_info_file <- "results/regularized_model_info.txt"
sink(model_info_file)
cat("Regularized Model Training Information\n")
cat("=====================================\n")
cat("Training date:", Sys.Date(), "\n")
cat("Best model:", best_model_name, "\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Number of features:", ncol(train_data) - 1, "\n")
cat("\nModel Comparison Summary:\n")
print(summary_model_comparison)
sink()
cat("Model information saved to:", model_info_file, "\n")

# =============================================================================
# Training Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("REGULARIZED MODEL TRAINING COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Best model:", best_model_name, "\n")
cat("CV RMSE:", round(min(summary_model_comparison$statistics$RMSE[,"Mean"]), 2), "\n")
cat("CV R²:", round(max(summary_model_comparison$statistics$Rsquared[,"Mean"]), 3), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- all_regularized_models.rds (all trained models)\n")
cat("- best_regularized_model.rds (best performing model)\n")
cat("- preprocessing_params.rds (data preprocessing parameters)\n")
cat("- regularized_model_info.txt (training information)\n")
if (best_model_name %in% c("RF_Simple", "RF_Regularized")) {
  cat("- regularized_variable_importance.csv (feature importance)\n")
  cat("- regularized_variable_importance_plot.png (importance plot)\n")
} else {
  cat("- regularized_coefficients.csv (model coefficients)\n")
  cat("- regularized_coefficients_plot.png (coefficients plot)\n")
}
cat("\nModels are ready for prediction using the prediction script.\n") 