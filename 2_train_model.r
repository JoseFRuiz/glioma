# =============================================================================
# Glioma Survival Prediction - Simplified Model Training Script
# =============================================================================

# Required packages
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

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
# Data Preprocessing
# =============================================================================

set.seed(123)

# Analyze target variable
target <- train_data$days_to_death.demographic
cat("\nTarget variable analysis:\n")
cat("Mean:", round(mean(target), 1), "\n")
cat("Median:", round(median(target), 1), "\n")
cat("SD:", round(sd(target), 1), "\n")

# Simple outlier handling - use 5th and 95th percentiles
q05 <- quantile(target, 0.05)
q95 <- quantile(target, 0.95)
target_winsorized <- pmin(pmax(target, q05), q95)
train_data$days_to_death.demographic <- target_winsorized

cat("Outliers handled by winsorization (5th-95th percentiles)\n")

# =============================================================================
# Feature Selection
# =============================================================================

cat("\nPerforming feature selection...\n")

# 1. Remove zero-variance features
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  cat("Removed", sum(nzv$zeroVar), "zero-variance features\n")
}

# 2. Calculate feature-target correlations
features <- train_data[, -1]
target <- train_data$days_to_death.demographic
feature_cors <- sapply(features, function(x) cor(x, target, use = "complete.obs"))
feature_cors_abs <- abs(feature_cors)

# 3. Select features with correlation > 0.1
cor_threshold <- 0.1
selected_features <- names(feature_cors_abs[feature_cors_abs > cor_threshold])
cat("Features with |correlation| >", cor_threshold, ":", length(selected_features), "\n")

# 4. Create dataset with selected features
train_selected <- train_data[, c("days_to_death.demographic", selected_features)]

# 5. Remove highly correlated features
remove_highly_correlated <- function(data, threshold = 0.8) {
  if (ncol(data) <= 2) return(data)
  
  cor_matrix <- cor(data[, -1])
  high_cor_pairs <- which(abs(cor_matrix) > threshold & cor_matrix != 1, arr.ind = TRUE)
  
  if (nrow(high_cor_pairs) > 0) {
    cat("Found", nrow(high_cor_pairs)/2, "highly correlated feature pairs (>", threshold, ")\n")
    
    # Remove features with lower correlation to target
    features_to_remove <- c()
    for (i in 1:nrow(high_cor_pairs)) {
      if (high_cor_pairs[i, 1] < high_cor_pairs[i, 2]) {
        feat1 <- colnames(cor_matrix)[high_cor_pairs[i, 1]]
        feat2 <- colnames(cor_matrix)[high_cor_pairs[i, 2]]
        
        cor1 <- abs(feature_cors[feat1])
        cor2 <- abs(feature_cors[feat2])
        
        if (cor1 < cor2) {
          features_to_remove <- c(features_to_remove, feat1)
        } else {
          features_to_remove <- c(features_to_remove, feat2)
        }
      }
    }
    
    features_to_remove <- unique(features_to_remove)
    if (length(features_to_remove) > 0) {
      data <- data[, !names(data) %in% features_to_remove]
      cat("Removed", length(features_to_remove), "features due to multicollinearity\n")
    }
  }
  return(data)
}

train_selected <- remove_highly_correlated(train_selected, threshold = 0.8)

# Save selected features
write.csv(data.frame(Feature = names(train_selected)[-1]), "results/selected_features.csv", row.names = FALSE)

# =============================================================================
# Train Random Forest Model
# =============================================================================

cat("\nTraining Random Forest model...\n")

# Fast Random Forest training with simplified parameters
rf_model <- train(
  days_to_death.demographic ~ .,
  data = train_selected,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
  tuneLength = 3,  # Reduced tuning
  importance = TRUE,
  ntree = 200,  # Reduced number of trees for speed
  mtry = max(1, floor(sqrt(ncol(train_selected) - 1)))
)

cat("Random Forest training completed\n")

# =============================================================================
# Model Evaluation
# =============================================================================

cat("\nModel Performance Summary:\n")
cat("==========================\n")
print(rf_model$results)

# Get best model performance
best_rmse <- min(rf_model$results$RMSE)
best_r2 <- max(rf_model$results$Rsquared)
best_mae <- min(rf_model$results$MAE)

cat("\nBest Model Performance:\n")
cat("RMSE:", round(best_rmse, 2), "\n")
cat("R²:", round(best_r2, 3), "\n")
cat("MAE:", round(best_mae, 2), "\n")

# =============================================================================
# Feature Importance Analysis
# =============================================================================

cat("\nFeature Importance Analysis:\n")
cat("============================\n")

imp <- varImp(rf_model)
importance_data <- imp$importance
importance_data$Variable <- rownames(importance_data)
importance_data <- importance_data[order(-importance_data$Overall), ]

# Save variable importance
write.csv(importance_data, "results/variable_importance.csv", row.names = FALSE)
cat("Variable importance saved to: results/variable_importance.csv\n")

# Separate positive and negative importance
positive_importance <- importance_data[importance_data$Overall > 0, ]
negative_importance <- importance_data[importance_data$Overall < 0, ]

cat("\nPositive importance variables:", nrow(positive_importance), "\n")
if (nrow(positive_importance) > 0) {
  cat("Top 5 positive importance variables:\n")
  print(head(positive_importance, 5))
}

cat("\nNegative importance variables:", nrow(negative_importance), "\n")
if (nrow(negative_importance) > 0) {
  cat("Top 5 negative importance variables:\n")
  print(head(negative_importance, 5))
}

# Save positive and negative importance separately
write.csv(positive_importance, "results/positive_importance.csv", row.names = FALSE)
write.csv(negative_importance, "results/negative_importance.csv", row.names = FALSE)

# Plot importance (focus on positive importance for visualization)
if (nrow(positive_importance) > 0) {
  png("results/variable_importance_plot.png", width = 10, height = 8, units = "in", res = 300)
  plot(imp, top = min(15, nrow(positive_importance)), main = "Top Variable Importance (Random Forest)")
  dev.off()
  cat("Variable importance plot saved to: results/variable_importance_plot.png\n")
}

# =============================================================================
# Save Model
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Save the model
saveRDS(rf_model, "results/best_model.rds")
cat("Best model saved to: results/best_model.rds\n")

# Save model information
model_info_file <- "results/model_info.txt"
sink(model_info_file)
cat("Simplified Model Training Information\n")
cat("====================================\n")
cat("Training date:", Sys.Date(), "\n")
cat("Model type: Random Forest\n")
cat("Training samples:", nrow(train_selected), "\n")
cat("Selected features:", ncol(train_selected) - 1, "\n")
cat("Correlation threshold:", cor_threshold, "\n")
cat("Best RMSE:", round(best_rmse, 2), "\n")
cat("Best R²:", round(best_r2, 3), "\n")
cat("Best MAE:", round(best_mae, 2), "\n")
cat("\nModel Parameters:\n")
cat("Number of trees:", rf_model$finalModel$ntree, "\n")
cat("mtry:", rf_model$finalModel$mtry, "\n")
cat("Cross-validation folds:", 5, "\n")
sink()
cat("Model information saved to:", model_info_file, "\n")

# =============================================================================
# Training Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("SIMPLIFIED MODEL TRAINING COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Model: Random Forest\n")
cat("CV RMSE:", round(best_rmse, 2), "\n")
cat("CV R²:", round(best_r2, 3), "\n")
cat("Selected features:", ncol(train_selected) - 1, "\n")
cat("Positive importance variables:", nrow(positive_importance), "\n")
cat("Negative importance variables:", nrow(negative_importance), "\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- best_model.rds (trained Random Forest model)\n")
cat("- model_info.txt (training information)\n")
cat("- variable_importance.csv (all feature importance)\n")
cat("- positive_importance.csv (positive importance features)\n")
cat("- negative_importance.csv (negative importance features)\n")
cat("- variable_importance_plot.png (importance plot)\n")
cat("- selected_features.csv (selected features)\n")
cat("\nModel is ready for prediction using the prediction script.\n") 