# =============================================================================
# Glioma Survival Prediction - Improved Model Training Script
# =============================================================================

# install.packages("corrplot")

# Required packages
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(glmnet)
library(e1071)
library(corrplot)

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
# Enhanced Data Analysis and Preprocessing
# =============================================================================

set.seed(123)

# Analyze target variable
target <- train_data$days_to_death.demographic
cat("\nTarget variable analysis:\n")
cat("Mean:", round(mean(target), 1), "\n")
cat("Median:", round(median(target), 1), "\n")
cat("SD:", round(sd(target), 1), "\n")
cat("CV:", round(sd(target)/mean(target), 3), "\n")

# Enhanced outlier handling - use IQR method for more robust winsorization
q1 <- quantile(target, 0.25)
q3 <- quantile(target, 0.75)
iqr <- q3 - q1
lower_bound <- q1 - 1.5 * iqr
upper_bound <- q3 + 1.5 * iqr

# Use 5th and 95th percentiles for winsorization
q05 <- quantile(target, 0.05)
q95 <- quantile(target, 0.95)

cat("5th percentile:", q05, "\n")
cat("95th percentile:", q95, "\n")
cat("IQR-based bounds:", lower_bound, "to", upper_bound, "\n")

# Winsorize target variable
target_winsorized <- pmin(pmax(target, q05), q95)
train_data$days_to_death.demographic <- target_winsorized

cat("Outliers handled by winsorization\n")

# =============================================================================
# Enhanced Feature Selection and Engineering
# =============================================================================

cat("\nPerforming enhanced feature selection...\n")

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

# 3. Enhanced feature selection with multiple thresholds
cor_threshold_high <- 0.15  # More stringent threshold
cor_threshold_medium <- 0.1  # Original threshold
cor_threshold_low <- 0.05   # Lower threshold for more features

selected_features_high <- names(feature_cors_abs[feature_cors_abs > cor_threshold_high])
selected_features_medium <- names(feature_cors_abs[feature_cors_abs > cor_threshold_medium])
selected_features_low <- names(feature_cors_abs[feature_cors_abs > cor_threshold_low])

cat("Features with |correlation| >", cor_threshold_high, ":", length(selected_features_high), "\n")
cat("Features with |correlation| >", cor_threshold_medium, ":", length(selected_features_medium), "\n")
cat("Features with |correlation| >", cor_threshold_low, ":", length(selected_features_low), "\n")

# 4. Create feature-engineered datasets with different feature sets
train_selected_high <- train_data[, c("days_to_death.demographic", selected_features_high)]
train_selected_medium <- train_data[, c("days_to_death.demographic", selected_features_medium)]
train_selected_low <- train_data[, c("days_to_death.demographic", selected_features_low)]

# 5. Enhanced multicollinearity handling
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

# Apply multicollinearity removal to all feature sets
train_selected_high <- remove_highly_correlated(train_selected_high, 0.8)
train_selected_medium <- remove_highly_correlated(train_selected_medium, 0.8)
train_selected_low <- remove_highly_correlated(train_selected_low, 0.8)

# 6. Create interaction features for the medium feature set (simplified)
if (ncol(train_selected_medium) > 2) {
  cat("Creating interaction features...\n")
  
  # Get top 3 features by correlation (reduced from 5 to avoid too many interactions)
  top_features <- names(sort(feature_cors_abs[names(train_selected_medium)[-1]], decreasing = TRUE)[1:3])
  
  # Create interaction terms (only top 3 interactions to avoid overfitting)
  interaction_count <- 0
  for (i in 1:(length(top_features)-1)) {
    for (j in (i+1):length(top_features)) {
      if (interaction_count < 3) {  # Limit to 3 interactions
        feat1 <- top_features[i]
        feat2 <- top_features[j]
        interaction_name <- paste0(feat1, "_x_", feat2)
        train_selected_medium[[interaction_name]] <- train_selected_medium[[feat1]] * train_selected_medium[[feat2]]
        interaction_count <- interaction_count + 1
      }
    }
  }
  cat("Added", interaction_count, "interaction features\n")
}

# =============================================================================
# Enhanced Data Scaling
# =============================================================================

# Scale features for regularization methods
preprocess_params_high <- preProcess(train_selected_high[,-1], method = c("center", "scale"))
preprocess_params_medium <- preProcess(train_selected_medium[,-1], method = c("center", "scale"))
preprocess_params_low <- preProcess(train_selected_low[,-1], method = c("center", "scale"))

train_scaled_high <- predict(preprocess_params_high, train_selected_high)
train_scaled_medium <- predict(preprocess_params_medium, train_selected_medium)
train_scaled_low <- predict(preprocess_params_low, train_selected_low)

# Save preprocessing parameters (use medium as default)
saveRDS(preprocess_params_medium, "results/improved_preprocessing_params.rds")
cat("Preprocessing parameters saved\n")

# =============================================================================
# Enhanced Model Training with Multiple Approaches
# =============================================================================

cat("\nTraining enhanced models with multiple feature sets...\n")

# Function to train models with different feature sets (simplified and more robust)
train_models_with_features <- function(data, data_scaled, suffix) {
  models <- list()
  
  # Ensure data is valid
  if (nrow(data) < 10 || ncol(data) < 2) {
    cat("Warning: Insufficient data for", suffix, "feature set\n")
    return(models)
  }
  
  # Check for any NA values
  if (any(is.na(data))) {
    cat("Warning: NA values found in", suffix, "feature set, removing rows\n")
    data <- na.omit(data)
    if (nrow(data) < 10) {
      cat("Warning: Insufficient data after removing NA values for", suffix, "feature set\n")
      return(models)
    }
  }
  
  tryCatch({
    # 1. Random Forest with simplified tuning
    cat("Training RF for", suffix, "features...\n")
    rf_model <- train(
      days_to_death.demographic ~ .,
      data = data,
      method = "rf",
      trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
      tuneLength = 3,  # Reduced from 10
      importance = TRUE,
      ntree = 500,  # Reduced from 1000
      mtry = max(1, floor(sqrt(ncol(data) - 1)))
    )
    models[["RF"]] <- rf_model
    cat("RF completed for", suffix, "\n")
  }, error = function(e) {
    cat("Error training RF for", suffix, ":", e$message, "\n")
  })
  
  tryCatch({
    # 2. Ridge Regression
    cat("Training Ridge for", suffix, "features...\n")
    ridge_model <- train(
      days_to_death.demographic ~ .,
      data = data_scaled,
      method = "ridge",
      trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
      tuneLength = 5,  # Reduced from 15
      preProcess = NULL
    )
    models[["Ridge"]] <- ridge_model
    cat("Ridge completed for", suffix, "\n")
  }, error = function(e) {
    cat("Error training Ridge for", suffix, ":", e$message, "\n")
  })
  
  tryCatch({
    # 3. Lasso Regression
    cat("Training Lasso for", suffix, "features...\n")
    lasso_model <- train(
      days_to_death.demographic ~ .,
      data = data_scaled,
      method = "lasso",
      trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
      tuneLength = 5,  # Reduced from 15
      preProcess = NULL
    )
    models[["Lasso"]] <- lasso_model
    cat("Lasso completed for", suffix, "\n")
  }, error = function(e) {
    cat("Error training Lasso for", suffix, ":", e$message, "\n")
  })
  
  tryCatch({
    # 4. Elastic Net
    cat("Training Elastic Net for", suffix, "features...\n")
    elastic_model <- train(
      days_to_death.demographic ~ .,
      data = data_scaled,
      method = "glmnet",
      trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
      tuneLength = 5,  # Reduced from 15
      preProcess = NULL
    )
    models[["Elastic"]] <- elastic_model
    cat("Elastic Net completed for", suffix, "\n")
  }, error = function(e) {
    cat("Error training Elastic Net for", suffix, ":", e$message, "\n")
  })
  
  tryCatch({
    # 5. Support Vector Regression (simplified)
    cat("Training SVR for", suffix, "features...\n")
    svr_model <- train(
      days_to_death.demographic ~ .,
      data = data_scaled,
      method = "svmRadial",
      trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE),
      tuneLength = 3,  # Reduced from 10
      preProcess = NULL
    )
    models[["SVR"]] <- svr_model
    cat("SVR completed for", suffix, "\n")
  }, error = function(e) {
    cat("Error training SVR for", suffix, ":", e$message, "\n")
  })
  
  # Add suffix to model names
  names(models) <- paste0(names(models), "_", suffix)
  
  return(models)
}

# Train models with different feature sets
cat("Training models with high feature set...\n")
models_high <- train_models_with_features(train_selected_high, train_scaled_high, "High")

cat("Training models with medium feature set...\n")
models_medium <- train_models_with_features(train_selected_medium, train_scaled_medium, "Medium")

cat("Training models with low feature set...\n")
models_low <- train_models_with_features(train_selected_low, train_scaled_low, "Low")

# Combine all models
all_models_enhanced <- c(models_high, models_medium, models_low)

# Check if we have any successful models
if (length(all_models_enhanced) == 0) {
  stop("No models were successfully trained. Please check the data and try again.")
}

cat("Successfully trained", length(all_models_enhanced), "models\n")

# =============================================================================
# Enhanced Model Comparison
# =============================================================================

# Compare all models
cat("Comparing models...\n")
model_comparison_enhanced <- resamples(all_models_enhanced)
summary_enhanced <- summary(model_comparison_enhanced)

cat("\nEnhanced Model Comparison (Cross-Validation Results):\n")
cat("====================================================\n")
print(summary_enhanced)

# Find best model overall
best_model_name_enhanced <- names(which.min(summary_enhanced$statistics$RMSE[,"Mean"]))
best_model_enhanced <- all_models_enhanced[[best_model_name_enhanced]]

# Find best model for each feature set
high_models <- grep("_High$", names(all_models_enhanced))
medium_models <- grep("_Medium$", names(all_models_enhanced))
low_models <- grep("_Low$", names(all_models_enhanced))

if (length(high_models) > 0) {
  best_high <- names(which.min(summary_enhanced$statistics$RMSE[high_models, "Mean"]))
} else {
  best_high <- "None"
}

if (length(medium_models) > 0) {
  best_medium <- names(which.min(summary_enhanced$statistics$RMSE[medium_models, "Mean"]))
} else {
  best_medium <- "None"
}

if (length(low_models) > 0) {
  best_low <- names(which.min(summary_enhanced$statistics$RMSE[low_models, "Mean"]))
} else {
  best_low <- "None"
}

cat("\nBest models by feature set:\n")
cat("High features:", best_high, "\n")
cat("Medium features:", best_medium, "\n")
cat("Low features:", best_low, "\n")
cat("Overall best:", best_model_name_enhanced, "\n")

# =============================================================================
# Save Enhanced Models
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Save all enhanced models
saveRDS(all_models_enhanced, "results/improved_models.rds")
cat("All enhanced models saved to: results/improved_models.rds\n")

# Save best model
saveRDS(best_model_enhanced, "results/best_improved_model.rds")
cat("Best enhanced model (", best_model_name_enhanced, ") saved to: results/best_improved_model.rds\n")

# =============================================================================
# Enhanced Feature Importance Analysis
# =============================================================================

cat("\nEnhanced Feature Importance Analysis:\n")
cat("=====================================\n")

# For Random Forest models
rf_models <- all_models_enhanced[grep("^RF_", names(all_models_enhanced))]
if (length(rf_models) > 0) {
  best_rf <- rf_models[[which.min(sapply(rf_models, function(x) x$results$RMSE[which.min(x$results$RMSE)]))]]
  
  imp <- varImp(best_rf)
  importance_data <- imp$importance
  importance_data$Variable <- rownames(importance_data)
  importance_data <- importance_data[order(-importance_data$Overall), ]
  
  # Save variable importance
  write.csv(importance_data, "results/improved_variable_importance.csv", row.names = FALSE)
  cat("Variable importance saved to: results/improved_variable_importance.csv\n")
  
  # Plot importance
  png("results/improved_variable_importance_plot.png", width = 10, height = 8, units = "in", res = 300)
  plot(imp, top = 15, main = "Top 15 Variable Importance (Enhanced RF)")
  dev.off()
  cat("Variable importance plot saved to: results/improved_variable_importance_plot.png\n")
}

# For regularized models
reg_models <- all_models_enhanced[grep("^(Ridge_|Lasso_|Elastic_)", names(all_models_enhanced))]
if (length(reg_models) > 0) {
  best_reg <- reg_models[[which.min(sapply(reg_models, function(x) x$results$RMSE[which.min(x$results$RMSE)]))]]
  
  coef_data <- coef(best_reg$finalModel, s = best_reg$bestTune$lambda)
  coef_df <- data.frame(
    Variable = rownames(coef_data),
    Coefficient = as.numeric(coef_data)
  )
  coef_df <- coef_df[order(-abs(coef_df$Coefficient)), ]
  
  write.csv(coef_df, "results/improved_coefficients.csv", row.names = FALSE)
  cat("Model coefficients saved to: results/improved_coefficients.csv\n")
  
  # Plot coefficients
  png("results/improved_coefficients_plot.png", width = 12, height = 8, units = "in", res = 300)
  barplot(coef_df$Coefficient[1:15], names.arg = coef_df$Variable[1:15], 
          main = paste("Top 15 Coefficients (", names(best_reg), ")"),
          las = 2, cex.names = 0.8)
  dev.off()
  cat("Coefficients plot saved to: results/improved_coefficients_plot.png\n")
}

# =============================================================================
# Save Enhanced Model Information
# =============================================================================

model_info_file <- "results/improved_model_info.txt"
sink(model_info_file)
cat("Enhanced Model Training Information\n")
cat("==================================\n")
cat("Training date:", Sys.Date(), "\n")
cat("Best model:", best_model_name_enhanced, "\n")
cat("Training samples:", nrow(train_selected_medium), "\n")
cat("Feature sets tested:\n")
cat("- High threshold (>0.15):", ncol(train_selected_high)-1, "features\n")
cat("- Medium threshold (>0.1):", ncol(train_selected_medium)-1, "features\n")
cat("- Low threshold (>0.05):", ncol(train_selected_low)-1, "features\n")
cat("Best models by feature set:\n")
cat("- High features:", best_high, "\n")
cat("- Medium features:", best_medium, "\n")
cat("- Low features:", best_low, "\n")
cat("\nEnhanced Model Comparison Summary:\n")
print(summary_enhanced)
sink()
cat("Enhanced model information saved to:", model_info_file, "\n")

# =============================================================================
# Enhanced Training Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("ENHANCED MODEL TRAINING COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Best model:", best_model_name_enhanced, "\n")
cat("CV RMSE:", round(min(summary_enhanced$statistics$RMSE[,"Mean"]), 2), "\n")
cat("CV R²:", round(max(summary_enhanced$statistics$Rsquared[,"Mean"]), 3), "\n")
cat("Number of successful models:", length(all_models_enhanced), "\n")
cat("\nEnhancements made:\n")
cat("- Multiple feature selection thresholds\n")
cat("- Enhanced outlier handling\n")
cat("- Interaction feature creation\n")
cat("- Advanced model tuning\n")
cat("- Robust error handling\n")
cat("\nFiles generated in 'results' directory:\n")
cat("- improved_models.rds (all enhanced models)\n")
cat("- best_improved_model.rds (best performing model)\n")
cat("- improved_preprocessing_params.rds (preprocessing parameters)\n")
cat("- improved_model_info.txt (training information)\n")
cat("- improved_variable_importance.csv (feature importance)\n")
cat("- improved_variable_importance_plot.png (importance plot)\n")
cat("- improved_coefficients.csv (model coefficients)\n")
cat("- improved_coefficients_plot.png (coefficients plot)\n")
cat("\nEnhanced models are ready for prediction using the prediction script.\n") 