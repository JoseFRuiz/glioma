# =============================================================================
# Glioma Survival Prediction - Model Training Script
# =============================================================================

# Required packages
# install.packages(c("caret", "randomForest", "ggplot2", "dplyr"), dependencies = TRUE)

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
# Train Random Forest Model
# =============================================================================

set.seed(123)

# Optional: Remove zero-variance features
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  train_data <- train_data[, !names(train_data) %in% rownames(nzv[nzv$zeroVar, ])]
  cat("Removed", sum(nzv$zeroVar), "zero-variance features\n")
}

ctrl <- trainControl(method = "none")

rf_model <- train(
  days_to_death.demographic ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  importance = TRUE
)

cat("Model trained on training data only\n")

# =============================================================================
# Save the Trained Model
# =============================================================================

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# Save the model
model_file <- "results/trained_rf_model.rds"
saveRDS(rf_model, model_file)
cat("Trained model saved to:", model_file, "\n")

# Save model information
model_info_file <- "results/model_info.txt"
sink(model_info_file)
cat("Random Forest Model Information\n")
cat("==============================\n")
cat("Training date:", Sys.Date(), "\n")
cat("Number of trees:", rf_model$finalModel$ntree, "\n")
cat("Variables per split (mtry):", rf_model$finalModel$mtry, "\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Number of features:", ncol(train_data) - 1, "\n")
cat("Features used:", paste(names(train_data)[-1], collapse = ", "), "\n")
sink()
cat("Model information saved to:", model_info_file, "\n")

# =============================================================================
# Variable Importance Analysis
# =============================================================================

imp <- varImp(rf_model)
importance_data <- imp$importance
importance_data$Variable <- rownames(importance_data)
importance_data <- importance_data[order(-importance_data$Overall), ]

# Save variable importance to CSV
importance_file <- "results/variable_importance.csv"
write.csv(importance_data, importance_file, row.names = FALSE)
cat("Variable importance saved to:", importance_file, "\n")

# Create and save variable importance plot
importance_plot <- plot(imp, top = 20, main = "Top 20 Variable Importance (Training Set)")

# Save the importance plot
png("results/variable_importance_plot.png", width = 10, height = 8, units = "in", res = 300)
plot(imp, top = 20, main = "Top 20 Variable Importance (Training Set)")
dev.off()
cat("Variable importance plot saved to: results/variable_importance_plot.png\n")

# =============================================================================
# Training Summary
# =============================================================================

cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("MODEL TRAINING COMPLETE\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat("Files generated in 'results' directory:\n")
cat("- trained_rf_model.rds (trained model)\n")
cat("- model_info.txt (model information)\n")
cat("- variable_importance.csv (feature importance)\n")
cat("- variable_importance_plot.png (importance plot)\n")
cat("\nModel is ready for prediction using the prediction script.\n") 