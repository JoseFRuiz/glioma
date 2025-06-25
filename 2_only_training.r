# =============================================================================
# Glioma Survival Prediction - Training Set Evaluation Script
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
# Predict and Evaluate
# =============================================================================

train_pred <- predict(rf_model, newdata = train_data)
train_actual <- train_data$days_to_death.demographic

# Evaluation metrics
correlation <- cor(train_actual, train_pred)
rmse <- sqrt(mean((train_actual - train_pred)^2))
mae <- mean(abs(train_actual - train_pred))
r_squared <- correlation^2

cat("\nPerformance on Training Set:\n")
cat("===========================\n")
cat("Correlation:", round(correlation, 3), "\n")
cat("RMSE:", round(rmse, 1), "days\n")
cat("MAE:", round(mae, 1), "days\n")
cat("R²:", round(r_squared, 3), "\n")

# =============================================================================
# Plot: Actual vs Predicted
# =============================================================================

# Create the scatter plot
scatter_plot <- ggplot(data.frame(Actual = train_actual, Predicted = train_pred), aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest: Predicted vs Actual (Training Set)",
       x = "Actual Days to Death",
       y = "Predicted Days to Death") +
  annotate("text", x = min(train_actual), y = max(train_pred), hjust = 0,
           label = paste("R² =", round(r_squared, 3),
                         "\nRMSE =", round(rmse, 1),
                         "\nMAE =", round(mae, 1)),
           color = "black") +
  theme_minimal()

# Display the plot
print(scatter_plot)

# Save the plot
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

ggsave("results/training_scatter_plot.png", plot = scatter_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")
cat("Scatter plot saved to: results/training_scatter_plot.png\n")

# Also save as PDF for vector graphics
ggsave("results/training_scatter_plot.pdf", plot = scatter_plot, 
       width = 10, height = 8, bg = "white")
cat("Scatter plot saved to: results/training_scatter_plot.pdf\n")

# =============================================================================
# Plot: Variable Importance
# =============================================================================

imp <- varImp(rf_model)
plot(imp, top = 20, main = "Top 20 Variable Importance (Training Set)")
