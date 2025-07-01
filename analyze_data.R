# Analyze training data to understand low R² values
library(dplyr)

# Load training data
data <- read.csv('results/train_data.csv')

cat("=== DATA ANALYSIS ===\n")
cat("Number of samples:", nrow(data), "\n")
cat("Number of features:", ncol(data)-1, "\n")

# Target variable analysis
target <- data$days_to_death.demographic
cat("\n=== TARGET VARIABLE ANALYSIS ===\n")
cat("Summary:\n")
print(summary(target))
cat("\nRange:", range(target), "\n")
cat("Variance:", var(target), "\n")
cat("Standard deviation:", sd(target), "\n")
cat("Coefficient of variation:", sd(target)/mean(target), "\n")

# Check for outliers
q1 <- quantile(target, 0.25)
q3 <- quantile(target, 0.75)
iqr <- q3 - q1
lower_bound <- q1 - 1.5 * iqr
upper_bound <- q3 + 1.5 * iqr
outliers <- target[target < lower_bound | target > upper_bound]

cat("\n=== OUTLIER ANALYSIS ===\n")
cat("Q1:", q1, "\n")
cat("Q3:", q3, "\n")
cat("IQR:", iqr, "\n")
cat("Lower bound:", lower_bound, "\n")
cat("Upper bound:", upper_bound, "\n")
cat("Number of outliers:", length(outliers), "\n")
cat("Outlier percentage:", round(length(outliers)/length(target)*100, 2), "%\n")

# Feature analysis
features <- data[, -1]  # Remove target column
cat("\n=== FEATURE ANALYSIS ===\n")
cat("Feature summary statistics:\n")
print(summary(features[, 1:5]))  # First 5 features

# Check for zero variance features
zero_var_features <- sapply(features, function(x) var(x) == 0)
cat("\nZero variance features:", sum(zero_var_features), "\n")

# Check for highly correlated features
cor_matrix <- cor(features)
high_cor_pairs <- which(abs(cor_matrix) > 0.95 & cor_matrix != 1, arr.ind = TRUE)
cat("Highly correlated feature pairs (>0.95):", nrow(high_cor_pairs)/2, "\n")

# Check feature-target correlations
feature_target_cors <- sapply(features, function(x) cor(x, target))
cat("\n=== FEATURE-TARGET CORRELATIONS ===\n")
cat("Max correlation:", max(abs(feature_target_cors)), "\n")
cat("Min correlation:", min(feature_target_cors), "\n")
cat("Mean absolute correlation:", mean(abs(feature_target_cors)), "\n")
cat("Number of features with |correlation| > 0.1:", sum(abs(feature_target_cors) > 0.1), "\n")
cat("Number of features with |correlation| > 0.2:", sum(abs(feature_target_cors) > 0.2), "\n")

# Show top correlated features
top_cors <- sort(abs(feature_target_cors), decreasing = TRUE)[1:10]
cat("\nTop 10 features by absolute correlation:\n")
for(i in 1:10) {
  feature_name <- names(top_cors)[i]
  cor_value <- feature_target_cors[feature_name]
  cat(sprintf("%s: %.4f\n", feature_name, cor_value))
} 