# =============================================================================
# Analyze Random Forest Variable Importance Results
# =============================================================================

# Load required libraries
library(dplyr)

# Read the importance results
importance_data <- read.csv("results/rf_variable_importance.csv")

# Analyze the distribution
cat("Random Forest Variable Importance Analysis\n")
cat("=========================================\n\n")

# Basic statistics
total_vars <- nrow(importance_data)
positive_vars <- sum(importance_data$Importance > 0)
negative_vars <- sum(importance_data$Importance <= 0)
mean_importance <- mean(importance_data$Importance)
median_importance <- median(importance_data$Importance)

cat("Summary Statistics:\n")
cat("==================\n")
cat("Total variables:", total_vars, "\n")
cat("Positive importance variables:", positive_vars, "(", round(100*positive_vars/total_vars, 1), "%)\n")
cat("Negative importance variables:", negative_vars, "(", round(100*negative_vars/total_vars, 1), "%)\n")
cat("Mean importance:", round(mean_importance, 3), "\n")
cat("Median importance:", round(median_importance, 3), "\n\n")

# Top positive variables
positive_vars_data <- importance_data[importance_data$Importance > 0, ]
positive_vars_data <- positive_vars_data[order(-positive_vars_data$Importance), ]

cat("Top 10 Most Important Variables (Positive):\n")
cat("==========================================\n")
for (i in 1:min(10, nrow(positive_vars_data))) {
  cat(sprintf("%2d. %-25s: %8.3f\n", i, positive_vars_data$Variable[i], positive_vars_data$Importance[i]))
}

# Most negative variables
negative_vars_data <- importance_data[importance_data$Importance <= 0, ]
negative_vars_data <- negative_vars_data[order(negative_vars_data$Importance), ]

cat("\nWorst 10 Variables (Most Negative):\n")
cat("===================================\n")
for (i in 1:min(10, nrow(negative_vars_data))) {
  cat(sprintf("%2d. %-25s: %8.3f\n", i, negative_vars_data$Variable[i], negative_vars_data$Importance[i]))
}

# Interpretation
cat("\n\nInterpretation of Negative Importance Scores:\n")
cat("=============================================\n")
cat("1. Negative scores mean permuting that variable IMPROVES model performance\n")
cat("2. These variables are either:\n")
cat("   - Adding noise to the model\n")
cat("   - Redundant with other variables\n")
cat("   - Not relevant for survival prediction\n")
cat("3. Consider removing negative importance variables for better model performance\n\n")

# Recommendations
cat("Recommendations:\n")
cat("================\n")
cat("1. Focus on variables with positive importance scores\n")
cat("2. Consider removing variables with negative scores\n")
cat("3. The top positive variables are your most predictive features\n")
cat("4. Variables with scores close to 0 have minimal impact\n")

# Save filtered datasets
write.csv(positive_vars_data, "results/rf_positive_importance_analysis.csv", row.names = FALSE)
write.csv(negative_vars_data, "results/rf_negative_importance_analysis.csv", row.names = FALSE)

cat("\nAnalysis complete! Check the 'results' directory for filtered datasets.\n") 