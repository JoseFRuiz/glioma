# Install readxl package if not already installed
install.packages("readxl", dependencies = TRUE)
install.packages("caret")

# Install additional packages if not already installed
install.packages("randomForest", dependencies = TRUE)
install.packages("gbm", dependencies = TRUE)
install.packages("kernlab", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)

# Load required libraries
library(readxl)
library(caret)

# Load additional libraries
library(randomForest)
library(gbm)
library(kernlab)
library(e1071)

# Load clinical data
clinica_data <- read_excel("data/ClinicaGliomasMayo2025.xlsx")

# Load gene expression data
# xcell_gene_tpm <- read_excel("data/xCell_gene_tpm_Mayo2025.xlsx")

# Set up a 4x3 plotting area to show all scatter plots at once

# Scatter plot: days_to_death.demographic vs Age_years_at_diagnosis
plot(clinica_data$days_to_death.demographic, clinica_data$Age_years_at_diagnosis,
     xlab = "Days to Death (Demographic)",
     ylab = "Age at Diagnosis (years)",
     main = "Scatter plot: Days to Death vs Age at Diagnosis",
     pch = 19, col = "blue")


par(mfrow = c(4, 3))

# Boxplot: days_to_death.demographic by Chr_7_gain_Chr_10_loss
boxplot(days_to_death.demographic ~ Chr_7_gain_Chr_10_loss, data = clinica_data,
        xlab = "Chr 7 gain / Chr 10 loss",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by Chr 7 gain / Chr 10 loss",
        col = "red")

# Boxplot: days_to_death.demographic by ATRX_status
boxplot(days_to_death.demographic ~ ATRX_status, data = clinica_data,
        xlab = "ATRX status",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by ATRX status",
        col = "blue")

# Boxplot: days_to_death.demographic by EGFR_CNV
boxplot(days_to_death.demographic ~ EGFR_CNV, data = clinica_data,
        xlab = "EGFR CNV",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by EGFR CNV",
        col = "purple")

# Boxplot: days_to_death.demographic by CDKN2A_CNV
boxplot(days_to_death.demographic ~ CDKN2A_CNV, data = clinica_data,
        xlab = "CDKN2A CNV",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by CDKN2A CNV",
        col = "orange")

# Boxplot: days_to_death.demographic by CDKN2B_CNV
boxplot(days_to_death.demographic ~ CDKN2B_CNV, data = clinica_data,
        xlab = "CDKN2B CNV",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by CDKN2B CNV",
        col = "brown")

# Boxplot: days_to_death.demographic by MGMT_promoter_status
boxplot(days_to_death.demographic ~ MGMT_promoter_status, data = clinica_data,
        xlab = "MGMT promoter status",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by MGMT promoter status",
        col = "pink")

# Boxplot: days_to_death.demographic by Category_Table_S1
boxplot(days_to_death.demographic ~ Category_Table_S1, data = clinica_data,
        xlab = "Category Table S1",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by Category Table S1",
        col = "cyan")

# Boxplot: days_to_death.demographic by gender_demographic
boxplot(days_to_death.demographic ~ gender_demographic, data = clinica_data,
        xlab = "Gender (Demographic)",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by Gender (Demographic)",
        col = "darkgreen")

# Boxplot: days_to_death.demographic by race.demographic
boxplot(days_to_death.demographic ~ race.demographic, data = clinica_data,
        xlab = "Race (Demographic)",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by Race (Demographic)",
        col = "darkred")

# Boxplot: days_to_death.demographic by tumor_descriptor_samples
boxplot(days_to_death.demographic ~ tumor_descriptor_samples, data = clinica_data,
        xlab = "Tumor Descriptor (Samples)",
        ylab = "Days to Death (Demographic)",
        main = "Days to Death by Tumor Descriptor (Samples)",
        col = "darkblue")

par(mfrow = c(1, 1))

# Train and evaluate a regressor using leave-one-out cross-validation
# to predict Age_years_at_diagnosis from selected features

# Select relevant columns
features <- c(
  "Chr_7_gain_Chr_10_loss", "TERT_promoter_status", "EGFR_CNV",
  "CDKN2A_CNV", "CDKN2B_CNV", "MGMT_promoter_status", "Category_Table_S1",
  "Age_years_at_diagnosis", "gender_demographic", "race.demographic", "tumor_descriptor_samples"
)

# Prepare data (remove rows with missing values in relevant columns)
data_model <- clinica_data[, c("days_to_death.demographic", features)]
data_model <- na.omit(data_model)

# Convert categorical variables to factors
cat_vars <- c(
  "Chr_7_gain_Chr_10_loss", "TERT_promoter_status", "EGFR_CNV",
  "CDKN2A_CNV", "CDKN2B_CNV", "MGMT_promoter_status", "Category_Table_S1",
  "gender_demographic", "race.demographic", "tumor_descriptor_samples"
)
data_model[cat_vars] <- lapply(data_model[cat_vars], as.factor)

# Define methods to try
methods_to_try <- c("lm", "rf", "knn", "svmRadial", "gbm")

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
  dir.create("results")
}

# Set up leave-one-out cross-validation
ctrl <- trainControl(method = "LOOCV")

# Store results
model_results <- list()

for (m in methods_to_try) {
  set.seed(123)
  model <- train(
    days_to_death.demographic ~ .,
    data = data_model,
    method = m,
    trControl = ctrl
  )
  predictions <- predict(model, newdata = data_model)
  correlation <- cor(data_model$days_to_death.demographic, predictions)
  rmse <- sqrt(mean((data_model$days_to_death.demographic - predictions)^2))
  model_results[[m]] <- list(model = model, predictions = predictions, correlation = correlation, rmse = rmse)

  # Save scatter plot
  pdf(paste0("results/predicted_vs_actual_survival_", m, ".pdf"))
  # Set margins to ensure text is visible
  par(mar = c(5, 4, 4, 6))
  plot(data_model$days_to_death.demographic, predictions,
       xlab = "Actual Days to Death",
       ylab = "Predicted Days to Death",
       main = paste("Predicted vs Actual Survival (", m, ")"),
       pch = 19, col = "blue")
  abline(0, 1, col = "red", lty = 2)
  
  # Calculate text position to ensure visibility
  x_pos <- min(data_model$days_to_death.demographic)
  y_pos <- max(predictions)
  # Add text with a white background for better visibility
  text(x_pos, y_pos,
       paste("Correlation:", round(correlation, 3), "\nRMSE:", round(rmse, 2), "days"),
       pos = 4, offset = 0.5,
       bg = "white",  # Add white background
       cex = 0.9)     # Slightly smaller text
  dev.off()
}

# Print summary of results
cat("\nModel Performance Summary:\n")
cat("========================\n")
for (m in methods_to_try) {
  cat("\nMethod:", m, "\n")
  cat("Correlation:", round(model_results[[m]]$correlation, 3), "\n")
  cat("RMSE:", round(model_results[[m]]$rmse, 2), "days\n")
}

