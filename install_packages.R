# =============================================================================
# Install Required Packages for Glioma Survival Prediction
# =============================================================================

# List of required packages
required_packages <- c(
  "readxl",
  "caret", 
  "randomForest",
  "ggplot2",
  "dplyr",
  "glmnet",
  "e1071",
  "corrplot",
  "xgboost"
)

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", package, "\n")
      install.packages(package, dependencies = TRUE)
      library(package, character.only = TRUE)
      cat("Successfully installed and loaded:", package, "\n")
    } else {
      cat("Package already installed:", package, "\n")
    }
  }
}

# Install packages
cat("Checking and installing required packages...\n")
install_if_missing(required_packages)

cat("\nAll required packages are now available!\n")
cat("You can now run the glioma survival prediction analysis.\n") 