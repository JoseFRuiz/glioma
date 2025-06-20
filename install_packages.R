# =============================================================================
# Package Installation Script for Glioma Analysis
# =============================================================================

# List of required packages
required_packages <- c("readxl", "caret", "randomForest", "dplyr")

# Function to check if packages are installed
check_and_install_packages <- function(packages) {
  cat("Checking and installing required packages...\n")
  cat("============================================\n\n")
  
  for (package in packages) {
    cat(sprintf("Checking package: %s\n", package))
    
    if (!require(package, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("  Installing %s...\n", package))
      install.packages(package, dependencies = TRUE)
      
      # Check if installation was successful
      if (require(package, character.only = TRUE, quietly = TRUE)) {
        cat(sprintf("  ✓ %s installed successfully\n", package))
      } else {
        cat(sprintf("  ✗ Failed to install %s\n", package))
      }
    } else {
      cat(sprintf("  ✓ %s is already installed\n", package))
    }
    cat("\n")
  }
  
  cat("Package installation check complete!\n")
  cat("====================================\n")
}

# Run the package installation
check_and_install_packages(required_packages)

# Test loading all packages
cat("Testing package loading...\n")
cat("========================\n")

for (package in required_packages) {
  tryCatch({
    library(package, character.only = TRUE)
    cat(sprintf("✓ Successfully loaded %s\n", package))
  }, error = function(e) {
    cat(sprintf("✗ Failed to load %s: %s\n", package, e$message))
  })
}

cat("\nAll packages are ready for use!\n") 