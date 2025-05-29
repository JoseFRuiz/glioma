# Install readxl package if not already installed
install.packages("readxl", dependencies = TRUE)

# Load required libraries
library(readxl)

# Load clinical data
clinica_data <- read_excel("data/ClinicaGliomasMayo2025.xlsx")

# Load gene expression data
xcell_gene_tpm <- read_excel("data/xCell_gene_tpm_Mayo2025.xlsx")