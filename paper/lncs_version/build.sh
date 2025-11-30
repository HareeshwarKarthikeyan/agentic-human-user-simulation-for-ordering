#!/bin/bash

# Build script for NeurIPS 2025 paper
# Run with: ./build.sh

echo "Building NeurIPS 2025 paper..."

# Compile LaTeX document with bibliography
# Step 1: First LaTeX pass
pdflatex lncs_version.tex

# Step 2: Process bibliography
bibtex lncs_version

# Step 3: Second LaTeX pass (to include citations)
pdflatex lncs_version.tex

# Step 4: Third LaTeX pass (to resolve cross-references)
pdflatex lncs_version.tex

echo "Build complete! PDF generated: lncs_version.pdf"


rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg
echo "Auxiliary files cleaned."

