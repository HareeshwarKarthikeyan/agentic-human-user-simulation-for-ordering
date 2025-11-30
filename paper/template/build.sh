#!/bin/bash

# Build script for NeurIPS 2025 paper
# Run with: ./build.sh

echo "Building NeurIPS 2025 paper..."

# Compile LaTeX document (run twice to resolve cross-references)
pdflatex neurips_2025.tex
pdflatex neurips_2025.tex

echo "Build complete! PDF generated: neurips_2025.pdf"


rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz
echo "Auxiliary files cleaned."

