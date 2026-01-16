#!/bin/bash
# Generate PDF from USER_MANUAL.md
#
# First, install pandoc:
#   sudo apt-get install pandoc texlive-xetex texlive-fonts-recommended
#
# Then run this script:
#   ./generate_pdf.sh

cd "$(dirname "$0")"

if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed."
    echo "Install it with: sudo apt-get install pandoc texlive-xetex texlive-fonts-recommended"
    exit 1
fi

echo "Generating USER_MANUAL.pdf..."
pandoc USER_MANUAL.md \
    -o USER_MANUAL.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V colorlinks=true \
    -V linkcolor=blue \
    --toc \
    --toc-depth=2 \
    -V title="RideView User Manual" \
    -V author="RideView Team"

if [ $? -eq 0 ]; then
    echo "Success! Created USER_MANUAL.pdf"
else
    echo "Failed to generate PDF"
    exit 1
fi
