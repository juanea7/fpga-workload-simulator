#!/bin/bash

# Clone the Incremental Learning Module repository
#
# Author      : Juan Encinas <juan.encinas@upm.es>
# Date        : November 2024
# Description : This script clones the Incremental Learning Module repository
#               and checks out the specific commit hash.


# Variables
FPGA_MODELING_REPO_URL="https://github.com/juanea7/fpga-modeling"   # Repository URL
COMMIT_HASH=deadbeef                                                # Particular commit hash
FPGA_MODELING_DIR_DIR="fpga-modeling"                               # fpga-modeling directory


# Clone the repository
echo "Cloning the fpga-modeling repository..."
git clone "$FPGA_MODELING_REPO_URL" "$FPGA_MODELING_DIR_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Failed to clone fpga-modeling repository."
    exit 1
fi

echo "Successfully cloned the fpga-modeling repository."

# # Checkout the specific commit
# echo "Checking out commit $COMMIT_HASH..."
# cd "$FPGA_MODELING_DIR_DIR" && git checkout "$COMMIT_HASH"

# if [ $? -ne 0 ]; then
#     echo "Error: Failed to checkout commit $COMMIT_HASH."
#     exit 1
# fi

# echo "Checked out commit $COMMIT_HASH successfully."


# Copy just the Incremental Learning module
echo "Copying the Incremental Learning module..."
cp -r fpga-modeling/incremental_learning incremental_learning

# Copy requirements
echo "Copying requirements..."
cp fpga-modeling/requirements.txt requirements.txt
echo "Remember to install the requirements."

# Remove the fpga-modeling directory
echo "Removing the fpga-modeling directory..."
rm -rf fpga-modeling