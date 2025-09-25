#!/bin/bash

# Define the image name and tag
IMAGE_NAME="bean-disease-classification-image"
IMAGE_TAG="latest"

# Check if a Python script name was provided as an argument
if [ -z "$1" ]; then
  echo "Error: No Python script specified."
  echo "Usage: $0 <path_to_script>"
  echo "Example: $0 scripts/fetch_data.py"
  exit 1
fi

# The path to the Python script is the first argument
PYTHON_SCRIPT="$1"

echo "=========================================="
echo "Running Python script '$PYTHON_SCRIPT' inside Docker container..."
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "=========================================="

# Run the container with the specified environment variables and script
docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --memory=20g \
  --shm-size=4g \
  --rm \
  "$IMAGE_NAME:$IMAGE_TAG" \
  python "$PYTHON_SCRIPT"

# Check the exit status of the docker run command
if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Python script '$PYTHON_SCRIPT' ran successfully."
  echo "=========================================="
else
  echo "=========================================="
  echo "Error: The Docker command or Python script failed."
  echo "Please check the output above for errors."
  echo "=========================================="
  exit 1
fi