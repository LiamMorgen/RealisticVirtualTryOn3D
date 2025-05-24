#!/bin/bash

# --- Configuration ---
# Set the name of your working directory here.
# All processed data related to this directory will be stored under data/<WORK_DIR_NAME>
# WORK_DIR_NAME="work_test_examples"

# 使用哪个cuda
export CUDA_VISIBLE_DEVICES=1

WORK_DIR_NAME="From_CatVTON"

# --- End Configuration ---

# Construct the full path relative to the 'data' directory
DATA_DIR="data/${WORK_DIR_NAME}"

echo "Using working directory: ${DATA_DIR}"

# Prepare your RGBA images in ${DATA_DIR}/images (or wherever Step 0 puts them)
# Step 0: Convert RGBA image to square images and estimate 2D keypoints
# Assumed to be done beforehand and results are in ${DATA_DIR}/images
echo "Step 0: Convert RGBA image to square images and estimate 2D keypoints"
python tools/centralize_rgba.py -i ${DATA_DIR}/rgba -o ${DATA_DIR}/images --remove-bg
echo "Step 0: Convert RGBA image to square images and estimate 2D keypoints done"


echo "DWpose to get 2d keypoints"
python DWpose.py -i ${DATA_DIR}/images -o ${DATA_DIR}/images
echo "DWpose to get 2d keypoints done"


# ./build/examples/openpose/openpose.bin ......

# Step 1: Fit SMPL-X to the input image, the result will be saved in ${DATA_DIR}/smplx
echo "Step 1: Fitting SMPL-X models..."
python fit.py --opt_orient --opt_betas -i ${DATA_DIR}/images -o ./${DATA_DIR}/smplx
echo "SMPL-X fitting results saved to ./${DATA_DIR}/smplx"

# Step 2: Hallucinated back-view images, the result will be saved in ${DATA_DIR}/back_images
# Note that the hallucination process is stochastic, therefore you may choose the best one manually.
echo "Step 2: Hallucinating back-view images..."
python hallucinate.py --num_validation_image 8 -i ${DATA_DIR} -o ./${DATA_DIR}/back_images
echo "Back-view images saved to ./${DATA_DIR}/back_images"

# Step 3: Reconstruct textured 3D meshes, the result will be saved in ${DATA_DIR}/meshes (implicitly by the script)
# original:
# python reconstruct.py --test_folder data/examples --config recon/config.yaml --resume checkpoints/recon_model.pth
echo "Step 3: Reconstructing textured 3D meshes from ${DATA_DIR}..."
python reconstruct.py --test_folder ${DATA_DIR} --config recon/config.yaml --resume checkpoints/recon_model.pth
echo "Reconstruction results potentially saved in ${DATA_DIR}/meshes or similar, check reconstruct.py output."

echo "Processing complete for directory '${WORK_DIR_NAME}'."