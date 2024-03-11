# Set up
```
# Clone the repository.
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A2_MED_Assessment/vj279.git

# Change the working directory to the repository.
cd vj279

# Download the Lung CT Dataset to the repository.
wget https://www.dropbox.com/s/1z3z0j5v3j5x1vz/Lung_CT_Dataset.zip

# Download the model weights to the repository.
wget https://www.dropbox.com/s/1z3z0j5v3j5x1vz/model_weights.h5

# Build the Docker image.
docker build -t medical_image_cw_vj297 .
```


# Training the model

To demo the training of the model on a single case, run the following command:

```
docker run -it medical_imaging_cw
python src/train.py configs/train_config_docker.yml
```
# Testing the model

To demo the testing of the model on a single case using the pretrained unet, run the following command:

`docker run -v medical_image_cw_vj297 python src/test.py configs/test_config_docker.yaml`

# Evaluating the model
TODO:
