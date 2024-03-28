# Set up

Copy the following commands to the terminal to clone the repository, download the model weights, download the pretrained UNet weights, download the data, and build the Docker image.
```
# Clone the repository.
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A2_MED_Assessment/vj279.git

# Change the working directory to the repository.
cd vj279

# Download the data.
git clone git@github.com:loressa/DataScience_MPhill_practicals.git
mv DataScience_MPhill_practicals/Dataset Dataset
rm -rf DataScience_MPhill_practicals

# Create the environment.
conda env update --file environment.yml

# Activate the environment.
conda activate mphil_medical_imaging_cw

# De identify the data.
python src/de_identify_dataset.py Dataset/Images

# Build the Docker image.
docker build -t medical_imaging_cw .
```


# Training the model
To demo the training of the model on a single case, run the following command:

```
docker run -it -v model_weights:/mphil_medical_imaging_cw/model_weights medical_imaging_cw /bin/bash -c "source activate mphil_medical_imaging_cw && python src/train.py configs/train_config_docker.yml"
```

# Testing the model

To demo the testing of the model on a single case using the pretrained unet, run the following command:
NOTE: NaN dice scores are expected as some slices in the case have no lungs to
segment. In such cases the dice score is always 0, so is set to NaN to avoid
confusion with a true dice score of 0 (empty pred for non empty ground truth).

```
docker run -it -v $(pwd)/model_weights:/mphil_medical_imaging_cw/model_weights -v $(pwd)/model_evaluation:/mphil_medical_imaging_cw/model_evaluation medical_imaging_cw /bin/bash -c "source activate mphil_medical_imaging_cw && python src/test.py configs/test_config_docker.yml"
```

# Evaluating the model

To print the summary statistics of the models performance, run the following command:

Docker:
```
docker run -it -v $(pwd)/model_evaluation:/mphil_medical_imaging_cw/model_evaluation medical_imaging_cw /bin/bash -c "source activate mphil_medical_imaging_cw && python src/eval.py model_evaluation/docker/metrics.csv"
```

Local (to see the box plots):
```
# Ensure environment is activated.
python src/eval.py model_evaluation/docker/metrics.csv
```
