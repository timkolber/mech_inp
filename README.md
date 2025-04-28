# Project: Exploring a Model trained on the synthetic task of predicting legal Gomoves

## Usage

- Install Python 3.11
- Install Python Dependency Manager (https://pdm-project.org/en/latest/)
- Type ```pdm install``` in repository root directory

- Execute data/get_data.sh
- Run parse_data.py to process and filter the data
- Run train_model.py to train the model
- Run validate_model.py to get the accuracy for the model on the test set
- Run train_probe.py to train the probe and get the accuracy on the probing test set