# Fast Sequence-aware Neural Networks for Intrusion Detection

This repository contains code and resources for developing and testing Sequence-aware Neural Networks for intrusion detection using the IEC 60870-5-104 Intrusion Detection Dataset.

## Create the Conda environment
Create the environment using the provided `spec-file.yml` file:

```sh
  conda env create -f spec-file.yml
  # Verify that the environment was created successfully:
  conda env list
  # Activate the environment
  conda activate base
```
## Repository Structure

- `.gitignore`: Specifies files and directories to be ignored by git.
- `related_src/`: Jupyter notebook for exploratory data analysis on the intrusion detection dataset and several other notebooks developed in the discipline.
- `LICENSE`: License information for the project.
- `LICENSEFLOWMETER`: Additional license information related to flowmeter usage.
- `baselines.ipynb`: Jupyter notebook for baseline tests.
- `baselines_dataset.ipynb`: Jupyter notebook for generating dataset with statistics to run the baselines.
- `constants.py`: Python script defining constants used across the project.
- `fids.ipynb`: Jupyter notebook for the custom model implementation and testing.
- `my_flowmeter.py`: Python script intended for generating features for the baselines dataset.
- `scapy_dataset.ipynb`: Jupyter notebook for pcap processing and dataset generation using Scapy.

## License

This project is licensed under the terms specified in the `LICENSE` file. Additional licensing information for the flowmeter code originally obtained from https://github.com/alekzandr/flowmeter can be found in the `LICENSEFLOWMETER` file.


## About

This project aims to develop Fast Sequence-aware Neural Networks for effective intrusion detection using the IEC 60870-5-104 Intrusion Detection Dataset.
