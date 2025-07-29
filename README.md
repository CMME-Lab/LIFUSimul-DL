# Acoustic Simulation with Deep Learning for Low-intensity Transcranial Focused Ultrasound Digital Twins

## Overview
This repository contains the official implementation of  **"Acoustic Simulation with Deep Learning for Low-intensity Transcranial Focused Ultrasound Digital Twins**", accepted to the _Workshop on Digital Twin for Healthcare 2025 (at MICCAI 2025)_.

<p align="center">
  <img width="1143" height="534" alt="Image" src="https://github.com/user-attachments/assets/6a35a1e6-1e27-4fb9-a2c4-6d3ef4c99c89"/>
</p>

## Features
- CNN-based Autoencoder and U-Net
- Swin Transformer-based U-Net
- Python codes for training, evaluation, loading dataset
- [Pre-trained model weights](https://drive.google.com/drive/folders/1IT0JOh9819Eo5B5ovuIkgjQV6c1pTUvJ?usp=drive_link)
> Note: Dataset is not provided due to privacy concerns.

## 0. Installation
Clone this repository: `git clone https://github.com/CMME-Lab/LIFUSimul-DL.git`</br>
Install all prerequisites with `pip install -r requirements.txt`

## 1. Preparing dataset
We do not provide the dataset due to privacy concerns.</br>
For your experiments, please prepare the dataset in the following format.
<details>
<summary>Instructions</summary>

* All data must be defined in the HDF5 file format. The keys for each data point within the HDF5 file must be sortable in order by the `natsorted` function.
* The data must be organized sequentially by subject, and it is assumed that each subject has the same number of data points.
    - This is to ensure that the same ratio of train/valid data is extracted for each subject.
    - To modify this behavior, please adjust the `split_dataset` function in dataset.py.
* Place the following files in the parent directory:
    - `ff_train.hdf5`, `ff_test.hdf5` (Acoustic free-field)
    - `ct_train.hdf5`, `ct_test.hdf5` or `mr_train.hdf5`, `mr_test.hdf5` (Skull images)
    - `td_train.hdf5`, `td_test.hdf5` (Transducer placement)
    - `target_train.hdf5`, `target_test.hdf5` (Intracranial acoustic field)</br>
    Afterwards, modify the default value of the `data_path` argument in `config.py` to ensure the model always references the correct dataset location.
> Note: For reproducibility, it is assumed that the transducer placement data has already undergone Fourier feature embedding. Please refer to the `fourier_feature_embed` function in `utils.py` to prepare your data by completing the embedding according to its format.

> Note: Compute maximum and minimum value of your acoustic free-field, and replace the value of `ff_max_value` and `ff_min_value` in `MinMaxScaling` (`utils.py`) for proper scaling.
    


</details>

## 2. Training
Run the training process using `train.py`.
> Example usage :</br>
`python train.py --run_name my_experiments --modality ct --model swin --num_epoch 100 --decay_epoch 100 --init_model --cuda`

## 3. Evaluation
Run the evaluation using `test.py`.
> Example usage :</br>
`python test.py --run_name my_experiments --modality ct --model swin --cuda --plot`

## Authors
- **Minjee Seo**, School of Mathematics and Computing (Computational Science and Engineering), Yonsei University, Seoul, Republic of Korea
- **Minwoo Shin**, Department of Software, Yonsei University, Wonju, Republic of Korea
- **Gunwoo Noh**, School of Mechanical Engineering, Korea University, Seoul, Republic of Korea
- **Seung-Schik Yoo**, Department of Radiology, Brigham and Women's Hospital, Harvard Medical School, Boston, MA, USA
- **Kyungho Yoon**, School of Mathematics and Computing (Computational Science and Engineering), Yonsei University, Seoul, Republic of Korea


## Acknowledgement
The work was supported by the National Research Foundation of Korea (NRF) funded by the Korean government (MSIT) under Grants RS-2024-00335185 and RS-2023-00220762.

## License
MIT License

## Contact
For any queries, please reach out to [Minjee Seo](mailto:islandz@yonsei.ac.kr).
