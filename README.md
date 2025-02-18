# GPU-accelerated SIFT-aided source identification of stabilized videos

This is the official code implementation of the "ICIP 2022" paper ["GPU-accelerated SIFT-aided source identification of stabilized videos"](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=C0v9f-cAAAAJ&citation_for_view=C0v9f-cAAAAJ:UeHWp8X0CEIC)

## Requirements

- Download the python libraries of [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) ;
 - if [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) is not already, reorganize the folders such that ```PRNU/CameraFingerprint``` ;
 - Download the Reference Camera Fingerprints [here](https://drive.google.com/drive/folders/1q6FpTvP5FYsgaQf5kbC3vjuT6s8jbmxs?usp=sharing);
 - at least 9G GPU.
## Set up Virtual-Env
```
conda env create -f environment.yml
```
## DivNoise DATASET

Download DivNoise dataset [here](https://divnoise.fotoverifier.eu/).

# Test
Before testing, it is necessary to estimate the scaling parameters (required for resizing the camera fingerprints). For this purpose, just run this command inside the basescaling folder:
```
python main_H1_basescaling.py --videos video_path --fingerprint fingerprint_path
```

## Test a match (H1) hypothesis case
```
nohup python -u main_H1.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H1.log & 
```

## Test a mis-match (H0) hypothesis case
```
nohup python -u main_H0.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H0.log & 
```

## Run both
Edit and Run ```bash runner.sh```

## NOTE:
You need to edit:
- ```PATH_TO_VIDEOS``` changing it with the path to your dataset
- ```PATH_TO_FINGERPRINTS``` changing it with the path to your reference camera fingerprints
- ```PATH_TO_OUTPUT_FOLDER``` changing it with the path to your output folder
- ```N``` chaging it with your GPU ID

# Results of the Paper

Check ["GPU-accelerated SIFT-aided source identification of stabilized videos"](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=C0v9f-cAAAAJ&citation_for_view=C0v9f-cAAAAJ:UeHWp8X0CEIC)

<p align="center">
  <img src="https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/ROC.png">
</p>

![tables](https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/table.png?raw=true)

# Cite Us
If you use this material please cite: 

@inproceedings{montibeller2022gpu, \
  title={GPU-accelerated SIFT-aided source identification of stabilized videos}, \
  author={Montibeller, Andrea and Pasquini, Cecilia and Boato, Giulia and Dell’Anna, Stefano and P{\'e}rez-Gonz{\'a}lez, Fernando}, \
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, \
  pages={2616--2620}, \
  year={2022}, \
  organization={IEEE} \
}

