# AURORA

## Installation

1) Clone this repository:
    ```bash
    git clone https://github.com/neuronflow/AURORA
    ```
2) Go into the repository and install:
    ```
    cd AURORA
    pip install -r requirements.txt 
    ```
## Usuage

**run_inference.py**: Example script for single inference. 

***Input: t1_file, t1c_file, t2_file, fla_file***

All 4 input files must be nifti (nii.gz) files containing 3D MRIs. Please ensure that all input images are correctly preprocessed (skullstripped, co-registered, registered on SRI-24, you can use [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit) for that).

***Output: segmentation_file***

Add path to your desired output folder.

***optional Output: whole_network_outputs_file, enhancing_network_outputs_file***

## Citation
when using the software please cite:

```
TODO
```

## Licensing

This project is licensed under the terms of the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.de.html).

Contact us regarding licensing.

## Contact / Feedback / Questions
For topics not fitting in a Github issue:

Florian Kofler
florian.kofler [at] tum.de

Josef Buchner
j.buchner [at] tum.de
