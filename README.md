# AURORA
Deep learning models for the manuscript
[Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)

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
    
## Recommended Environment
* CUDA 11.4+
* Python 3.10+
* GPU with at least 8GB of VRAM

further details in requirements.txt

## Usuage

**run_inference.py**: Example script for single inference. 

***Input: t1_file, t1c_file, t2_file, fla_file***

All 4 input files must be nifti (nii.gz) files containing 3D MRIs. Please ensure that all input images are correctly preprocessed (skullstripped, co-registered, registered on SRI-24, you can use [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit) for that).

***Output: segmentation_file***

Add path to your desired output folder.

***optional Output: whole_network_outputs_file, enhancing_network_outputs_file***


## Citation
when using the software please cite https://www.sciencedirect.com/science/article/pii/S0167814022045625

```
@article{buchner2022development,
  title={Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study},
  author={Buchner, Josef A and Kofler, Florian and Etzel, Lucas and Mayinger, Michael and Christ, Sebastian M and Brunner, Thomas B and Wittig, Andrea and Menze, Bj{\"o}rn and Zimmer, Claus and Meyer, Bernhard and others},
  journal={Radiotherapy and Oncology},
  year={2022},
  publisher={Elsevier}
}
```

## Licensing

This project is licensed under the terms of the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.de.html).

Contact us regarding licensing.

## Contact / Feedback / Questions
If possible please open a GitHub issue [here](https://github.com/neuronflow/AURORA/issues).

For inquiries not suitable for GitHub issues:

Florian Kofler
florian.kofler [at] tum.de

Josef Buchner
j.buchner [at] tum.de
