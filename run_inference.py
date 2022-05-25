from lib import single_inference

single_inference(
    t1_file="*_t1.nii.gz",
    t1c_file="*_t1c.nii.gz",
    t2_file="*_t2.nii.gz",
    fla_file="*_fla.nii.gz",
    segmentation_file="your_segmentation_file.nii.gz",
    whole_network_outputs_file="your_whole_metastasis_file.nii.gz",
    enhancing_network_outputs_file="your_enhancing_metastasis_file.nii.gz",
)
