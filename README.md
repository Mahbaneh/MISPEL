# Multi-scanner Image harmonization via Structure Preserving Embedding Learning (MISPEL)
*** We are currently working on this repository. The code and documentation will be upoladed by 5/31/22 the latest.
# Reference to paper: 
Method | Citation | Links 
--- | --- | --- 
MISPEL | Torbati, M.E., Tudorascu, D.L., Minhas, D.S., Maillard, P., DeCarli, C.S. and Hwang, S.J., 2021. Multi-Scanner Harmonization of Paired Neuroimaging Data via Structure Preserving Embedding Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3284-3293). | [Paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/html/Torbati_Multi-Scanner_Harmonization_of_Paired_Neuroimaging_Data_via_Structure_Preserving_Embedding_ICCVW_2021_paper.html) [Poster](https://github.com/Mahbaneh/MISPEL/blob/main/iccv21_CVAMD_Paper15_Final.pdf) [Presentation](https://github.com/Mahbaneh/MISPEL/blob/main/MISPEL_Presentation.pptx)
# Table of content:
[Software requirements](#Software-requirements)\
[MISPEL harmonization](#MISPEL-Harmonization)\
[Structure of input data for MISPEL](#Structure-of-input-data-for-MISPEL)\
[Image preprocessing](#Image-Preprocessing)\
[Running](#Running)


# Software requirements:
Python and Keras. 

# MISPEL harmonization: 
MISPEL is a deep learning harmonization technique for paired data. In paired data, each participant is scanned on multiple scanners to generate a set of images (called paired images), which differ solely due to the scanner effects. In MISPEL, the images of each scanner have their own unit of harmonization, which enables the entire framework to be expandable to multiple scanners. This unit consists of a U-Net encoder to extract the latent embeddings of input (unharmonized) images, followed by a linear decoder to reconstruct harmonized images using the embeddings. For harmonization, MISPEL has a two-step training process: (1) Embedding Learning, and (2) Harmonization. In the Embedding Learning step, all units are trained simultaneously to learn embeddings that are similar across scanners and could be used to reconstruct input unharmonized images. In the Harmonization step, the decoders of all units are trained simultaneously to harmonized images by generating identical images across scanners.

MISPEL was applied to a paired dataset consisting of N = 18 participants, each with T1-weighted (T1-w) MR acquisitions on M = 4 different 3T scanners: General Electric (GE), Philips, Siemens Prisma, and Siemens Trio. The median age was 72 years (range 51-78 years), 44% were males, 44% were cognitively normal, and the remaining had diagnoses of Alzheimer’s disease. 

We compared MISPEL with two commonly used methods of MRI normalization and harmonization: White Stripe (WS) and RAVEL. For evaluating harmonization, we studied the similarity of paired images using three evaluation criteria: (1) visual quality, (2) image similarity, and (3) volumetric similarity. We estimated image similarity using structural similarity index measure (SSIM). For volumetric similarity, we extracted the volumes of gray matter and white matter tissue types. Greater similarity indicates better harmonization.

![This is an image](https://github.com/Mahbaneh/MISPEL/blob/main/pipeline.png)


# Structure of input data for MISPEL:
MISPEL accepts a paired dataset as input data which should include images of "ALL" subjects across "ALL" scanners. We expect this data to be grouped in n (# of scanners) folders which named after the scanner names. Images of each subject should have identical names across these folders. Please refer to [Data](https://github.com/Mahbaneh/MISPEL/tree/main/Data) folder as an example of such data. Please note that our in-house paired data is not publicly available and the data in [Data](https://github.com/Mahbaneh/MISPEL/tree/main/Data) folder is a simulated data to be used as an example of running the code.

# Image preprocessing:
For the preprocessing step please read the 'preprocessing' paragraph in section 4.1 of the [Paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/html/Torbati_Multi-Scanner_Harmonization_of_Paired_Neuroimaging_Data_via_Structure_Preserving_Embedding_ICCVW_2021_paper.html). The steps are: (1) Registration to a template, (2) N4 bias correction, (3) Skull stripping, and (4) Image scaling.
For the first three steps, we used the instruction prepared in the [RAVEL repositoty](https://github.com/Jfortin1/RAVEL). You can run Step 4 by our code setting paparemer ... to True. 


# Running:

