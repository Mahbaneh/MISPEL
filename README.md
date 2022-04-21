# Multi-scanner Image harmonization via Structure Preserving Embedding Learning (MISPEL)
*** We are currently working on this repository. The code and documentation will be upoladed by 5/31/22 the latest.
# Reference to paper: 
Method | Citation | Links 
--- | --- | --- 
MISPEL | Torbati, M.E., Tudorascu, D.L., Minhas, D.S., Maillard, P., DeCarli, C.S. and Hwang, S.J., 2021. Multi-Scanner Harmonization of Paired Neuroimaging Data via Structure Preserving Embedding Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3284-3293). | [Paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/html/Torbati_Multi-Scanner_Harmonization_of_Paired_Neuroimaging_Data_via_Structure_Preserving_Embedding_ICCVW_2021_paper.html) [Poster](https://github.com/Mahbaneh/MISPEL/blob/main/iccv21_CVAMD_Paper15_Final.pdf) [Presentation](https://github.com/Mahbaneh/MISPEL/blob/main/MISPEL_Presentation.pptx)
# Table of content:
[Software requirements](#Software-requirements)\
[Structure of input data for MISPEL](#Structure-of-input-data-for-MISPEL)\
[Image Preprocessing](#Image-Preprocessing)\
[MISPEL Harmonization](#MISPEL-Harmonization)

# Software requirements:
Python and Keras. 

# Structure of input data for MISPEL:
MISPEL accepts a paired dataset as input data which should include images of "ALL" subjects across "ALL" scanners. We expect this data to be grouped in n (# of scanners) folders which named after the scanner names. Images of each subject should have identical names across these folders. Please refer to [Data folder](https://github.com/Mahbaneh/MISPEL/tree/main/Data) as an example of such data.

# Image preprocessing:
For the preprocessing step please read the 'preprocessing' paragraph in section 4.1. Multi-scanner Dataset of our paper. The steps are as follows:\
  Step 1. Registration to a template.\
  Step 2. N4 bias correction.\
  Step 3. Skull stripping.\
  Step 4. Image scaling.\
For the first three steps we used the instruction prepared in the [RAVEL repositoty](https://github.com/Jfortin1/RAVEL). You can run Step 4 by our code setting paparemer ... to True. 


# MISPEL harmonization:
