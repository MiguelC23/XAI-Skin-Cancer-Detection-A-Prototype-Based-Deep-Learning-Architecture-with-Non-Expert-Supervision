# Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision
## Interpretable Skin Cancer Detection with Prototypes
Code Regarding the models associated with the section "Interpretable Skin Cancer Detection with Prototypes" of the Master's Thesis in Electrical and Computer Engineering titled 
"Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision," authored by Miguel Joaquim Nobre Correia conducted at the Instituto Superior Técnico, Lisbon, Portugal.

This code was used to train and evaluate the models identified in the results using the following loss functions $L_{\text{P}}$, $L_{\text{P}}+L_{\text{M}}$, $L_{\text{P}}+L_{\text{R}}$ and $L_{\text{P-Masked}}$.  

In this type of models, we have prototypes for all classes, not just prototypes of the malignant class. These models can be applied to both a binary problem, Melanoma vs Nevus, and a multiclass problem where we have 8 classes of skin lesions: AK, BCC, BKL, DF, MEL, NV, SCC, VASC.

The labels in the binary problem are 0 (MEL) and 1 (NV), while in the problem with the 8 classes of skin lesions, the labels are 0 (AK), 1 (BCC), 2 (BKL), 3 (DF), 4 (MEL), 5 (NV), 6 (SCC), and 7 (VASC).

Here is the meaning of each acronym for the classes:  
Melanoma (MEL)  
Melanocytic nevus (NV)  
Basal cell carcinoma (BCC)  
Actinic keratosis (AK) 
Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis also called lichenoid keratosis) (BKL)  
Dermatofibroma (DF)  
Vascular lesion (VASC)  
Squamous cell carcinoma (SCC)

## 1-How to train the model?

To train the model, simply run the main.py file. It is not necessary to provide arguments in the code line, as the model settings are edited in the settings.py file. After that, just run main.py.
Regarding the settings.py file, carefully examine the file as it is commented, explaining each of the parameters of the model settings. However, I would like to emphasize here that it is important to pay attention to the paths for the folders where the data is located.
