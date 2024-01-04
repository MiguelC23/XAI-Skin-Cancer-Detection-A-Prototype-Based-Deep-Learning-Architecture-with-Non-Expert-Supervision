# Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision
## Interpretable Skin Cancer Detection with Prototypes
Code Regarding the models associated with the section "Interpretable Skin Cancer Detection with Prototypes" of the Master's Thesis in Electrical and Computer Engineering titled 
"Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision," authored by Miguel Joaquim Nobre Correia conducted at the Instituto Superior Técnico, Lisbon, Portugal.

This code was used to train and evaluate the models identified in the results using the following loss functions $LP_$, $L_P+L_M$,$L_P+L_R$ and $L_{P-Masked}$.  
In this type of models, the decision is made solely based on the prototypes of the malignant class, namely melanoma (MEL), with the prototypes of the benign class, namely nevus (NV), not used in the classification process. 
In other words, only the similarities with the melanoma prototypes are provided as input to the final fully connected layer of the model, the classifier.

This code was designed solely for application to a binary classification problem of skin cancer lesions with two classes. It has been exclusively used to distinguish between melanoma and nevus (MEL vs NV), with MEL identified as class 0 and NV as class 1.
