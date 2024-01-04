Miguel Joaquim Nobre Correia.

Master's in Electrical and Computer Engineering.

Instituto Superior Técnico, Lisboa, Portugal.

Universidade de Lisboa.

In this GitHub repository, you can find the code used in the master's thesis project titled "Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision," 
authored by Miguel Joaquim Nobre Correia and supervised by Prof. Carlos Jorge Andrade Mariz Santiago and Prof. Ana Catarina Fidalgo Barata.

[See Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision](https://drive.google.com/file/d/1eKjxa3VhYLG_qV73ySK-5XtMqZHzTxLW/view?usp=sharing)

The code in this repository is an adaptation of the code in the following repositories:

|                                                    | Repository                                 | License                                                                                                         |
|----------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| ProtoPNet                                          | https://github.com/cfchen-duke/ProtoPNet   | [See License](https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/LICENSE)   |  
| IAIA-BL loss                                       | https://github.com/alinajadebarnett/iaiabl | [See License](https://github.com/alinajadebarnett/iaiabl/blob/04efedb3f6bd0b4495e90b4d4bfcbeacfde0db57/LICENSE) |
| Concept-level Debugging of Part-Prototype Networks | https://github.com/abonte/protopdebug      | [See License](https://github.com/abonte/protopdebug/blob/main/LICENSE)                                          |

## Used datasets [See sections 4.1, 4.2 and 5.3 of the thesis Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision](https://drive.google.com/file/d/1eKjxa3VhYLG_qV73ySK-5XtMqZHzTxLW/view?usp=sharing) 
**[ISIC 2019](https://challenge.isic-archive.com/data/#2019)- Images for Training and Validation** 

Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)

Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)", 2017; arXiv:1710.05006.

Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: "BCN20000: Dermoscopic Lesions in the Wild", 2019; arXiv:1908.02288.

**[ISIC 2018](https://challenge.isic-archive.com/data/#2018)- Part of the Masks for images present in ISIC 2019**

Some training and validation image masks were obtained from the ISIC 2018 dataset because there are images present in ISIC 2019 that are also found in ISIC 2018. All remaining masks were obtained from an automated skin lesion segmentation network. Please refer to sections 4.1 and 4.2 of the thesis document 'Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision' for further details.

Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018)

**[$\text{PH}^2$](https://www.fc.up.pt/addi/ph2%20database.html)-Used as a Test Dataset**

T. Mendonça, P. M. Ferreira, J. S. Marques, A. R. S. Marcal and J. Rozeira, "PH2 - A dermoscopic image database for research and benchmarking," 2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Osaka, Japan, 2013, pp. 5437-5440, doi: 10.1109/EMBC.2013.6610779.

**[Derm7pt](https://derm.cs.sfu.ca/Welcome.html)-Used as a Test Dataset**

J. Kawahara, S. Daneshvar, G. Argenziano and G. Hamarneh, "Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets," in IEEE Journal of Biomedical and Health Informatics, vol. 23, no. 2, pp. 538-546, March 2019, doi: 10.1109/JBHI.2018.2824327.

**[EASY Dermoscopy Expert Agreement Study Dataset (EDEASD)](https://api.isic-archive.com/collections/166/?page=1)-Used for concept analysis**

Please refer to sections 5.3 of the thesis document 'Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision' for further details.




