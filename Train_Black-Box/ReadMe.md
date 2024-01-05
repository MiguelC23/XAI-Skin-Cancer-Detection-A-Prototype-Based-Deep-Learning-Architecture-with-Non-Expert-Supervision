## 1-How to train the Black-Box model?

To train the model, simply run the **Train.py** file. It is not necessary to provide arguments in the code line, as the model settings are edited in the **settings.py** file. After that, just run **Train.py**.
Regarding the **settings.py** file, carefully examine the file as it is commented, explaining each of the parameters of the model settings. However, I would like to emphasize here that it is important to pay attention to the paths for the folders where the data is located.

  train_dir = r"..." # Path to the training set; in our case, it refers to the training images from ISIC 2019.
  
  test_dir = r"..." Path to the validation set; in our case, it refers to the validation images from ISIC 2019.

Whether it was the training or validation folders of the images, they always contained the folders corresponding to the 8 classes AK, BCC, BKL, DF, MEL, NV, SCC, and VASC. Subsequently, depending on the user's choice in the settings.py file, with num_classes = 2 or num_classes = 8, the code will selectively retrieve only the folders for MEL and NV or all of them, respectively.
