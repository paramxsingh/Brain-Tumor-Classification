# Brain-Tumor-Classification# **Import needed modules**
import os
import pandas as pd
from fastai.vision.all import *
from fastai.vision import models
from fastai.metrics import error_rate, accuracy

import warnings
warnings.filterwarnings("ignore")
set_seed(42)

print ('modules loaded')
# **Data Preprocessing**
#### **Read data and store it in dataframe**
# Generate data paths with labels
data_dir = '../input/brats-2019-traintestvalid/dataset/train'
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)
df
dls = ImageDataLoaders.from_df(df,
                                fn_col=0, # filepaths
                                label_col=1, # labels
                                valid_pct=0.2,
                                folder='', 
                                item_tfms=Resize(224))
dls.show_batch(max_n=16)
# **Model Structure**
learn = vision_learner(dls, 'efficientnet_b3', metrics=[accuracy, error_rate], path='.').to_fp16()
learn.summary()
learn.lr_find(suggest_funcs=(valley, slide))
## **Training**
learn.fit_one_cycle(20)
learn.show_results()
# Save the model
learn.save('/kaggle/working/Model')
# Build a Classification Interpretation object from our learn model
# it can show us where the model made the worse predictions:
interp = ClassificationInterpretation.from_learner(learn)
# Plot the top ‘n’ classes where the classifier has least precision.
interp.plot_top_losses(15, figsize=(12,12))
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
#To view the list of classes most misclassified as a list
interp.most_confused(min_val=2) #We are ignoring single image misclassification

#Sorted descending list of largest non-diagonal entries of confusion matrix, 
#presented as actual, predicted, number of occurrences.
## Thank You..
If you find this notebook is good enough, please upvote it..!
