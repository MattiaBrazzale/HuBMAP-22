`Work in progress...`

# HuBMAP-22
This is my solution for the **HuBMAP + HPA - Hacking the Human Body** challenge that I joined in Kaggle, which can be reached through <a href="https://www.kaggle.com/competitions/hubmap-organ-segmentation" target="_blank">this link</a>. 
It was sponsored by Google, Genentech and the Indiana University.

### Objective
The **goal** of this competition is to develop an image segmentation model, that generalizes across five different organs and that is robust across different datasets, that helps to segment functional tissue units (FTUs), a task which is heavily time-consuming.
The main difficulties of this challenge are:
1. Data is frow two different consortia: the Human Protein Atlas (HPA) and Human BioMolecular Atlas Program (HuBMAP).
2. The training set is from a single source: HPA data.
3. The images were of different resolutions and of different pixel size.
4. The test set is completely hidden.


### Data
The train data, which is the only one available, consists of 351 images, with their respective annotations, of tissue area. Of the test set we have a single example and some basic information. Considering together train and test set, HPA and HuBMAP images ranges from 160x160 pixels to 4500x4500 pixels. 

All the data is available, by accepting the competition terms, at the link provided before.


## My Solution
### Data Preprocessing
The data loading and preprocessing step that I decided to use was done through a PyTorch dataset and dataloader. 
At first all the images are resized to mitigate the fact that different organ have different pixel size. Secondly, due to the low amount of images in the training set, I had to perform heavy augmentations to the images in order to ensure that the model generalizes well. The augmentations include:
* Rotate
* Flips
* Distortions
* Colour and Brightness
* etc.

Moreover, during the data loading process, each augmentations has a certain probability to be executed. In this way, we ensure that the model will train on sligthly different images each epoch.


### Model Architectures
At the beginning of this challenge I experimented with different architectures, mainly with Unet with different backbones. This was my choice in order to familiarize with one of the most common algorithm for this kind of tasks, since this was my first serious project related to image segmentation (or classification).

After this initial approach which led to some interesting results, but not sufficient to compete with SoTA algorithms, I moved to more recent and better performing architectures. 
The model which I used for my final submission, and with which I got the best score, was the Co-Scale Conv-Attentional Image Transformers (CoaT), which is presented, implemented and introduced in the following <a href="https://github.com/mlpc-ucsd/CoaT" target="_blank">github repository</a>. The architecture that I used is the CoaT-Lite Medium, with the pretrained weights on the ImageNet dataset. Despite the fact that I didn't have time to experiment with the model architecture, it still outperformed my UneT-based implementations.


### Training
These are the main settings that I used during the training of the model:
* As stated before, the images are resized to the maximum size possible without causing memory issues, i.e. 768x768. Therefore, due to these GPU memory constraints the batch size was set to 1, but the gradient is accumulated for 24 batches before doing an optimization step. 
* The loss used is a combination of the Jaccard and Dice loss.
* The metric used to evaluate the performances is the Dice Coefficient, i.e. the metric used to evaluate the challenge leaderboard.
* The starting learning rate was the same for the encoder and decoder and was scheduled using the OneCycleLR included in the PyTorch module. 

I have used two different approaches for the training of the model:
* In the first case I trained the model on the whole training set.
* In the second case the training was performed using a 5-fold split and creating an ensemble of 5 different models.


### Inference
To perform the inference on the test set I used a few tricks to boost the performances:
* For each one of the 5-fold splits, after the model was trained, I checked, for each organ, which was the threshold for the mask that gives the best performances of the model on the validation set. I then used an average of these values for the submissions.
* A further boost on the score was given by performing Test Time Augmentation (TTA): each image is modified with different augmentation, which are reversable, such as flips, rotations, re-scales or pixel-wise transformations. Then inference is performed on each one of the augmented image and then the mask is reverted. At the end the predictions were averaged and submitted.



### Placement
At the end of the challenge I placed: `126 out of 1175 teams`, just out of the medal zone...

Considering that my GPUs quota were heavily limited and the fact that I joined the competition late, I am quite satisfied with this result. 


### Possible Improvements
Possible improvements that I didn't have the time to try are:
* **Increase Size**: resize the image to a bigger size than 768x768.
* **Tiling**: instead of training the model on each image, each image is divided in *n* non-overlapping tiles, on which the model is trained. Then, for inference, the mask is predicted on each tile and then is combined into one.
* **Ensemble**: this is probably what would have improved more my score, since I was able only to test and ensemble of 3 out of 5 of my 5-fold split (ran out of GPU quota...). Moreover ensembling also different models (and even architectures) trained with different approaches, such as: with different image sizes, with tiling and without tiling, etc. would have surely improved the score even more.
* **External Data**: using external data for the training (such as from previous challenges or online) and pseudo-labeling them.



