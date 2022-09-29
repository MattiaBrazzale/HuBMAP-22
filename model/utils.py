"""
This script collects all the useful functions and classes used for the challenge.
"""


''' Imports '''
import random
import pandas as pd
from sklearn.model_selection import KFold




def seed_everything(seed):    
    """
    Function used to seed everything, to make the results reproducible
    """
  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
    
def make_fold(num_fold=5, val_fold=0, df):
    """
    Function used to seed everything, to make the results reproducible.
    
    Args: 
      num_fold: number of folds in which the DataFrame is split
      val_fold: which of the folds is used as a validation set
      df: pandas DataFrame that needs to be split
      
    Returns:
      train_df: the training DataFrame
      valid_df: the validation DataFrame
    """
    
    num_fold = 5
    skf = KFold(n_splits=num_fold, shuffle=True,random_state=42)

    df.loc[:,'fold']=-1
    for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
        df.iloc[v_idx,-1]=f

    train_df=df[df.fold!=fold].reset_index(drop=True)
    valid_df=df[df.fold==fold].reset_index(drop=True)
    return train_df, valid_df
      
      

def get_mask(image_id, df):
    """
    Function used obtain the mask related to the given image.
    
    Args: 
      image_id: id of the image of which we want the mask
      df: pandas DataFrame that contains the images ids
      
    Returns:
      mask: the corresponding mask
    """
    
    row = df.loc[df['id'] == image_id].squeeze()
    h, w = row[['img_height', 'img_width']]
    mask = np.zeros(shape=[h * w], dtype=np.uint8)
    s = row['rle'].split()
    starts, lengths = [ np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]) ]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo : hi] = 1
        
    mask = mask.reshape([h, w]).T
    mask = np.expand_dims(mask, axis=2)
        
    return mask
  
  

def rle_encode(img):
    """ 
    Function used to save the predicted mask in string format
    
    Args:
        img: numpy array where 1 indicates mask and 0 indicates background
    
    Returns: 
        run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
