import glob
import tifffile
import numpy as np
import lightgbm as lgb 
import warnings
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

warnings.simplefilter('ignore')


# train model
train_path = './data/train/s2_image/'
mask_path = './data/train/mask'
trains = glob.glob(f'{train_path}/*')
masks = glob.glob(f'{mask_path}/*')
trains.sort()
masks.sort()

X, y, g = [], [], []
for i, (t, m) in enumerate(zip(trains, masks)):
    img = tifffile.imread(t).astype(float)
    mask = tifffile.imread(m).astype(float)
    X.append(img.reshape(-1, 12))
    y.append(mask.reshape(-1))
    g.append(np.ones_like(mask.reshape(-1)*i))

X = np.vstack(X)
y = np.hstack(y)
g = np.hstack(g)


lgb_params = {
    'boosting_type' : 'gbdt',
    'num_leaves' : 31,
    'max_depth' : -1,
    'n_estimators' : 300,
    'random_state' : 136
}



