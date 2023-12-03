import glob
import tifffile
import numpy as np
import lightgbm as lgb 
import warnings
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import os

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
    X.append(img.reshape(-1, 12)) # (x, y, 12) to (x*y, 12)
    y.append(mask.reshape(-1)) # (x, y) to (x*y)
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



# gkfold = GroupKFold(n_splits=4)
gkfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=136)

models = []

for i, (train_idx, valid_idx) in enumerate(gkfold.split(X, y, g)):
    train_x = X[train_idx]
    train_y = y[train_idx]

    val_x = X[valid_idx]
    val_y = y[valid_idx]

    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(train_x, train_y,
          eval_metric='logloss',
          eval_set=[(val_x, val_y)],
          early_stopping_rounds=10,
          verbose=1,
          callbacks=[lgb.log_evaluation(100)]
        )
    models.append(m)


test_path = './data/evaluation/'
test_mask_path = './data/sample/'


masks = glob.glob(f'{test_mask_path}/*')
tests = glob.glob(f'{test_path}/*')
masks.sort()
tests.sort()

if not os.path.isdir('output'):
    os.mkdir('output')

threshold = 0.5

for i, (m, t) in tqdm.tqdm(enumerate(zip(masks, tests))):
    basename = os.path.basename(m)
    output_file = f'output/{basename}'

    img = tifffile.imread(t).astype(np.float)
    mask = tifffile.imread(m).astype(np.float)

    X = img.reshape(-1, 12)
    shape_mask = mask.shape

    pred = 0
    for model in models:
        pred += model.predict_proba(X) 
    pred /= len(models)

    pred_mask = np.argmax(pred, axis=1).astype(np.uint8)
    pred_mask = pred_mask.reshape(shape_mask[0], shape_mask[1])

    tifffile.imwrite(output_file, pred_mask)
