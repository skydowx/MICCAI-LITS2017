# -----------------------Path related parameters---------------------------------------

train_ct_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/train/CT/'  # CT data path of the original training set

train_seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/train/seg/'  # Original training set labeled data path

test_ct_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/CT/'  # CT data path of the original test set

test_seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'  # Original test set labeled data path

training_set_path = './train/'  # Data storage address used to train the network

pred_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/liver_pred'  # Save path of network prediction results

crf_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/crf'  # CRF optimization result save path

module_path = './module/net550-0.028-0.022.pth'  # Test model address

# -----------------------Path related parameters--------------------------


# ---------------------Training data to obtain relevant parameters---------

size = 48  # Use 48 consecutive slices as input to the network

down_scale = 0.5  # Cross-sectional downsampling factor

expand_slice = 20  # Only use the liver and the upper and lower 20 slices of the liver as training samples

slice_thickness = 1  # Normalize the spacing of all data on the z-axis to 1mm

upper, lower = 200, -200  # CT data gray-scale truncation window

# ---------------------Training data to obtain relevant parameters---------

# -----------------------Network structure related parameters-------------

drop_rate = 0.3  # dropout random drop probability

# ---------------------Parameters relevant to training---------------------

# ---------------------Network training related parameters----------------

gpu = '0'  # Graphic card identifier

Epoch = 1000

learning_rate = 1e-4

learning_rate_decay = [500, 750]

alpha = 0.33  # Depth supervision attenuation coefficient

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------Network training related parameters----------------

# ----------------------Model test related parameters----------------------

threshold = 0.5  # Threshold degree threshold

stride = 12  # Sliding sampling step

maximum_hole = 5e4  # Largest void area

# ----------------------Model test related parameters----------------------

# ---------------------CRF post-processing optimization related parameters-

z_expand, x_expand, y_expand = 10, 30, 30  # The number of expansions in three directions based on the predicted results

max_iter = 20  # Number of CRF iterations

s1, s2, s3 = 1, 10, 10  # CRF Gaussian kernel parameters

# ---------------------CRF post-processing optimization related parameters-