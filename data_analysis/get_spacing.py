"""

View data axial spacing distribution
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from tqdm import tqdm
import SimpleITK as sitk

import parameter as para


spacing_list = []

for file in tqdm(os.listdir(para.train_ct_path)):

    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    temp = ct.GetSpacing()[-1]

    print('-----------------')
    print(temp)

    spacing_list.append(temp)

print('mean:', sum(spacing_list) / len(spacing_list))

spacing_list.sort()
print(spacing_list)

# The average spacing in the training set is 1.59mm
# The spacing of the data in the test set is 1mm
