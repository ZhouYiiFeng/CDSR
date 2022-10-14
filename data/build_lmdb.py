import glob
import os
import pickle

import imageio
import lmdb
from tqdm import tqdm

if __name__ == '__main__':

    dataroot_GT = "your/path/to/images/DF2klmdb/HR/"

    target_img_file1 = "your/path/to/Flickr2K/Flickr2K_HR"
    target_img_file2 = "your/path/to/DIV2K_train_HR"

    target_imgs_path = glob.glob(os.path.join(dataroot_GT, target_img_file1, "*.png")) + glob.glob(
        os.path.join(dataroot_GT, target_img_file2, "*.png"))

    if not os.path.exists(dataroot_GT):
        os.makedirs(dataroot_GT)
    env = lmdb.open(os.path.join(dataroot_GT, 'df2k_imgs_train'), map_size=int(1099511627776))
    dict_data = {}
    keys = []
    reslutions = []
    for ii, img_path in tqdm(enumerate(target_imgs_path)):
        pch_gt = imageio.imread(img_path)
        h, w, c = pch_gt.shape
        reslutions.append([h, w, c])
        with env.begin(write=True) as txn:
            # pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
            pch_gt = pch_gt.tobytes()

            key_ = str(ii)
            keys.append(key_)
            txn.put(key_.encode('ascii'), pch_gt)

    dict_data["keys"] = keys
    dict_data["res"] = reslutions
    with open(os.path.join(dataroot_GT, "meta_info.pkl"), 'wb') as fo:
        pickle.dump(dict_data, fo)
