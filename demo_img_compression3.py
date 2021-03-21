import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def main():

    path1 = 'datajpg/'
    path2 = 'datapng/'
    f1 = '.jpg'
    f2 = '.png'

    path, f = path2, f2
    print(path+'imgg_origin'+f)
    img_origin = cv2.imread(path+'imgg_origin'+f, 1)

    # split matrix image
    img0 = img_origin[:, :, 0]
    img1 = img_origin[:, :, 1]
    img2 = img_origin[:, :, 2]
    print(img0.shape)

    with open(path+'imgj_origin.json', 'w') as f:
        json.dump(img_origin, f, cls=NpEncoder)

    def compress_img(k, img):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = ipca.fit_transform(img)
        list_att = ['components_', 'mean_', 'explained_variance_', 'whiten']
        ipca_att = {}
        for att in list_att:
            if type(np.array([])) == type(ipca.__getattribute__(att)):
                att_values = np.round(ipca.__getattribute__(att), 4)
            else:
                att_values = ipca.__getattribute__(att)
            # ipca_att[att] = ipca.__getattribute__(att)
            ipca_att[att] = att_values

        return img_compressed, ipca_att

    def extract_img(k, result_compressed):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = result_compressed[0]
        dict_att = result_compressed[1]
        for key in dict_att.keys():
            ipca.__setattr__(key, dict_att[key])

        img_extracted = ipca.inverse_transform(img_compressed)
        return img_extracted

    def concat_img(img0, img1, img2):
        img_beu = np.ones((img0.shape[0], img0.shape[1], 3), 'int')
        img_beu[:, :, 0] = img0
        img_beu[:, :, 1] = img1
        img_beu[:, :, 2] = img2
        return img_beu

    def show_img(img):
        plt.figure(figsize=[20, 20])
        plt.imshow(img)
        plt.show()

    img_s = [img0, img1, img2]
    n_components = 100
    img_compressed = [compress_img(n_components, img_tmp) for img_tmp in img_s]
    # img_compressed = {}
    # for i in range(len(img_s)):
    #     img_com = compress_img(n_components, img_s[i])
    #     img_compressed['img'+str(i)] = img_com
    # print(type(img_compressed))

    print('--------------------------------------------------')
    tmp = img_compressed[0][1]
    print(img_compressed[0][0].shape)
    print(tmp['components_'].shape)
    print(tmp['mean_'].shape)
    print(tmp['explained_variance_'].shape)
    print(tmp['whiten'])
    print('--------------------------------------------------')

    # write json file
    with open(path+'imgj_compressed.json', 'w') as f:
        json.dump(img_compressed, f, cls=NpEncoder)

    # read json file
    img_compressed1 = {}
    with open(path+'imgj_compressed.json', 'r') as f:
        img_compressed1 = json.load(f)

    # extract image
    img_extracted = [extract_img(n_components, img_tmp)
                     for img_tmp in img_compressed1]
    img_extracted = concat_img(*img_extracted)
    print(img_extracted.shape)

    # export image
    fname = path+'imgg_extracted.png'
    cv2.imwrite(fname, img_extracted)
    print('Done')


if __name__ == '__main__':
    main()
