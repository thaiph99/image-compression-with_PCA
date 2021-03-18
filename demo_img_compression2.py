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

    img_origin = cv2.imread('thaiph.jpg', 1)

    # split matrix image
    img0 = img_origin[:, :, 0]
    img1 = img_origin[:, :, 1]
    img2 = img_origin[:, :, 2]
    print(img0.shape)
    # testjson
    img_dict = {'img0': img0, 'img1': img1, 'img2': img2}

    json_obj1 = json.dumps(img_dict, cls=NpEncoder)

    with open('img_origin.json', 'w') as f:
        json.dump(json_obj1, f)

    def compress_img(k, img):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = ipca.fit_transform(img)

        return img_compressed, ipca.__dict__

    def extract_img(k, result_compressed):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = result_compressed[0]
        dict_att = result_compressed[1]

        for key in dict_att.keys():
            ipca.__setattr__(key, dict_att[key])

        img_extracted = ipca.inverse_transform(img_compressed)
        return img_extracted

    # def compress_extract_img(k, img):
    #     ipca = IncrementalPCA(n_components=k)
    #     img_compressed = ipca.fit_transform(img)
    #     # print(img.shape)
    #     # print(img_compressed.shape)
    #     print('components : ', ipca.components_.shape)
    #     print('variance : ', ipca.explained_variance_.shape)
    #     print('variance ratio : ', ipca.explained_variance_ratio_.shape)
    #     img_extracted = ipca.inverse_transform(img_compressed)
    #     # print(img_extracted)
    #     return img_extracted

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
    # img_compressed = [compress_img(n_components, img_tmp) for img_tmp in img_s]
    img_compressed = {}

    for i in range(len(img_s)):
        img_com = compress_img(n_components, img_s[i])
        img_compressed['img'+str(i)] = img_com

    print(type(img_compressed))

    print('--------------------------------------------------')
    tmp = img_compressed['img0'][1]
    print(tmp.keys())
    print(tmp['components_'].shape)
    print(tmp['mean_'].shape)
    print(tmp['explained_variance_'].shape)
    print(tmp['whiten'])
    print('--------------------------------------------------')

    print(type(img_compressed))
    json_obj = json.dumps(img_compressed, cls=NpEncoder)
    with open('img_compressed.json', 'w') as f:
        json.dump(json_obj, f)

    # with open('data.json', 'w') as f:

    img_extracted = [extract_img(n_components, img_tmp)
                     for img_tmp in img_compressed.values()]
    img_extracted = concat_img(*img_extracted)
    print(img_extracted.shape)

    cv2.imwrite('img_extracted.jpg', img_extracted)
    print('Done')


if __name__ == '__main__':
    main()
