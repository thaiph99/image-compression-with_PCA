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


def stand_dict(dict_tmp):
    for key in dict_tmp.keys():
        if type(dict_tmp[key]) == type(np.array([])):
            print(key, dict_tmp[key])
            dict_tmp[key] = dict_tmp[key].tolist()
    return dict_tmp


def main():
    # read image by opencv
    # img_origin = imread('thaiph.jpg')
    img_origin = cv2.imread('thaiph.jpg', 1)
    # print(img_origin)

    # split matrix image
    img0 = img_origin[:, :, 0]
    img1 = img_origin[:, :, 1]
    img2 = img_origin[:, :, 2]

    img_s = [img0, img1, img2]
    # pre

    def compress_img(k, img):
        # img_bit = img/255
        ipca = IncrementalPCA(n_components=k)
        img_compressed = ipca.fit_transform(img)
        # print(ipca.__dict__)
        # list_att = ipca.__dir__()
        # list_att.remove('_repr_html_')
        # list_values = [ipca.__getattribute__(att) for att in list_att]
        # print('attribute : ', list_values.shape)
        return img_compressed, ipca.__dict__

    def extract_img(k, com_set):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = com_set[0]
        dict_att = com_set[1]

        for key in dict_att.keys():
            ipca.__setattr__(key, dict_att[key])

        img_extracted = ipca.inverse_transform(img_compressed)
        return img_extracted

    def compress_extract_img(k, img):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = ipca.fit_transform(img)
        # print(img.shape)
        # print(img_compressed.shape)
        print('components : ', ipca.components_.shape)
        print('variance : ', ipca.explained_variance_.shape)
        print('variance ratio : ', ipca.explained_variance_ratio_.shape)
        img_extracted = ipca.inverse_transform(img_compressed)
        # print(img_extracted)
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

    img_s = [compress_img(100, img_tmp) for img_tmp in img_s]

    '''
    dict_keys(['n_components', 'whiten', 'copy', 'batch_size', 'components_', 
    'n_samples_seen_', 'mean_', 'var_', 'singular_values_', 'explained_variance_', 
    'explained_variance_ratio_', 'noise_variance_', 'n_features_in_', 'batch_size_', 
    'n_components_'])
    '''

    print(img_s[0][1]['components_'].shape)
    print(img_s[0][1]['explained_variance_'].shape)
    print(img_s[0][1]['explained_variance_ratio_'].shape)
    print('-------')
    
    # json_obj = json.dumps(img_s[0][1], cls=NpEncoder)
    # print(json_obj)
    # with open('data.json', 'w') as f:
    #     json.dump(json_obj, f)

    img_test = [extract_img(100, img_tmp) for img_tmp in img_s]
    # img_test = [compress_extract_img(200, img_tmp) for img_tmp in img_s]
    img_test = concat_img(*img_test)
    print(img_test.shape)

    cv2.imwrite('test1.jpg', img_test)
    # show_img(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
    # test = compress_img(200, img0)
    # print(type(test))


if __name__ == '__main__':
    main()
