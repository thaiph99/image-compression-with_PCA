import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from matplotlib.image import imread


def main():
    # read image by opencv
    img_origin = imread('thaiph.jpg')
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
        print('components : ', ipca.n_components_)
        print(ipca.components_.shape)
        print(ipca.explained_variance_.shape)
        print(ipca.explained_variance_ratio_.shape)
        return img_compressed, ipca

    def extract_img(k, img_com):
        ipca = IncrementalPCA(n_components=k)
        img_extracted = ipca.inverse_transform(img_com)
        # print(img_extracted)
        return img_extracted

    def compress_extract_img(k, img):
        ipca = IncrementalPCA(n_components=k)
        img_compressed = ipca.fit_transform(img)
        # print(img.shape)
        # print(img_compressed.shape)
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
    img_test = [extract_img(100, img_tmp) for img_tmp in img_s]
    # img_test = [compress_extract_img(100, img_tmp) for img_tmp in img_s]
    img_test = concat_img(*img_test)
    print(img_test.shape)
    print(img_origin.shape)
    cv2.imwrite('test1.jpg', img_test)


if __name__ == '__main__':
    main()
