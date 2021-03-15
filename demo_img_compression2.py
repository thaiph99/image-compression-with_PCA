import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


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
        print('components : ', ipca.n_components_)
        print(ipca.components_.shape)
        print(ipca.explained_variance_.shape)
        print(ipca.explained_variance_ratio_.shape)
        return ipca.components_

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

    # img_s = [compress_img(100, img_tmp) for img_tmp in img_s]
    # img_test = [extract_img(100, img_tmp) for img_tmp in img_s]
    img_test = [compress_extract_img(200, img_tmp) for img_tmp in img_s]
    img_test = concat_img(*img_test)
    # cv2.imshow('test1', img_test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(img_test.shape)
    cv2.imwrite('test1.jpg', img_test)

    test = compress_img(200, img0)
    print(type(test))


if __name__ == '__main__':
    main()
