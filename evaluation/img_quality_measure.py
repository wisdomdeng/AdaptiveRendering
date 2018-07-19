import cv2
import os
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim
import sys

def our_mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    tmp = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

    # diff = err - tmp
    # print diff,

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compute_error(style_dir, generated_dir):
    # per pixel mse error along sequence
    person_num = 6138

    mse_overall = 0
    psnr_overall = 0
    ssim_overall = 0

    total_num_people = 0

    for i_person in range(1, person_num + 1):
        total_num_people += 1

        target_image_name = style_dir + str(i_person) + '.jpg'
        generated_image_name = generated_dir + 'test_' + str(i_person) + '.png'

        target_image = cv2.imread(target_image_name)
        target_image = cv2.resize(target_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        target_image = np.array(target_image)
        target_image = target_image / 127.5 - 1

        generate_image = cv2.imread(generated_image_name)
        # generate_image=cv2.resize(generate_image,(256,256),interpolation=cv2.INTER_LINEAR)
        generate_image = np.array(generate_image)
        generate_image = generate_image / 127.5 - 1

        mse_overall += mse(target_image, generate_image)
        psnr_overall += psnr(target_image, generate_image)
        ssim_overall += ssim(target_image, generate_image, multichannel=True)

    mse_overall = mse_overall / total_num_people
    psnr_overall = psnr_overall / total_num_people
    ssim_overall = ssim_overall / total_num_people

    print('\n')
    print 'MSE', 'PSNR', 'SSIM'
    print mse_overall, psnr_overall, ssim_overall


if __name__ == "__main__":
    # style_dir = '/local-scratch/dengr/ruizhid_cedar/local-scratch2/mzhai/cvpr18/dataset/fashion/testing_data_using_gt/target_img/'

    # generated_dir = './test-result/'

    # style_dir = sys.argv[1]
    # generated_dir = sys.argv[2]
    style_dir = sys.argv[1]
    generated_dir = sys.argv[2]

    compute_error(style_dir, generated_dir)

