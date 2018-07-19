from random import randint
import os.path as osp

def create_database(MODE, TIMESTEPS, base_dir='./dataset', dataset='fashion'):
    if MODE == 'train' and dataset == 'fashion':
        pose_dir = osp.join(base_dir, 'fashion/training_data_using_gt/target_posemap/')
        style_dir = osp.join(base_dir, 'fashion/training_data_using_gt/ref_img/')
        target_dir = osp.join(base_dir, 'fashion/training_data_using_gt/target_img/')

        person_num = 14370
        num_examples_per_epoch = 1000

    elif MODE == 'test' and dataset == 'fashion':
        pose_dir = osp.join(base_dir, 'fashion/testing_data_using_gt/target_posemap/')
        style_dir = osp.join(base_dir, 'fashion/testing_data_using_gt/ref_img/')
        target_dir = osp.join(base_dir, 'fashion/testing_data_using_gt/target_img/')

        person_num = 6138
        num_examples_per_epoch = 6138
    elif MODE=='test' and dataset == 'volleyball':
        pose_dir = osp.join(base_dir, 'volleyball/testing_data_using_gt/new_posemap_testing_256x256/')
        style_dir = osp.join(base_dir, 'volleyball/testing_data_using_gt/new_obj_crop_testing_256x256/')

        tracklet_length = 11
        person_num = 7145
        num_examples_per_epoch = 7145

        #pose_dir = '/local-scratch2/mzhai/cvpr18/dataset/data/training_data_using_gt/posemap_training_256x256/'
        #style_dir = '/local-scratch2/mzhai/cvpr18/dataset/data/training_data_using_gt/obj_crop_training_256x256/'

        #tracklet_length = 11
        #person_num = 16187
        #num_examples_per_epoch = 16187
    elif MODE=='train' and dataset == 'volleyball':
        pose_dir = osp.join(base_dir, 'volleyball/testing_data_using_gt/limited_posemap_testing_256x256/')
        style_dir = osp.join(base_dir, 'volleyball/testing_data_using_gt/limited_obj_crop_testing_256x256/')

        tracklet_length = 11
        person_num = 343
        num_examples_per_epoch = 343

    else:
        pose_dir = ''
        style_dir = ''

        tracklet_length = -1
        person_num = -1

    data = []

    if MODE == 'train' and dataset == 'fashion':
        for i in range(1, num_examples_per_epoch + 1):
            i_person = randint(1, person_num)
            # i_person = i

            style_image = style_dir + str(i_person) + '.jpg'
            pose_image = pose_dir + str(i_person) + '.jpg'
            target_image = target_dir + str(i_person) + '.jpg'

            data.append(style_image + ',' + pose_image + ',' + target_image)

        print(len(data))

    elif MODE == 'test' and dataset == 'fashion':
        for i in range(1, num_examples_per_epoch + 1):
            i_person = i

            style_image = style_dir + str(i_person) + '.jpg'
            pose_image = pose_dir + str(i_person) + '.jpg'
            target_image = target_dir + str(i_person) + '.jpg'

            data.append(style_image + ',' + pose_image + ',' + target_image)

        print(len(data))
    elif dataset == 'volleyball' and (MODE == 'test' or MODE == 'train'):
        for i in range(1, num_examples_per_epoch+1):
            if MODE == 'train':
                i_person = randint(1,person_num)
            else:
                i_person = i

            for i_frame in range(6, tracklet_length):
                style_image = style_dir + str(i_person) + '_' + str(0) + '.jpg'
                pose_image = pose_dir + str(i_person) + '_' + str(i_frame) + '.jpg'
                target_image = style_dir + str(i_person) + '_' + str(i_frame) + '.jpg'
                data.append(style_image + ',' + pose_image + ',' + target_image)
        print len(data)
    else:
        raise ValueError('Invalid MODE and dataset {} {}'.format(MODE, dataset))

    return data
