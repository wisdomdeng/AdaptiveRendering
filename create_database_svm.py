from random import randint


def create_database_svm(MODE, TIMESTEPS):
    if MODE == 'test':
        pose_dir = '/local-scratch/dengr/ruizhid_cedar/local-scratch2/mzhai/cvpr18/dataset/fashion/testing_data_using_gt/target_posemap/'
        style_dir = '/local-scratch/dengr/ruizhid_cedar/local-scratch2/mzhai/cvpr18/dataset/fashion/testing_data_using_gt/ref_img/'
        target_dir = '/local-scratch/dengr/ruizhid_cedar/local-scratch2/mzhai/cvpr18/dataset/fashion/testing_data_using_gt/target_img/'

        person_num = 6138
        num_examples_per_epoch = 6138
    else:
        pose_dir = ''
        style_dir = ''

        tracklet_length = -1
        person_num = -1

    data = []

    for i in range(1, num_examples_per_epoch + 1):
        i_person = i

        style_image = style_dir + str(i_person) + '.jpg'
        pose_image = pose_dir + str(i_person) + '.jpg'
        target_image = target_dir + str(i_person) + '.jpg'

        data.append(style_image + ',' + pose_image + ',' + target_image)

    print len(data)
    return data
