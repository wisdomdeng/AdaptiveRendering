import os
import cv2

# Fashion Dataset

dataset_path = '/local-scratch/cjc/pose_parse/pose_score/fashion'
target_base = '/local-scratch/cjc/pose_parse/pose_score/raw_pose/fashion'

for image_dir in sorted(os.listdir(dataset_path)):
    image_dir_path = os.path.join(dataset_path, image_dir)
    
    if not os.path.isdir(image_dir_path):
        continue

    resolution_str = None
    for filename in sorted(os.listdir(image_dir_path)):
        file_path = os.path.join(image_dir_path, filename)
        image = cv2.imread(file_path)
        resolution_str = '{}x{}'.format(image.shape[1], image.shape[0])
        break
    print('resolution: ', resolution_str)
    target_dir_path = os.path.join(target_base, image_dir)
    print(target_dir_path)
    if os.path.exists(target_dir_path):
        print('{} already exists, skip it'.format(target_dir_path))
        continue
    p = os.popen('{}/get_pose.sh {} {} {}'.format(os.getcwd(), image_dir_path, resolution_str, target_dir_path), 'r')
    while True:
        line = p.readline()
        if not line: break
        print(line)

# Volleyball

dataset_path = '/cs/vml3/mzhai/BMVC/volleyball'
target_base = '/local-scratch/cjc/pose_parse/pose_score/raw_pose/volleyball'

for image_dir in sorted(os.listdir(dataset_path)):
    image_dir_path = os.path.join(dataset_path, image_dir)
    
    if not os.path.isdir(image_dir_path):
        continue

    resolution_str = None
    for filename in sorted(os.listdir(image_dir_path)):
        file_path = os.path.join(image_dir_path, filename)
        image = cv2.imread(file_path)
        resolution_str = '{}x{}'.format(image.shape[1], image.shape[0])
        break
    print('resolution: ', resolution_str)
    target_dir_path = os.path.join(target_base, image_dir)
    print(target_dir_path)
    if os.path.exists(target_dir_path):
        print('{} already exists, skip it'.format(target_dir_path))
        continue
    p = os.popen('{}/get_pose.sh {} {} {}'.format(os.getcwd(), image_dir_path, resolution_str, target_dir_path), 'r')
    while True:
        line = p.readline()
        if not line: break
        print(line)
