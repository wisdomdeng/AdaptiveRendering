import json
import os
import os.path as osp
import numpy as np

gt_dir = '/local-scratch/cjc/pose_parse/pose_score/raw_pose/fashion/GT'
test_base_dir = '/local-scratch/cjc/pose_parse/pose_score/raw_pose/fashion'
test_list = ['ours', 'ours-no-style-loss', 'ours-no-l1-loss', 'ours-no-content-loss', 'baseline2', 'baseline3', 'vam-new', 'vam']

def get_pose_score(gt_dir, test_dir):
    total_loss = 0
    succ = 0
    fail = 0
    invalid = 0
    for gt_name in sorted(os.listdir(gt_dir)):
        # gt_split = gt_name.split('_')
        # if gt_split[1] in ['0', '1', '2', '3', '4', '5']:
            # continue
        gt_path = os.path.join(gt_dir, gt_name) 
        test_path = os.path.join(test_dir, 'test_' + gt_name)
        gt_pose = get_single_pose(gt_path)
        test_pose = get_single_pose(test_path)
        status, loss = pose_mse(gt_pose, test_pose)
        if status == 'succ':
            total_loss += loss
            succ += 1
        elif status == 'fail':
            fail += 1
        elif status == 'invalid':
            invalid += 1
        else:
            raise ValueError('invalid return value {}'.format(status))
    total_loss /= succ
    return succ, fail, invalid, total_loss


def get_single_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        all_people = data['people']
    best_conf = 0
    best_pose = None
    best_joint_num = 0

    for idx, person in enumerate(all_people):
        single_pose = _separate_pose_list(person['pose_keypoints'])
        total_conf = 0
        cnt = 0
        for conf in single_pose[:,2]:
            if conf > 0:
                total_conf += conf
                cnt += 1
        total_conf /= cnt
        if  cnt > best_joint_num + 3 and total_conf > 0.3:
            best_conf = total_conf
            best_pose = single_pose    
            best_joint_num = cnt
        elif total_conf > best_conf and cnt > best_joint_num:
            best_conf = total_conf
            best_pose = single_pose    
            best_joint_num = cnt

    return best_pose


def _separate_pose_list(lst):
    '''
    Convert a list with length of 18 * 3 into a ndarray of shape (18, 3)
    :param lst:
    :return:
    '''
    i = 0
    pose_list = []
    for i in range(0, len(lst), 3):
        pose_list.append(lst[i:i + 3])
    return np.array(pose_list)


def pose_mse_recall(gt, output):
    if gt is None:
        return 'invalid', 0

    valid_gt_joint = 0
    for idx, joint in enumerate(gt):
        if joint[2] > 0:
            valid_gt_joint += 1

    if valid_gt_joint < 8:
        return 'invalid', 0

    # failure case        
    if output is None:
        return 'fail', 0
        
    total_mse = 0
    hit = 0    

    for idx, joint in enumerate(gt):
        output_joint = output[idx]
        if joint[2] > 0:
            t = ( (joint[0] / 2  - output_joint[0])  **2 + (joint[1] / 2 - output_joint[1]) **2 ) / 2
            if t <= 100:
                hit += 1
                total_mse += t
    
    # matched, return matching status and mse
    if hit > 0.8 * valid_gt_joint:
        total_mse /= hit
        return 'succ', total_mse 
    # not matched. failure case
    else:
        return 'fail', 0

def pose_mse(gt, output):
    """
        The simple version of MSE between two poses, where invalid/failed pose is considered to be all (0, 0)
    """
    total_mse = 0
    cnt = 0
    for idx, joint in enumerate(gt):
        if output is not None:
            output_joint = output[idx]
        else:
            output_joint = (0, 0)

        if joint[2] > 0:
            t = ( ( (joint[0] - 127.5)/127.5  - (output_joint[0]-63.5) / 63.5)  **2 + ((joint[1] - 127.5) / 127.5 - (output_joint[1]-63.5) / 63.5) **2 ) / 2  # normalize to [-1, 1] and then compute distance
            total_mse += t
            cnt += 1

    total_mse /= cnt

    return 'succ', total_mse 


if __name__ == '__main__':
    for dirname in test_list:
        test_dir = osp.join(test_base_dir, dirname)
        print('testing poses in dir {}'.format(dirname))
        succ, fail, invalid, score = get_pose_score(gt_dir, test_dir)
        print('succ num: ', succ)
        print('fail num: ', fail)
        print('invalid num: ', invalid)
        print('score: ', score)
        print('recall {}'.format(succ * 1.0 / (succ + fail)))
