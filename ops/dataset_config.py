# Code for paper:
# [Title]  - "PAN: Towards Fast Action Recognition via Learning Persistence of Appearance"
# [Author] - Can Zhang, Yuexian Zou, Guang Chen, Lei Gan
# [Github] - https://github.com/zhang-can/PAN-PyTorch

import os

ROOT_DATASET = '/data/zhangcan/dataset/'


def return_ucf101(modality):
    filename_categories = 101
    if modality in ['RGB', 'PA', 'Lite']:
        root_data = ROOT_DATASET + 'ucf101_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/ucf101/split1/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/ucf101/split1/val.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality in ['RGB', 'PA', 'Lite']:
        root_data = ROOT_DATASET + 'hmdb51_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/hmdb51/split1/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/hmdb51/split1/val.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 174
    if modality in ['RGB', 'PA', 'Lite']:
        root_data = ROOT_DATASET + 'sthv1_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/sthv1/split/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/sthv1/split/val.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 174
    if modality in ['RGB', 'PA', 'Lite']:
        root_data = ROOT_DATASET + 'sthv2_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/sthv2/split/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/sthv2/split/val.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 27
    if modality in ['RGB', 'PA', 'Lite']:
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/jester/split/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/jester/split/val.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality in ['RGB', 'PA', 'Lite']:
        root_data = ROOT_DATASET + 'kinetics400_frames'
        filename_imglist_train = '/data/zhangcan/file_lists/kin400/split/train.txt'
        filename_imglist_val = '/data/zhangcan/file_lists/kin400/split/val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
