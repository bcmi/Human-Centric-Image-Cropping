import os

class Config:
    data_root = '/workspace/aesthetic_cropping/human_centric/dataset/'
    # the pre-defined candidate crops are downloaded from:
    # https://github.com/luwr1022/listwise-view-ranking/blob/master/pdefined_anchors.pkl
    predefined_pkl = os.path.join(data_root, 'pdefined_anchors.pkl')
    CPC_path  = os.path.join(data_root, 'CPCDataset')
    CPC_image = os.path.join(CPC_path,  'images')
    CPC_anno  = os.path.join(CPC_path,  'image_crop.json')
    CPC_human = os.path.join(CPC_path,  'human_bboxes.json')

    FCDB_path = os.path.join(data_root, 'FCDB')
    FCDB_image= os.path.join(FCDB_path, 'data')
    FCDB_anno = os.path.join(FCDB_path, 'image_crop.json')
    FCDB_human= os.path.join(FCDB_path, 'human_bboxes.json')
    FCDB_split= os.path.join(FCDB_path, 'data_split.json')

    FLMS_path = os.path.join(data_root, 'FLMS')
    FLMS_image= os.path.join(FLMS_path, 'image')
    FLMS_anno = os.path.join(FLMS_path, 'image_crop.json')
    FLMS_human= os.path.join(FLMS_path, 'human_bboxes.json')

    GAIC_path = os.path.join(data_root, 'GAICD')
    GAIC_image= os.path.join(GAIC_path, 'images')
    GAIC_anno = os.path.join(GAIC_path, 'image_crop.json')
    GAIC_human= os.path.join(GAIC_path, 'human_bboxes.json')
    GAIC_split= os.path.join(GAIC_path, 'original_data_split.json')
    GAIC_human_split = os.path.join(GAIC_path, 'human_data_split.json')

    heat_map_dir = '/workspace/aesthetic_cropping/human_centric/code/heat_map/heat_map_gt'
    CPC_heat_map = os.path.join(heat_map_dir, 'CPC', 'mask')
    GAIC_heat_map= os.path.join(heat_map_dir, 'GAICD', 'mask')

    heat_map_size = 1./4
    image_size = (256,256)
    backbone = 'vgg16'
    backbone_weight_path = ('/workspace/pretrained_models/{}.pth'.format(backbone))
    # training
    training_set = 'CPC' # ['GAICD', 'CPC']
    loss_type = ['L1Loss','RankLoss']
    gpu_id = 0
    num_workers = 8
    only_human = False
    data_augmentation = True
    keep_aspect_ratio = False

    use_partition_aware = True
    partition_aware_type = 9  # [0,1,2,9]
    visualize_partition_feature = False

    use_content_preserve  = True
    only_content_preserve = False
    content_preserve_type = 'gcn'
    content_loss_weight   = 1.
    visualize_heat_map    = False

    use_rod_feature = True
    reduced_dim = 32

    batch_size  = 8
    lr_scheduler= 'cosine'
    max_epoch = 30
    lr_decay_epoch = [max_epoch+1]
    eval_freq = 1

    view_per_image = 16
    lr = 1e-4
    lr_decay = 0.1
    weight_decay = 1e-4

    save_freq = max_epoch+1
    if only_human:
        display_freq = 10
    else:
        display_freq = 100
    visualize_freq = 100

    prefix = training_set
    if only_human:
        prefix += '-Human'
    if not data_augmentation:
        prefix += '_wodataaug'
    if reduced_dim != 32:
        prefix += f'_{reduced_dim}redim'
    if loss_type != ['L1Loss','RankLoss']:
        if isinstance(loss_type, list) and len(loss_type) > 1:
            for i in range(len(loss_type)):
                if loss_type[i] == 'L1Loss':
                    continue
                prefix += f'_{loss_type[i]}'
    if use_partition_aware:
        prefix += ('_PA')
        if partition_aware_type != 9:
            prefix += f'-{partition_aware_type}part'
    if use_content_preserve:
        prefix += ('_CP')
        if content_preserve_type != 'gcn':
            prefix += f'-{content_preserve_type}'
        if only_content_preserve:
            prefix += f'_onlycontent'
    exp_root = os.path.join(os.getcwd(), './experiments')

    # prefix = f'{content_loss_weight}_content_loss'
    # exp_root = os.path.join(os.getcwd(), './experiments/hyper_parameter')

    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)
    # print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

cfg = Config()

if __name__ == '__main__':
    cfg = Config()