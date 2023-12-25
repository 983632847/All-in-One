class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/Data/Datasets/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/Data/Datasets/GOT-10k-full_data/train'
        self.trackingnet_dir = '/SATA/TrackingNet'
        self.coco_dir = '/Data/Datasets/COCO'
        self.refcoco_dir = '/Data/Datasets/COCO'
        self.visualgenome_dir = '/Data/Datasets/Visual_Genome'
        self.webuav3m_dir = '/SATA/WebUAV-3M/Train'
        self.tnl2k_dir = '/Data/Datasets/TNL2K/train'
        self.otb99lang_dir = '/Data/Datasets/OTB99-LANG/train'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
