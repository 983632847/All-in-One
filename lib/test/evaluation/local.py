from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    
    # Set your local paths here.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One'
    settings.network_path = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One/output' # Where tracking networks are stored.
    settings.results_path = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One/output/test/tracking_results' # Where to store tracking results
    settings.result_plot_path = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One/output/test/result_plots'
    settings.segmentation_path = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One/output/test/segmentation_results'
    settings.save_dir = '/home/dell/WorkSpace/Tracking_by_NL/All-in-One/output'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/Data/Datasets/LaSOT/LaSOTBenchmark'
    settings.lasotext_path = '/Data/Datasets/LaSOT/LaSOT_extension_subset'
    settings.tnl2k_path = "/Data/Datasets/TNL2K/test"
    settings.otb99lang_path = "/Data/Datasets/OTB99-LANG/test"
    settings.webuav3m_path = "/SATA/WebUAV-3M/Test"

    return settings


