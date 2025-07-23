from sbg_inference import SBGGraspDetector

det = SBGGraspDetector(
    checkpoint_path="/home/nikita/diplom/Scale-Balanced-Grasp/logs/log_full_model/checkpoint.tar",
    onnx_seg        ="/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/best.onnx",
    seg_yaml        ="/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/merged_yolo_dataset2.yaml")