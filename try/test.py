# 2. 去重
'''unique_ids = []
            for id_array in track_ids:
                for unique_id in id_array:
                    if unique_id not in unique_ids:
                        unique_ids.append(unique_id)'''

'''from self_utils.overall_method import Object_Counter, Image_Capture
import torch, sys, argparse, cv2, os, time
def main(yolo5_config):
    mycap = Image_Capture(yolo5_config.input)
    #ret, img = mycap.read()
    print(mycap.read())


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="0", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="./output",
                        help='folder to save result imgs, can not use input folder')
    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 1 2 3')

    yolo5_config = parser.parse_args()
    main(yolo5_config)'''
[1,2,3]