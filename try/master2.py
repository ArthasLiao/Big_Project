# coding=utf8
from typing import List, Any, Union
from self_utils.inference import img_preprocessing
import torch, sys, argparse, cv2, os, time
from datetime import datetime
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter, Image_Capture
from self_utils.post_processing import deepsort_update
from self_utils.inference import yolov5_prediction
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort
import imutils
import torch.nn as nn
from packaging import version


# 比较两个包的版本号,因为pytorch1.10.0以上版本会导致代码出现小bug,所以必须判断不同版本的pytorch以增强代码的健壮性
def compare_versions(version1, version2):
    v1 = version.parse(version1)
    v2 = version.parse(version2)
    if v1 > v2:
        return 1
    elif v1 < v2:
        return -1
    else:
        return 0
def main(yolo5_config):
    # 初始化模型等
    print("=> main task started: {}".format(datetime.now().strftime('%H:%M:%S')))
    a = time.time()
    # 添加全局变量
    class_names = []
    # 加载模型
    if yolo5_config.device != "cpu":
        Model = \
            torch.load(yolo5_config.weights, map_location=lambda storage, loc: storage.cuda(int(yolo5_config.device)))[
                'model'].float().fuse().eval()
    else:
        Model = torch.load(yolo5_config.weights, map_location=torch.device('cpu'))['model'].float().fuse().eval()

    # 版本判断
    best_torch_version = "1.10.0"
    current_torch_version = str(torch.__version__).split('+')[0]
    result = compare_versions(best_torch_version, current_torch_version)
    if result == -1:
        for m in Model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    # 获取类别名
    classnames = Model.module.names if hasattr(Model, 'module') else Model.names
    class_names.append(classnames[0])

    b = time.time()
    print("==> class names: ", class_names)
    print("=> load model, cost:{:.2f}s".format(b - a))

    os.makedirs(yolo5_config.output, exist_ok=True)
    c = time.time()

    # 初始化追踪器
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort_tracker = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST,
                                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE,
                                n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True, use_appearence=True)
    # 视频读取
    cap = Image_Capture(yolo5_config.input)
    # 实例化计数器
    Obj_Counter = Object_Counter(class_names)
    # 总帧数
    total_num = cap.get_length()
    fps = int(cap.get(5))
    if fps == 0:
        fps = 25
    t = int(1000 / fps)
    mkfile_time = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    tracked_targets = []


    # 鼠标点击回调函数
    def mouse_callback(event, x, y,flags, params):
        global tracked_targets
        if event == cv2.EVENT_LBUTTONDOWN:
            for target in tracked_targets:
                bbox = target['bbox']
                if x >=bbox[0] and x<=bbox[2] and y>=bbox[1] and y<=bbox[3]:
                    target['selected'] = not target['selected']
                    break
    def draw_bboxes(frame,bboxes,targets):
        colors = [(0,255,0) if target['selected'] else (0,0,255) for target in targets]
        for bbox,color in zip(bboxes,colors):
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        return frame

    cap = cv2.VideoCapture(0)
    while True:
        # 初始化摄像头
        ret, frame = cap.read()
        if ret:
            tracked_bboxes = Counting_Processing(frame, yolo5_config, Model, class_names, deepsort_tracker, Obj_Counter)


            tracked_targets = []
            for bbox in tracked_bboxes:
                tracked_targets.append({'bbox':bbox,'selected':False})
            frame = draw_bboxes(frame,tracked_bboxes,tracked_targets)
            new_tracked_targets = []
            for target in tracked_targets:
                bbox = target['bbox']
                if bbox[0] >= 0 and bbox[1]>=0 and bbox[2] <= frame.shape[1] and bbox[3] <= frame.shape[0]:
                    new_tracked_targets.append(target)
            tracked_targets = new_tracked_targets
            cv2.imshow('Tracking',frame)
            if cv2.waitKey(1) & 0xFF == ('q'):
                break
            '''
            tensor_img = img_preprocessing(frame, yolo5_config.device, yolo5_config.img_size)
            pred = yolov5_prediction(Model, tensor_img, yolo5_config.conf_thres, yolo5_config.iou_thres,
                                     yolo5_config.classes)
            outputs = deepsort_update(deepsort_tracker, pred, tensor_img, frame)
            box = []
            trackid = []
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, 5]
                for i in range(len(outputs)):
                    box = bbox_xyxy[i]
                    trackid = identities[i]
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            poz=[c1,c2]
            results = Counting_Processing(frame, yolo5_config, Model, class_names, deepsort_tracker, Obj_Counter)
            cv2.setMouseCallback('video', select_person)
            def draw_bboxes(frame,bboxes,targets):
                colors = [(0,255,0) if target[sel]]



            if type(results) == AttributeError:
                print("错误为{}".format(results))
                exit(1)
            if videowriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videowriter = cv2.VideoWriter(
                    './output/result' + mkfile_time + '.mp4', fourcc, fps, (results.shape[1], results.shape[0]))
            videowriter.write(results)
            results = imutils.resize(results, height=500)
            cv2.imshow('video', results)
            cv2.waitKey(t)

        if cv2.waitKey(100) & 0xff == ord('q'):
            # 点x退出
            break'''

    sys.stdout.write("\r=> processing at %d; total: %d" % (cap.get_index(), total_num))
    sys.stdout.flush()

    sys.stdout.write("\r=> processing at %d; total: %d" % (cap.get_index(), total_num))
    sys.stdout.flush()

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    # 参数解析
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
    main(yolo5_config)