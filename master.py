
import torch, sys, argparse, cv2, os, time
from datetime import datetime
from self_utils.multi_tasks import Counting_Processing
from self_utils.overall_method import Object_Counter, Image_Capture
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort
import imutils
import torch.nn as nn
from packaging import version
import data

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

    # 获取视频源
    mycap = Image_Capture(yolo5_config.input)
    # 实例化计数器
    Obj_Counter = Object_Counter(class_names)
    #总帧数
    total_num = mycap.get_length()
    videowriter = None
    fps = int(mycap.get(5))
    if fps == 0:
        fps = 25
    t = int(1000 / fps)
    mkfile_time = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')

    # 鼠标点击回调函数
    def select_person(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(len(outs)):
                #print("x=",x)
                box = bboxes[i]
                trackid = track_ids[i]
                x1 =box[0]
                y1 =box[1]
                x2 =box[2]
                y2 =box[3]
                if (x1 < x) and (x < x2) and (y1 < y) and (y < y2):
                    sid = int(trackid)
                    #print("selected_id1", data.selected_id)
                    if sid not in data.selected_id:
                        data.selected_id.append(int(trackid))
                        #print("selected_id2",data.selected_id)

                        #print("id=", data.selected_id[i])
        if event == cv2.EVENT_RBUTTONDOWN:
            data.selected_id = []

    def out_man():
        if len((data.selected_id)):
            #print("selected_id删除前选中的ID", data.selected_id)
            #print("track_ids删除前所有ID",track_ids)
            i=0
            while i < len(data.selected_id):

                if track_ids is not None:
                    if data.selected_id[i] not in track_ids:
                        data.selected_id.remove(data.selected_id[i])


                else:
                    data.selected_id=[]
                    #print('---------------',data.selected_id)
                i+=1

        #print("selected_id删除后选中的ID",data.selected_id)
        #print("track_ids删除后所有ID", track_ids)


    cv2.namedWindow('video')
    while mycap.ifcontinue():
        ret, img,feiwu= mycap.read()
        if ret:

            # 处理每帧图片
            result_img, bboxes, track_ids, outs = Counting_Processing(img, yolo5_config, Model,
                          class_names, deepsort_tracker, Obj_Counter, data.selected_id)
            out_man()
            #print("id2=", data.selected_id)
            if type(result_img) == AttributeError:
                print("错误为{}".format(result_img))
                exit(1)


            # 保存视频
            if videowriter is None:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                videowriter = cv2.VideoWriter('./output/result' + mkfile_time + '.mp4', fourcc, fps,
                                              (result_img.shape[1], result_img.shape[0]))
            videowriter.write(result_img)
            result_img = imutils.resize(result_img, height=500)
            cv2.setMouseCallback('video', select_person)
            cv2.imshow('video', result_img)


            cv2.waitKey(t)


            if cv2.waitKey(100) & 0xff == ord('q'):
                break

        sys.stdout.write("\r=> processing at %d; total: %d" % (mycap.get_index(), total_num))
        sys.stdout.flush()

    videowriter.release()
    cv2.destroyAllWindows()
    mycap.release()

    print("\n=> process done {}/{} images, total cost: {:.2f}s [{:.2f} fps]"
          .format(len(os.listdir(yolo5_config.output)), total_num, time.time() - c,
                  len(os.listdir(yolo5_config.output)) / (time.time() - c)))

    print("=> main task finished: {}".format(datetime.now().strftime('%H:%M:%S')))


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
    print(yolo5_config)

    main(yolo5_config)

    print("结果保存在:", yolo5_config.output)