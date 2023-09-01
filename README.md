## 基于yolov5和deepsort的行人跟踪计数系统
# 成员介绍
组长:董佳骐（U202112217）
组员:张书健（U202112246）
     彭凌刚（U202112465）
     廖书志（U202112462）
     
# 项目分工
              董佳骐   张书健   彭凌刚   廖书志
程序编写         1        1       1        1
资料查找         1        1       1        1
代码修改         1                1        1 
PPT及报告制作             1       
答辩             1

# 项目介绍
    此次项目完成离不开小组全体成员的共同努力奉献，前期我们一起学习了git与github的使用，并于计划答辩前各自在网上收集前人资料，并尝试运行，但遗憾的是没能跑起来。而计划答辩以后，我们调整策略，集中突破一个前人项目，这一过程让我们深刻地理解环境的搭建之重要，并从中学习到安包的各种知识，包的适配性即版本既要满足编译器版本又要满足本身的版本需求。经过一个晚上的努力，我们搭建了虚拟环境，跑起来了这个项目https://github.com/MichistaLin/yolov5-deepsort-pedestraintracking.git。完成了人体识别与计数的功能。
     我们在此基础上接着完成了跟踪的功能，鼠标左键点击被选中的人的框会变为红色，其余人为绿色。我们制作取消按钮为鼠标右键，右键GUI视频任意一处即可取消对红框目标追踪，相较于创建可视按钮更符合用户使用习惯；实现了被跟踪人离开画面自动取消跟踪的功能，即再回到画面时为绿色框。
     在基础功能实现的基础上我们做了多目标跟踪，即可标记多个人进行跟踪。
       

项目运行环境：win10，pycharm，python3.6+

主要需要的包：pytorch >= 1.7.0，opencv

运行master.py即可开始追踪检测，可以在控制台运行

```python
python master.py --input="你的视频路径"
```

也可以在pycharm中直接右键运行（把--input中的defalt改为你要检测的视频路径即可），这样执行的都是默认参数

输入的参数：

```python
parser = argparse.ArgumentParser()
# 视频的路径，默认是本项目中的一个测试视频test.mp4，可自行更改
parser.add_argument('--input', type=str, default="./test.mp4",
                        help='test imgs folder or video or camera')  # 输入'0'表示调用电脑默认摄像头
# 处理后视频的输出路径
parser.add_argument('--output', type=str, default="./output",
                        help='folder to save result imgs, can not use input folder')
parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path(s)')
parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
# GPU（0表示设备的默认的显卡）或CPU
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# 通过classes来过滤检测类别
parser.add_argument('--classes', default=0, type=int, help='filter by class: --class 0, or --class 0 2 3')  

```



检测效果：

![](https://img-blog.csdnimg.cn/965128beb6804047980329b7c4911275.jpeg#pic_center)
