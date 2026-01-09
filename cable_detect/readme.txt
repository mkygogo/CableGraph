

数据格式说明：
    images：640*640的图片
    label:是图片中obb数据
    label_line:链接obb框的线条数据
其中的images跟label是外面X-Anylabling处理好的，label里的json文件本身包含了obb跟line的数据，直接拷贝过来，
process_lables.py处理完后label中的json只有obb，label_line中json只有line

分割训练数据, 80训练20测试：python preprocess_dataset.py

开始训练：python cable_train.py

测试训练的模型：python cable_inference.py 注意运行前，看看代码里的图片名，确认是不是要推理用的,访问的是test文件夹下对应的文件。

新增数据的处理方法：
将标注好的数据label文件夹放到temp文件夹，, label中的json文件包含obb跟line标签。
运行：precess_lables.py。生成label_line文件夹包含分离出来的line标注json文件，
然后把images， label， label_line中的文件分别放到cable_detect下面的几个文件夹，新增数据就完成了。
