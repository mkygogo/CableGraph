这个项目是用在电科院端子排插孔判断用的,数据制作过程如下：
先标注obb数据，用X-Anylabling工具就可以。

obb识别就是标准的yolo那套。具体看yolo_obb里的说明。


obb框连线数据标注，基于之前带obb框的数据，标注线条，或者多线条。
标注完的label文件夹里的json文件包含了obb框跟line，用cable_detect中process_labels.py，
把其中的line的标注数据分离到lable_line文件夹，这样就可以后面训练连线模型训练。 