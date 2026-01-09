用X-Anylabling标注数据，标注前用resize_to_640.py转换图片为640*640

把X-Anlylabling的output dir设置到images同级目录:label

如果之前标注过，有训练好的模型，就先用pre_label_obb.py，做一次推理，生成预标注的label

标注完后，chack_labels.py检查一下标注有没有错误

最好通过X-Anylabling工具转换成Dota数据，会在label目录生成labelTxt文件夹。

用preprocess_dota_dataset.py 把数据划分成训练集跟测试集

用dota_to_yolo.py 把划分好的数据转换成yolo能用的数据格式，生成的dataset.yaml内容要做调整，参考dataset.yaml_backup

最后用train.py训练obb模型。



