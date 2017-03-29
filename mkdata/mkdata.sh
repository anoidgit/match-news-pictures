#!/bin/bash

#收集图片文件
python tools/ldpic.py News_train.pmap News_pic_train.hdf5 News_pic_info_train/

#过滤无法获取对应图片的文本
python tools/filtxtbypic.py News_train.pmap News_train.txt errf.txt News_full.txt

#生成均值，标准差矩阵
python tools/buildavg.py News_pic_train.hdf5 pavg.hdf5

#预处理图像数据
python tools/normp.py News_pic_train.hdf5 pavg.hdf5 full_pic.hdf5

#切分数据集
rm -fr pduse
mkdir pduse
#映射文本
python tools/mapfile.py News_full.txt tools/ref_weight/idf.txt pduse/fullt.txt
#切分
python tools/splitdataset.py full_pic.hdf5 pduse/fullt.txt pduse/trainp.hdf5 pduse/traint.txt pduse/devp.hdf5 pduse/devt.txt

#生成数据集
#查看词表大小，确定unk编号（259185=259184+1）
python tools/mapfile.py tools/ref_weight/idf.txt
#训练集:确定切分批量尺寸
python tools/buildspliter.py pduse/traint.txt pduse/train.splitter
#训练集:生成数据
python tools/buildata.py pduse/trainp.hdf5 duse/trainp.hdf5 pduse/traint.txt duse/traint.hdf5 pduse/train.splitter 259185
#测试集:确定切分批量尺寸
python tools/buildspliter.py pduse/devt.txt pduse/dev.splitter
#测试集:生成数据
python tools/buildata.py pduse/devp.hdf5 duse/devp.hdf5 pduse/devt.txt duse/devt.hdf5 pduse/dev.splitter 259185
