# Lidar_Openpcdet_ST3D

## Use Openpcdet to train your own dataset
> Recently I've been working with my own dataset.
> I think there are still many people who don't know how to use Openpcdet, an advanced lidar opensource, to train their own datasets, including me lol.
> So I summarize a little bit what I did.

> No matter what kind of lidar you use (velodyne, hesai, livox etc.), just get the data of (x,y,z,intensity) and the rest of the conversion is easy.
> (x,y,z,intensity)->.txt, .pcd, bin, .npy etc.
- Build a dataloader and import your own data(✔)
- Training my own dataset（TO do）
- Test my own dataset（TO do）

# 1. Dataloader
## 1.1 Prepare your own lidar data

```
ST3D/Openpcdet
├── data
│   ├── mydata
│   │   │── lidar
│   │   │── label
```

``` │   │   │── lidar```

- lidar information
 
![Image text](https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/main/picture/lidar_file.png)


``` │   │   │── label```
- label information

![Image text](https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/main/picture/json_file.png)
![Image text](https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/main/picture/annotation_information.png)


## 1.2 Align the coordinates of your own dataset with openpcdet
            # My dataset coordinates are:
            # - x pointing to the right
            # - y pointing to the front
            # - z pointing up
            # Openpcdet Normative coordinates are:
            # - x pointing foreward
            # - y pointings to the left
            # - z pointing to the top
            # So a transformation is required to the match the normative coordinates
            points = points[:, [1, 0, 2]] # switch x and y
            points[:, 1] = - points[:, 1] # revert y axis
            
   ![Image text](https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/main/picture/switch_coor.png)     
    
## 1.3 Modify & create the three configuration files 
### 1.3.1 ~/pcdet/datasets/`__init__.py`
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/__init__.py#L9
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/__init__.py#L20

Feel free to replace mydata with your own dataset name.

### 1.3.2 ~/pcdet/datasets/mydata/`mydata_dataset.py`
- Configuration about your own dataset
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/mydata/mydata_dataset.py

- Get your Lidar path and annotation path
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/mydata/mydata_dataset.py#L88
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/mydata/mydata_dataset.py#L91

- Get all annotation labels informations，Regarding the annotation information, you need to change it to your own annotation information. There is only car in my json file. Since I haven't used it for training, I haven't tested this part yet.
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/pcdet/datasets/mydata/mydata_dataset.py#L140-L146


### 1.3.3 ~/tools/cfgs/dataset_configs/`mydata_dataset.yaml`

- Basically copy from kitti_dataset.yaml

https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/tools/cfgs/dataset_configs/mydata_dataset.yaml

- change here
https://github.com/Leozyc-waseda/Lidar_Openpcdet_ST3D/blob/365ac6b57906a01aeac27bfbdd8f1df29b5c2744/ST3D/tools/cfgs/dataset_configs/mydata_dataset.yaml#L1-L2

## 1.4 Generate the data infos by running the following command:
```
python -m pcdet.datasets.mydata.mydata_dataset create_mydata_infos tools/cfgs/dataset_configs/mydata_dataset.yaml
```

# 2. Training my own dataset（TO do）
# 3. Test my own dataset（TO do）

## License

MIT

