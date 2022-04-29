import numpy as np
import copy
import pickle
import os
import json
import numpy as np
import pcl
import pandas
import sys
import random
from pypcd import pypcd
from skimage import io
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils,  common_utils
from pathlib import Path 


class mydataDataset(DatasetTemplate):

    def __init__(self,dataset_cfg,class_names,training= True, root_path=None,logger = None):

        super().__init__(
            dataset_cfg = dataset_cfg,class_names=class_names, 
            training = training, root_path = root_path,logger = logger
        )
        self.mydata_infos =[]
        # file list path
        self.files_list_pcd = []
        self.files_list_label = []
        self.files_list_label_train = []
        self.files_list_label_val = []
        self.files_list_pcd_train = []
        self.files_list_pcd_val = []
        self.train_ratio_of_all_labels=0.8

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.include_mydata_data(self.mode)


    def include_mydata_data(self,mode):
        if self.logger is not None:
            self.logger.info('Loading mydata dataset')
        
        mydata_infos =[]
        '''
        INFO_PATH:{
            'train':[mydata_infos_train.pkl],
            'test':[mydata_infos_val.pkl],}
        '''
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            
            info_path = str(self.root_path)+'/'+ info_path
            #info_path = self.root_path/ info_path
            if not Path(info_path).exists():
                continue

            with open(info_path,'rb') as f:
                infos = pickle.load(f)
                mydata_infos.extend(infos)

        self.mydata_infos.extend(mydata_infos)

        if self.logger is not None:
            self.logger.info('Total samples for mydata dataset: %d'%(len(mydata_infos)))

    #Get folder list
    def get_folder_list(self,root_path):
        folder_list = []
        root_path =root_path
        folder_list = os.listdir(root_path)
        return folder_list

    #return files_list_pcd and files_list_label 
    def get_files_name_list(self):
        folder_list = []
        folder_list = self.get_folder_list(self.root_path) # folder_list : ['label', 'lidar']

        files_list_pcd = []
        files_list_label = []

        for per_folder in folder_list:

            one_road_path = self.root_path   #one_road_path /home/algo-4/work/ST3D/data/mydata/label
            for one_folder in folder_list:
                if one_folder == 'lidar':
                    pcd_path = one_road_path / one_folder #/home/algo-4/work/ST3D/data/mydata/lidar

                if one_folder == 'label':
                    label_path = one_road_path / one_folder # #/home/algo-4/work/ST3D/data/mydata/label

            #get all lidar files
            pcd_files = self.get_folder_list(pcd_path)
            for thisfile in pcd_files:
                if thisfile.endswith(".pcd"):
                    files_list_pcd.append(str(pcd_path  / thisfile))

            #get all label files
            label_files = self.get_folder_list(label_path)
            for thisfile in label_files:
                if thisfile.endswith(".json"):
                    files_list_label.append(str(label_path / thisfile))

        return files_list_pcd,files_list_label

    # from label path to get pcd path
    def from_label_path_to_pcd_path(self,single_label_path):
        single_pcd_path = ''
        strl1 = 'label'
        strl2 = '.json'
        if strl1 in single_label_path:
            single_pcd_path = single_label_path.replace(strl1,'lidar')
        if strl2 in single_pcd_path:
            single_pcd_path = single_pcd_path.replace(strl2,'.pcd')
        #label to pcd path：single_pcd_path
        return single_pcd_path

    
    # get all annotation labels informations
    def get_all_labels(self,num_workers = 4,files_list_label=None):
        import concurrent.futures as futures

        global i 
        i =0
        def get_single_label_info(single_label_path):
            global i
            i=i+1
            single_label_path = single_label_path
            #Open my annotation files
            #labels [{'annotator': 'a', 'obj_id': '10', 'obj_type': 'Car', 'psr': {'position': {'x': 3.8293005220139094, 'y': 4.686827434320677, 'z': -1.1914207637310028}, 'rotation': {'x': 0, 'y': 0, 'z': -3.0106929596902186}, 'scale': {'x': 4.449836446223111, 'y': 1.6537831961454343, 'z': 1.4916328191757202}}}]

            with open(single_label_path,encoding = 'utf-8') as f:
                labels = json.load(f)

            #store information of objects from label
            single_objects_label_info = {}
            single_objects_label_info['single_label_path'] = single_label_path
            single_objects_label_info['single_pcd_path'] = self.from_label_path_to_pcd_path(single_label_path)
            single_objects_label_info['name'] = np.array([label['obj_type'] for label in labels]) #  ['Car']
            single_objects_label_info['box_center'] = np.array([[label['psr']['scale']['x'], label['psr']['scale']['y'],label['psr']['scale']['z']]  for  label in labels]) #single_objects_label_info['box_center'] [[4.44983645 1.6537832  1.49163282]]

            #print("single_objects_label_info['box_center']",single_objects_label_info['box_center'])
            single_objects_label_info['box_size'] = np.array([[label['psr']['position']['x'],label['psr']['position']['y'],label['psr']['position']['z']] for label in labels])
            single_objects_label_info['box_rotation'] = np.array([[label['psr']['rotation']['x'],label['psr']['rotation']['y'],label['psr']['rotation']['z']]  for label in labels])
            single_objects_label_info['tracker_id'] = np.array([ label['obj_id'] for label in labels])
            
            box_center = single_objects_label_info['box_center']
            box_size = single_objects_label_info['box_size']
            box_rotation = single_objects_label_info['box_rotation']

            rotation_yaw = box_rotation[:,2].reshape(-1,1)
            gt_boxes = np.concatenate([box_center,box_size,rotation_yaw],axis=1).astype(np.float32)
            single_objects_label_info['gt_boxes'] = gt_boxes

            print("The current processing progress is %d / %d "%(i,len(files_list_label)))
            return single_objects_label_info

        files_list_label = files_list_label
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(get_single_label_info,files_list_label)
        infos = list(infos)
        print("*****************************Done!***********************")
        print("type  of  infos :",type(infos))
        print("len  of  infos :",len(infos))
    
        return infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.mydata_infos) * self.total_epochs

        return len(self.mydata_infos)

    #remove some nan information
    def remove_nan_data(self,data_numpy):
        data_numpy = data_numpy
        data_pandas = pandas.DataFrame(data_numpy)
        data_pandas = data_pandas.dropna(axis=0,how='any')
        data_numpy = np.array(data_pandas)

        return data_numpy

    # return point cloud （M,4）
    def get_single_pcd_info(self,single_pcd_path):
        single_pcd_path = single_pcd_path
        
        single_pcd_points = pypcd.PointCloud.from_path(single_pcd_path) #/home/algo-4/work/ST3D/data/mydata/lidar/0202.pcd
       
        #pcd point cloud to np.array        
        single_pcd_points=single_pcd_points.pc_data.copy()
        single_pcd_points_np=np.array([list(single_pcd_points) for single_pcd_points in single_pcd_points])

        single_pcd_points_np = self.remove_nan_data(single_pcd_points_np)

        return single_pcd_points_np

    # remove single_objects_label_info ‘unknown’ information
    def drop_info_with_name(self,info,name):
        ret_info = {}
        info = info 
        keep_indices =[ i for i,x in enumerate(info['name']) if x != name]
        for key in info.keys():
            if key == 'single_label_path' or key == 'single_pcd_path':
                ret_info[key] = info[key]
                continue
            ret_info[key] = info[key][keep_indices]

        return ret_info
    
    #get pcd path list
    def from_labels_path_list_to_pcd_path_list(self,labels_path_list):
        pcd_path_list = []
        for m in labels_path_list:
            pcd_path_list.append(self.from_label_path_to_pcd_path(m))
        return pcd_path_list


    def list_subtraction(self,list_minute,list_minus):
        list_difference = []
        for m in list_minute:
            if m not in list_minus:
                list_difference.append(m)
        return list_difference

    def __getitem__(self,index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.mydata_infos)

        single_objects_label_info = copy.deepcopy(self.mydata_infos[index])
        single_label_path = single_objects_label_info['single_label_path']
        single_pcd_path = self.from_label_path_to_pcd_path(single_label_path)

        points = self.get_single_pcd_info(single_pcd_path)

        input_dict = {
            'points': points,
            'frame_id': single_pcd_path,
            'single_pcd_path':single_pcd_path,
        }

        single_objects_label_info = self.drop_info_with_name(info=single_objects_label_info,name='unknown')
        name =single_objects_label_info['name']             #(N,)
        box_center = single_objects_label_info['box_center']          #(N,3)
        box_size = single_objects_label_info['box_size']                    #(N,3)
        box_rotation  = single_objects_label_info['box_rotation']  #(N,3)
        tracker_id = single_objects_label_info['tracker_id']               #(N,)

        
        #data dataformat  (N, 7) [x, y, z, l, h, w, r]
        # gt_boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center"""
        rotation_yaw = box_rotation[:,2].reshape(-1,1)
        gt_boxes = np.concatenate([box_center,box_size,rotation_yaw],axis=1).astype(np.float32)
        #print(gt_boxes.shape)
        #print(type(gt_boxes))

        input_dict.update({
                'gt_names':name,
                'gt_boxes':gt_boxes,
                'tracker_id':tracker_id
        })
        #print(input_dict)
        
        # send data to self.prepare_data()
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def from_filepath_get_filename(self,filepath):
        filename = ''
        filepath = filepath

        filepath_and_filename = os.path.split(filepath)
        filename = filepath_and_filename[1]

        filename_and_extension = os.path.splitext(filename)
        filename = filename_and_extension[0]
        return filename


    def create_groundtruth_database(self,info_path = None,used_classes =None,split = 'train'):
        import torch
        database_save_path = Path(self.root_path)/('gt_database' if split =='train' else ('gt_database_%s'%split))
        db_info_save_path = Path(self.root_path)/('mydata_dbinfos_%s.pkl'%split)

        database_save_path.mkdir(parents=True,exist_ok=True)
        all_db_infos = {}

        with open(info_path,'rb') as f:
            infos = pickle.load(f)
        
        for k in range(len(infos)):
            print('gt_database sample:%d/%d'%(k+1,len(infos)))
            info = infos[k]
            info = self.drop_info_with_name(info=info,name='unknown')


            single_label_path = info['single_label_path']
            single_pcd_path = info['single_pcd_path']
            points = self.get_single_pcd_info(single_pcd_path)

            single_filename = self.from_filepath_get_filename(single_label_path)

            name = info['name']
            box_center = info['box_center']
            box_size = info['box_size']
            box_rotation = info['box_rotation']
            tracker_id = info['tracker_id']
            gt_boxes = info['gt_boxes']
            #num_obj is the number of valid objects
            num_obj = len(name)

            #Processing of parameters: first convert to tensor format (M, 3) (N, 7)
            ##Return a tensor of "all zeros" (a cuda function is run later, so the value may change),
            # The dimension is (N,M), N is the number of valid objects, M is the number of point clouds, after converting to numpy
            #point_indices means the indices of the point
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:,0:3]),torch.from_numpy(gt_boxes)
            ).numpy()   # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin'%(single_filename,name[i],i)
                filepath = database_save_path / filename

                #point_indices[i] > 0 gets a true and false index such as [T,F,T,T,F...], a total of M
                # Then take the point cloud data corresponding to true from points and put it in gt_points
                gt_points = points[point_indices[i]>0]

                #The first three columns of data for each of #gt_points
                # Subtract the position information of the first three columns of the current object in gt_boxes
                gt_points[:, :3] -= gt_boxes[i, :3]

                #Write the information of gt_points to the file
                
                with open(filepath,'w') as f:
                    gt_points.tofile(f)
                
                
                if (used_classes is None) or name[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))   # gt_database/xxxxx.bin
                    #Get the information of the current object
                    db_info = {
                        'name':name[i],'path':db_path,'image_idx':single_filename,
                        'gt_idx':i,'box3d_lidar':gt_boxes[i],'num_points_in_gt':gt_points.shape[0],
                        'box_center':box_center,'box_size':box_size,'box_rotation':box_rotation,'tracker_id':tracker_id
                    }

                    if name[i] in all_db_infos:
                        all_db_infos[name[i]].append(db_info)
                    else:
                        all_db_infos[name[i]] = [db_info]
        for k,v in all_db_infos.items():
            print('Database %s: %d'%(k,len(v)))
        
        with open(db_info_save_path,'wb') as f:
            pickle.dump(all_db_infos,f)
        
    #Receive model predictions in self.generate_prediction_dicts()
    # The 3D detection frame represented in the unified coordinate system can be converted back to the desired format.
    @staticmethod
    def generate_prediction_dicts(batch_dict,pred_dicts,class_names,output_path = None):

        '''
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.
        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:
        '''

        #Get the predicted template dictionary ret_dict, all defined as all-zero vectors
        #Parameter num_samples is the number of objects in this frame
        def get_template_prediction(num_samples):
            ret_dict = {
                'name':np.zeros(num_samples),
                'box_center':np.zeros([num_samples,3]),
                'box_size':np.zeros([num_samples,3]),
                'box_rotation':np.zeros([num_samples,3]),
                'tracker_id':np.zeros(num_samples),
                'scores':np.zeros(num_samples),
                'pred_labels':np.zeros(num_samples),
                'pred_lidar':np.zeros([num_samples,7])
            }
            
            return ret_dict

        def generate_single_sample_dict(box_dict):

            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            #Define an empty dictionary of frames to store information from predictions
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            
            pred_dict['name'] = np.array(class_names)[pred_labels -1]
            pred_dict['scores'] = pred_scores
            pred_dict['pred_labels'] = pred_labels
            pred_dict['pred_lidar'] = pred_boxes

            pred_dict['box_center'] = pred_boxes[:,0:3]
            pred_dict['box_size'] = pred_boxes[:,3:6]
            pred_dict['box_rotation'][:,-1] = pred_boxes[:,6]

            return pred_dict
        
        #Get the name of the file from the full path of the file (remove redundant information)
        def from_filepath_get_filename2(filepath):
            filename = ''
            filepath = filepath
            filepath_and_filename = os.path.split(filepath)
            filename = filepath_and_filename[1]
            filename_and_extension = os.path.splitext(filename)
            filename = filename_and_extension[0]
            return filename
        
        annos = []
        for index,box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            

            frame_id = batch_dict['frame_id'][index]
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)


            if output_path is not None:
                filename = from_filepath_get_filename2(frame_id)
                cur_det_file = Path(output_path)/('%s.txt'%filename)
                with open(cur_det_file,'w') as f:
                    name =single_pred_dict['name']
                    box_center = single_pred_dict['box_center']
                    box_size = single_pred_dict['box_size']
                    box_rotation = single_pred_dict['box_rotation']

                    for idx in range(len(single_pred_dict['name'])):
                        print('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,'
                        %(name[idx],
                        box_center[idx][0],box_center[idx][1],box_center[idx][2],
                        box_size[idx][0],box_size[idx][1],box_size[idx][2],
                        box_rotation[idx][0],box_rotation[idx][1],box_rotation[idx][1]),
                        file=f)
        
        return annos

    def evaluation(self,det_annos,class_names,**kwargs):
        if 'name' not in self.mydata_infos[0].keys():
            return None,{}


        from ..kitti.kitti_object_eval_python import eval3 as kitti_eval
        
        #Copy the parameter det_annos
        #copy.deepcopy() has the same effect on the nesting of tuples and lists, both of which are deeply copied (recursive)
        The content of #eval_det_info is the result predicted from the model, which is equal to det_annos
        eval_det_info = copy.deepcopy(det_annos)
        '''
        print('---------------------------eval_det_info--------------------------------------')
        print(eval_det_info[0].keys())
        print(type(eval_det_info))
        print(len(eval_det_info))
        '''
        

        # An info represents the information of a frame of data, then the following is to take out the annos attribute of all data and copy it
        #Essentially still equal to: eval_gt_infos = self.mydata_infos
        The content of #eval_gt_infos is actually the real collection information of val,
        eval_gt_infos = [copy.deepcopy(info) for info in self.mydata_infos]
        
        '''
        print('---------------------------eval_gt_infos--------------------------------------')
        print(eval_gt_infos[0].keys())
        print(type(eval_gt_infos))
        print(len(eval_gt_infos))
        print(class_names)
        '''
        

        #Call the function to predict the value of ap
        #ap_result_str,ap_dict = kitti_eval.get_coco_eval_result1(eval_gt_infos,eval_det_info,class_names)
        ap_result_str,ap_dict = kitti_eval.get_official_eval_result(eval_gt_infos,eval_det_info,class_names)

        return ap_result_str,ap_dict 

def create_mydata_infos(dataset_cfg,class_names,data_path,save_path,workers=4):
    dataset = mydataDataset(dataset_cfg=dataset_cfg,class_names=class_names,root_path=data_path,training=False)
    train_split,val_split = 'train','val'
    #Set the proportion of training set
    TRAIN_RATIO_OF_ALL_LABELS = dataset.train_ratio_of_all_labels

    #Define the path and name of the file to save
    train_filename = save_path /('mydata_infos_%s.pkl'%train_split)
    val_filename = save_path /('mydata_infos_%s.pkl'%val_split)
    
    trainval_filename = save_path / 'mydata_infos_trainval.pkl'
    test_filename = save_path / 'mydata_infos_test.pkl'

    files_list_pcd,files_list_label =dataset.get_files_name_list()
    # Take the data of TRAIN_RATIO_OF_ALL_LABELS(0.5) from the total list label as the training set train,
    # The rest are treated as val, and get the corresponding file path list
    files_list_label_train = random.sample(files_list_label,int(TRAIN_RATIO_OF_ALL_LABELS*len(files_list_label)))
    files_list_label_val = dataset.list_subtraction(files_list_label,files_list_label_train)
    files_list_pcd_train = dataset.from_labels_path_list_to_pcd_path_list(files_list_label_train)
    files_list_pcd_val = dataset.from_labels_path_list_to_pcd_path_list(files_list_label_val)

    #Assign the parameters in the class
    dataset.files_list_pcd =files_list_pcd
    dataset.files_list_label =files_list_label
    dataset.files_list_label_train =files_list_label_train
    dataset.files_list_label_val =files_list_label_val
    dataset.files_list_pcd_train = files_list_pcd_train
    dataset.files_list_pcd_val = files_list_pcd_val

    print('------------------------Start to generate data infos-----------------------')

    mydata_infos_train = dataset.get_all_labels(files_list_label=files_list_label_train)
    with open(train_filename,'wb') as f:
        pickle.dump(mydata_infos_train,f)
    print('mydata info train file is saved to %s'%train_filename)

    mydata_infos_val = dataset.get_all_labels(files_list_label=files_list_label_val)
    with open(val_filename,'wb') as f:
        pickle.dump(mydata_infos_val,f)
    print('mydata info val file is saved to %s'%val_filename)

    with open(trainval_filename,'wb') as f:
        pickle.dump(mydata_infos_train + mydata_infos_val,f)
    print('mydata info trainval file is saved to %s'%trainval_filename)

    mydata_infos_test = dataset.get_all_labels(files_list_label=files_list_label)
    with open (test_filename,'wb') as f:
        pickle.dump(mydata_infos_test,f)
    print('mydata info test file is saved to %s'%test_filename)

    print('---------------------Strat create groundtruth database for data augmentation ----------------')
    dataset.create_groundtruth_database(info_path=train_filename,split=train_split)
    print('---------------------Congratulation !  Data preparation Done !!!!!!---------------------------')

    pass


if __name__ == '__main__':
    import sys
    if sys.argv.__len__()>1 and sys.argv[1] == 'create_mydata_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        class_names= ['Car']
        
        create_mydata_infos(
            dataset_cfg=dataset_cfg,
            class_names= class_names,
            data_path= ROOT_DIR / 'data' / 'mydata',
            save_path= ROOT_DIR / 'data' / 'mydata'
        )


