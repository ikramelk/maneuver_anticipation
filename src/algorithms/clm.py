from numpy import *
import os
from scipy import io
import ast
from matplotlib.path import Path
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from flask import flash,render_template
import matplotlib.pyplot as plt 
import mpld3

class ManAnt:
class GOCLM:
    # ==========================Initialization================================ #

    def __init__(self, **kwargs):
        """inisialize the Predictor"""
        # load settings from file
        # init attributes
    #Generation des observations par CLM Tracker pour la methode de prediction d'un seul fichier (manfile)
    def generateObservationsCLMAction(self,manfile):
        actions=['lturn','rturn','lchange','rchange','end_action']
        observationDir=os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/'
    def generateObservationsCLMAction(self,manfile):
        actions=['lturn','rturn','lchange','rchange','end_action']
        observationDir=os.path.abspath(".")+'/app/plugins/data/maneuver_anticipation/data/featuress2/'
        feature_type=13
        delta_frames=20
        window_width=20
        manfileparams=''
        main_dir=os.path.abspath(".")+'/app/plugins/maneuver_anticipation/new_params/'
        main_dir=os.path.abspath(".")+'/app/plugins/data/maneuver_anticipation/data/new_params/'
        
        for i in arange(0,len(actions)):
            action=actions[i]
            data2={}
            inputObs={}
            data_sample=[]
            input_sample=[]
            number=0
            curr_dir=main_dir+actions[i]+'/'
            param_list=os.listdir(curr_dir)
            for j in arange(1,len(param_list)):
                if param_list[j]== manfile[10:-4]:
                    manfileparams=curr_dir+param_list[j]+'/new_param_'+param_list[j]+'.mat'
                  
            for j in arange(1,len(param_list)):
                
                params_file=curr_dir+param_list[j]+'/new_param_'+param_list[j]+'.mat'   

                # faire le traitement sur nos fichiers sans le fichier a predire qu'on va le traiter tout seul

                
                if os.path.exists(params_file) and params_file!= manfileparams:
                    data=io.loadmat(params_file,squeeze_me=True, struct_as_record=False)
                    fs=data['param_new'].frame_start
                    fe=data['param_new'].frame_end
                    if ((fe - fs) < 20):
                        continue
                    count = 0
                    for k in arange(fe,fs-1,- delta_frames):
                        data['param_new'].frame_end =k
                        if k - window_width > fs:
                            data['param_new'].frame_start = k - window_width
                        else:
                            continue
                        features=self.extractFeatures(data['param_new'])
                        try:
                            data_sample[count]=self.returnFeatures(features,feature_type)
                        except IndexError:
                            data_sample.append(self.returnFeatures(features,feature_type))
                            
                        a=features['lane_features']
                        a.extend([features['speed_features'][2]])
                        try:
                            input_sample[count]=a
                        except IndexError:
                            input_sample.append(a)
                        count = count + 1
                    data2['t'+str(number)]=fliplr(asarray(data_sample).T)
                    inputObs['t'+str(number)]=fliplr(asarray(input_sample).T)
                    number=number + 1
            disp('saving ovservations:'+action)
            io.savemat(observationDir+'clm_'+action+'_f_'+str(feature_type)+'_ww_'+str(window_width)+'_df_'
                 +str(delta_frames)+'.mat',mdict={'data':data2,'inputObs':inputObs})
        
        # traitement du fichier a predire et enregistrer par suite les donnes dans un fichier (.mat) qui contient que les donnees du fichier a predire 
        #file to predict 
        data2={}
        inputObs={}
        data_sample=[]
        input_sample=[]
        if os.path.exists(manfileparams):
            data=io.loadmat(manfileparams,squeeze_me=True, struct_as_record=False)
            fs=data['param_new'].frame_start
            fe=data['param_new'].frame_end
            if ((fe - fs) >= 20):
                count = 0
                for k in arange(fe,fs-1,- delta_frames):
                    data['param_new'].frame_end =k
                    if k - window_width > fs:
                        data['param_new'].frame_start = k - window_width
                        features=self.extractFeatures(data['param_new'])
                        try:
                            data_sample[count]=self.returnFeatures(features,feature_type)
                        except IndexError:
                            data_sample.append(self.returnFeatures(features,feature_type))

                        a=features['lane_features']
                        a.extend([features['speed_features'][2]])
                        try:
                            input_sample[count]=a
                        except IndexError:
                            input_sample.append(a)
                        count = count + 1
                data2['t'+str(number)]=fliplr(asarray(data_sample).T)
                inputObs['t'+str(number)]=fliplr(asarray(input_sample).T)
                number=number + 1
                disp('saving predecting file ')
                io.savemat(observationDir+'clm_'+'maneuver'+'_f_'+str(feature_type)+'_ww_'+str(window_width)+'_df_'
                +str(delta_frames)+'.mat',mdict={'data':data2,'inputObs':inputObs})
            else:
                print('choose another file wich (frame_end - frame_start) >= 20')
            
        
        return
    #generation des observations par CLM Tracker pour la 2 eme methode de cross validation 
    def generateObservationsCLM(self):
        actions=['lturn','rturn','lchange','rchange','end_action']
        observationDir=os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/'
        feature_type=13
        delta_frames=20
        window_width=20
        main_dir=os.path.abspath(".")+'/app/plugins/maneuver_anticipation/new_params/'
    
    def generateObservationsCLM(self):
        actions=['lturn','rturn','lchange','rchange','end_action']
        observationDir=os.path.abspath(".")+'/app/plugins/data/maneuver_anticipation/data/featuress/'
        feature_type=13
        delta_frames=20
        window_width=20
        main_dir=os.path.abspath(".")+'/app/plugins/data/maneuver_anticipation/data/new_params/'
        
        for i in arange(0,len(actions)):
            action=actions[i]
            data2={}
            inputObs={}
            data_sample=[]
            input_sample=[]
            number=0
            curr_dir=main_dir+actions[i]+'/'
            param_list=os.listdir(curr_dir)

            #on traite tous les fichiers qui se trouve dans le repertoire(dataset)
            
            for j in arange(1,len(param_list)):
                params_file=curr_dir+param_list[j]+'/new_param_'+param_list[j]+'.mat'   
                if os.path.exists(params_file):
                    data=io.loadmat(params_file,squeeze_me=True, struct_as_record=False)
                    fs=data['param_new'].frame_start
                    fe=data['param_new'].frame_end
                    if ((fe - fs) < 20):
                        continue
                    count = 0
                    for k in arange(fe,fs-1,- delta_frames):
                        data['param_new'].frame_end =k
                        if k - window_width > fs:
                            data['param_new'].frame_start = k - window_width
                        else:
                            continue
                        features=self.extractFeatures(data['param_new'])
                        try:
                            data_sample[count]=self.returnFeatures(features,feature_type)
                        except IndexError:
                            data_sample.append(self.returnFeatures(features,feature_type))
                            
                        a=features['lane_features']
                        a.extend([features['speed_features'][2]])
                        try:
                            input_sample[count]=a
                        except IndexError:
                            input_sample.append(a)
                        count = count + 1
                    data2['t'+str(number)]=fliplr(asarray(data_sample).T)
                    inputObs['t'+str(number)]=fliplr(asarray(input_sample).T)
                    number=number + 1
            disp('saving ovservations:'+action)
            io.savemat(observationDir+'clm_'+action+'_f_'+str(feature_type)+'_ww_'+str(window_width)+'_df_'
                 +str(delta_frames)+'.mat',mdict={'data':data2,'inputObs':inputObs})

        flash('Features has been generated successfully !!','success')
        return render_template("maneuver.html")

    #selection des variables on a utilise le type 13 seulement 

    def returnFeatures(self,features,feature_type):

        if feature_type == 1:
            feature_out=[]
            feature_out.extend(features['hist_angle_subframe'])
            feature_out.extend(features['hist_move_in_x_subframe'])
            feature_out.extend(features['hist_distance_subframe'])
        else:
            if feature_type == 2:
                feature_out=features['hist_angle_subframe']
            else:
                if feature_type == 3:
                    feature_out=[]
                    feature_out=features['hist_angle']
                    feature_out.extend(features['hist_move_in_x'])
                    feature_out.extend(features['hist_distance'])
                else:
                    if feature_type == 4:
                        feature_out=features['hist_angle']
                    else:
                        if feature_type == 5:
                            feature_out=[]
                            feature_out.extend(features['bbox_center']['movement_x_positive'])
                            feature_out.extend(features['bbox_center']['movement_x_negative'])
                            feature_out.extend(features['bbox_center']['motion_angle'])
                            feature_out.extend(features['bbox_center']['net_displacement'])
                        else:
                            if feature_type == 6:
                                feature_out=features['hog']
                            else:
                                if feature_type == 7:
                                    feature_out=features['bbox_center']['xcenter']
                                else:
                                    if feature_type == 8:
                                        feature_out=[]
                                        feature_out.extend(features['unified_subframe_hist'])
                                        feature_out.extend([mean(features['mean_movement_x'])])
                                    else:
                                        if feature_type == 9:
                                            feature_out=[]
                                            feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                            feature_out.extend([mean(features['mean_movement_x'])])
                                        else:
                                            if feature_type == 10:
                                                feature_out=mean(features['mean_movement_x'])
                                            else:
                                                if feature_type == 11:
                                                    feature_out=[]
                                                    feature_out.extend(features['unified_fullframe_hist'])
                                                    feature_out.extend([mean(features['mean_movement_x'])])
                                                else:
                                                    if feature_type == 12:
                                                        feature_out=[]
                                                        feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                                        feature_out.extend(features['unified_fullframe_hist'][0])
                                                        feature_out.extend([mean(features['mean_movement_x'])])
                                                    else:
                                                        if feature_type == 13:
                                                            feature_out=[]
                                                            feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                                            feature_out.extend(features['unified_fullframe_hist'][0])
                                                            feature_out.extend([mean(features['mean_movement_x'])])
                                                            feature_out.extend(features['lane_features'])
                                                            feature_out.extend([features['speed_features'][2]])
                                                        else:
                                                            if feature_type == 14:
                                                                feature_out=[]
                                                                feature_out.extend(features['lane_features'])
                                                                feature_out.extend(features['speed_features'][2])
                                                            else:
                                                                if feature_type == 15:
                                                                    feature_out=[]
                                                                    feature_out.extend(features['euler'])
                                                                    feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                                                    feature_out.extend(features['unified_fullframe_hist'])
                                                                    feature_out.extend([mean(features['mean_movement_x'])])
                                                                    feature_out.extend(features['lane_features'])
                                                                    feature_out.extend(features['speed_features'][2])
                                                                else:
                                                                    if feature_type == 16:
                                                                        feature_out=[]
                                                                        feature_out.extend(features['lane_features'])
                                                                        feature_out.extend(features['speed_features'][2])
                                                                    else:
                                                                        if feature_type == 17:
                                                                            feature_out=[]
                                                                            feature_out.extend(features['euler'])
                                                                            feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                                                            feature_out.extend(features['lane_features'])
                                                                            feature_out.extend(features['speed_features'][2])
                                                                        else:
                                                                            if feature_type == 18:
                                                                                feature_out=[]
                                                                                feature_out.extend(features['euler'])
                                                                                feature_out.extend(features['hist_mean_movement_x'] / linalg.norm(features['hist_mean_movement_x']))
                                                                                feature_out.extend(features['unified_fullframe_hist'])
                                                                                feature_out.extend(mean(features['mean_movement_x']))
        
        return feature_out

    #l'extraction des variables 

    def extractFeatures(self,params):
        features={}
        hist_angle_values=[- 0.75*pi,- 0.25*pi,0.25*pi,0.75*pi]
        hist_distance_values = [1.5,2.5,7.5,8.5,11.5]
        hist_distance_x = [-3.0,-1.0,1.0,3.0]
        frame_start=params.frame_start
        frame_end=params.frame_end
        if frame_end <= frame_start:
            features={}
            return features
        
        bbox_center=[[],[]]
        Time=[]
        
        
        features_hist_angle=[]
        features_hist_distance=[]
        features_hist_move_in_x=[]
        features_mean_movement=[]
        features_mean_movement_x=[]
        features_hist_angle_subframe=[]
        features_hist_distance_subframe=[]
        features_hist_move_in_x_subframe=[]
        mean_frame_movement=[]
        mean_subframe_movement=[]
        speedInfo=[]
        iter=0
        transform=[]
        unified_subframe_hist=zeros((4,len(hist_angle_values)))
        unified_fullframe_hist=zeros((1,len(hist_angle_values)))
        if params.laneInfo=='' or params.laneInfo=='-1':
            lane_features=[1,1,0]
        else:
            lane=params.laneInfo.split(',')
            lane_no=ast.literal_eval(lane[0])
            total_lanes=ast.literal_eval(lane[1])
            intersection=ast.literal_eval(lane[2])
            if total_lanes > lane_no:
                left_action=1
            else:
                left_action=0
            if lane_no > 1:
                right_action=1
            else:
                right_action=0
            lane_features=[left_action,right_action,intersection]
        
        overall_points=[]
        motion_vectors=[]
        time_instants=[]
        euler=zeros((1,3))
        for i in arange(frame_start-1,frame_end-1):
            frame_data_cur=params.frame_data[i]
            frame_data_next=params.frame_data[i+1]
            if params.tracker=='CLM':
                euler=euler + frame_data_cur.Euler.T
            bbox_cur=frame_data_cur.box
            bbox_next=frame_data_next.box
            bbox_center=column_stack((bbox_center,[bbox_cur[0]+0.5*bbox_cur[2],bbox_cur[1]+0.5*bbox_cur[3]]))
            match_cur=frame_data_cur.match_next
            match_next=frame_data_next.match_prev
            speedInfo.append(frame_data_cur.speed)
            if match_next.shape[1] == 0:
                match_cur=bbox_center[:,iter]
                match_next=bbox_center[:,iter]
            if params.tracker=='KLT':
                transform[iter]=frame_data_cur.T
                if iter > 1:
                    transform[iter]=dot(transform[iter - 1],frame_data_cur.T)
                    if size(match_cur,2) < 20:
                        transform[iter]=transform[iter - 1]
            points_move_vec=match_next - match_cur
            points_distance=sqrt((points_move_vec**2).sum(axis=0))
            points_angle=arctan2(points_move_vec[1,:],points_move_vec[0,:])
            points_move_in_x=points_move_vec[1,:]
            overall_points.extend(points_angle)
            motion_vectors=[[motion_vectors],[points_move_vec.T]]
            time_instants=[[time_instants],[kron(ones(1),(dot((i - frame_start),1.0) / 25,points_move_vec.shape[1]))]]
            sub_bbox=self.subBBoxes(bbox_cur,bbox_next)
            
            
            
            features_hist_angle_fullframe = self.histo(points_angle,hist_angle_values);
            features_hist_distance_fullframe = self.histo(points_distance,hist_distance_values);
            features_hist_move_in_x_fullframe = self.histo(points_move_in_x,hist_distance_x);
            
            unified_fullframe_hist=unified_fullframe_hist + features_hist_angle_fullframe
            features_hist_angle.extend(features_hist_angle_fullframe / max(1.0,linalg.norm(features_hist_angle_fullframe)))
            features_hist_distance.extend(features_hist_distance_fullframe / max(1.0,linalg.norm(features_hist_distance_fullframe)))
            features_hist_move_in_x.extend(features_hist_move_in_x_fullframe / max(1.0,linalg.norm(features_hist_move_in_x_fullframe)))
            features_mean_movement.append(mean(points_distance))
            features_mean_movement_x.append(mean(points_move_in_x))
            
            for j in arange(0,len(sub_bbox)):
                sub_bbox_cur=sub_bbox[j][0]
                sub_bbox_next=sub_bbox[j][1]
                
                Xv,Yv=self.BBoxToXvYv(sub_bbox_cur)
                points = array([match_cur[0,:],match_cur[1,:]]).reshape(68, 2)
                p=array([Xv,Yv]).reshape(4, 2)
                path = Path(p)
                IN_cur=double(path.contains_points(points))
                
                Xv,Yv=self.BBoxToXvYv(sub_bbox_next)
                points1 = array([match_next[0,:],match_next[1,:]]).reshape(68, 2)
                p1=array([Xv,Yv]).reshape(4, 2)
                path1 = Path(p1)
                IN_next=double(path1.contains_points(points1))
                IN=multiply(IN_cur,IN_next)
                features_hist_angle_=self.histo(points_angle[nonzero(IN == 1)],hist_angle_values)
                features_hist_distance_=self.histo(points_distance[nonzero(IN == 1)],hist_distance_values)
                features_hist_move_in_x_=self.histo(points_move_in_x[nonzero(IN == 1)],hist_distance_x)
                
                unified_subframe_hist[j,:]=unified_subframe_hist[j,:] + features_hist_angle_
                features_hist_angle_=features_hist_angle_ / max(1.0,linalg.norm(features_hist_angle_))
                features_hist_distance_=features_hist_distance_ / max(1.0,linalg.norm(features_hist_distance_))
                features_hist_move_in_x_=features_hist_move_in_x_ / max(1.0,linalg.norm(features_hist_move_in_x_))
                features_hist_angle_subframe.extend(features_hist_angle_)
                features_hist_distance_subframe.extend(features_hist_distance_)
                features_hist_move_in_x_subframe.extend(features_hist_move_in_x_)
            mean_frame_movement.append(mean(points_move_vec))
            iter=iter + 1
            
            
        if params.tracker=='CLM':
            euler=euler / (frame_end - frame_start + 1)
            features['euler'] = copy(euler)
        unified_fullframe_hist=unified_fullframe_hist / max(1.0,linalg.norm(unified_fullframe_hist))
        unified_subframe_hist=unified_subframe_hist / kron(ones(1),(maximum(ones((4,1),float),unified_subframe_hist.sum(axis=1)),1,len(hist_angle_values)))[0]
        features['unified_subframe_hist'] = copy(reshape(unified_subframe_hist,(1,dot(4,len(hist_angle_values))),order='F'))
        features['unified_fullframe_hist']= copy(unified_fullframe_hist)
        features['hist_angle']= copy(features_hist_angle)
        features['hist_distance']= copy(features_hist_distance)
        features['hist_move_in_x']= copy(features_hist_move_in_x)
        features['mean_movement'] = copy(features_mean_movement)
        features['mean_movement_x']= copy(features_mean_movement_x)
        features['hist_mean_movement_x']= copy(self.histo(features_mean_movement_x,hist_distance_x))
        features['hist_angle_subframe'] = copy(features_hist_angle_subframe)
        features['hist_distance_subframe'] = copy(features_hist_distance_subframe)
        features['hist_move_in_x_subframe']= copy(features_hist_move_in_x_subframe)
        
        
        if params.tracker=='KLT':
            features['face_transform'] = copy(transform)
            
        motion_vector=bbox_center[:,1:] - bbox_center[:,:-1]
        features['bbox_center'] = self.BBoxCenterTrajectoryFeature(motion_vector,bbox_center[0,:])
        img=zeros((960,1))
        
        
        if params.tracker=='KLT':
            face_mean_x_pos=mean(bbox_center,2)
            face_mean_x_pos=face_mean_x_pos[1]
            for i in arange(frame_start,frame_end).reshape(-1):
                frame_data=params.frame_data[i]
                klt_points=frame_data.klt_points[0,:]
                klt_points=klt_points.T
                img[round(klt_points - face_mean_x_pos + 300),i - frame_start + 1]=1
            img=flipud(img)
            features.hog = copy(double(extractHOGFeatures(img)))
            features.hog_image = copy(img)
            
        speedInfo=delete(speedInfo, nonzero(array(speedInfo)==-1))
        'speedInfo[nonzero(array(speedInfo)==-1)[0]]=[]'
        
        
        if len(speedInfo) == 0:
            speed_features=[30.0 / 160,30.0 / 160,30.0 / 160]
        else:
            speed_features=dot((1.0 / 160),[max(speedInfo),min(speedInfo),mean(speedInfo)])
        
        features['speed_features']= speed_features
        features['lane_features']= lane_features
        Xmark=cos(overall_points)
        Ymark=sin(overall_points)
        return features
        
    if __name__ == '__main__':
        pass
        
    #traitement du trajectoire des points du visage 
    
    def BBoxCenterTrajectoryFeature(self,motion_vector,bbox_xcenter):

        bbox_center={}
        movement=motion_vector.sum(axis=1)
        movement_x_positive=sum(motion_vector[0,nonzero(motion_vector[0,:] > 0)])
        movement_x_negative=sum(motion_vector[0,nonzero(motion_vector[0,:] < 0)])
        motion_angle=arctan2(movement[1],movement[0])
        net_displacement=linalg.norm(movement)
        bbox_center['movement_x_positive'] = movement_x_positive
        bbox_center['movement_x_negative'] = movement_x_negative
        bbox_center['motion_angle'] = motion_angle
        bbox_center['net_displacement'] = net_displacement
        delta=max(bbox_xcenter) - min(bbox_xcenter)
        overlap=dot((dot(delta,1.0) / 3.0),(1.0 / 10.0))
        l1=min(bbox_xcenter)
        l2=l1 + dot(delta,1.0) / 3.0
        l3=l1 + dot(delta,2.0) / 3.0
        hist_bbox_xcenter=[]
        width=[]
        idx=nonzero(bbox_xcenter < (l2 + overlap))[0]
        if len(idx)!=0:
            width.append(max(idx) - min(idx))
        hist_bbox_xcenter.append(len(idx))
        idx=nonzero(logical_and(bbox_xcenter > (l2 - overlap), bbox_xcenter < (l3 + overlap)))[0]
        if len(idx)!=0:
            width.append(max(idx) - min(idx))
        hist_bbox_xcenter.append(len(idx))
        idx=nonzero(bbox_xcenter > (l3 - overlap))[0]
        if len(idx)!=0:
            width.append(max(idx) - min(idx))
        hist_bbox_xcenter.append(len(idx))
        width=[float(x) / max(width) for x in width]
        hist_bbox_xcenter=[float(x) / sum(hist_bbox_xcenter) for x in hist_bbox_xcenter]
        bbox_center['xcenter'] = concatenate((hist_bbox_xcenter,width))
        return bbox_center
        
    if __name__ == '__main__':
        pass
        
    
    def BBoxToXvYv(self,bbox):
        Xv=[bbox[0],bbox[0] + bbox[2],bbox[0] + bbox[2],bbox[0]]
        Yv=[bbox[1],bbox[1],bbox[1] + bbox[3],bbox[1] + bbox[3]]
        return Xv,Yv
        
    if __name__ == '__main__':
        pass
    
    
    def subBBoxes(self,bbox_cur,bbox_next):
        sub_bbox={}
        w_cur=round(dot(bbox_cur[2],0.5))
        h_cur=round(dot(bbox_cur[3],0.5))
        w_next=round(dot(bbox_next[2],0.5))
        h_next=round(dot(bbox_next[3],0.5))
        sub_bbox[0] = [[bbox_cur[0],bbox_cur[1],w_cur,h_cur],[bbox_next[0],bbox_next[1],w_next,h_next]]
        sub_bbox[1] = [[bbox_cur[0] + w_cur,bbox_cur[1],w_cur,h_cur],[bbox_next[0] + w_next,bbox_next[1],w_next,h_next]]
        sub_bbox[2] = [[bbox_cur[0],bbox_cur[1] + h_cur,w_cur,h_cur],[bbox_next[0],bbox_next[1] + h_next,w_next,h_next]]
        sub_bbox[3] = [[bbox_cur[0] + w_cur,bbox_cur[1] + h_cur,w_cur,h_cur],[bbox_next[0] + w_next,bbox_next[1] + h_next,w_next,h_next]]
        return sub_bbox
        
    if __name__ == '__main__':
        pass
    
    def histo(self,A,centers):
        d=diff(centers)/2
        x = [centers[0]-d[0]]
        y=centers[0:-1]+d
        z=[centers[-1]+d[-1]]
        x.extend(y)
        vv=x
        vv.extend(z)
        vv[1:-1] = vv[1:-1]+spacing(vv[1:-1])
        hi,bi=histogram(A,vv)
        return hi


    #l'apprentissage du modele pour le cas d'un seul fichier 
    def AIOhmmTrainAction(self):
        train_model_for='turns'
        use_fixed_split=False
        learnedModels={}
        testData={}
        trainData={}
        testObs={}
        trainObs={}
        separate_actions=True
        retrain_models=True
        delta_frames=20
        fname_ext='_f_13_ww_20_df_20.mat'
        if separate_actions:
            if train_model_for=='turns':
                actions=['rturn','lturn','end_action']
                showName=['Right Turn','Left Turn','Straight']
                fixed_split_name='fold_turns_'
                thresh=60
            else:
                if train_model_for=='lane':
                    actions=['lchange','rchange','end_action']
                    showName=['L Lane','R Lane','Straight']
                    fixed_split_name='fold_io2_'
                    thresh=130
                else:
                    if train_model_for=='all':
                        actions=['lchange','rchange','lturn','rturn','end_action']
                        showName=['L Lane','R Lane','L Turn','R Turn','Straight']
                        thresh=130
                        fixed_split_name=''
                    else:
                        actions=['lchange','rchange','end_action']
                        showName=['L Lane','R Lane','Straight']
                        train_model_for='turns'
                        fixed_split_name='fold_io2_'
                        thresh=130
        else:
            train_model_for='combined'
            actions=['combined','end_action']
        
        # pour refaire l'apprentissage du modele changer arange par le nombre de fois que vous voulez
        if retrain_models:
            for numiter in arange(1,2):
                if use_fixed_split:
                    if os.path.exists(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/'+'clm_'+fixed_split_name+str(numiter)+'.mat'):
                        io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/'+'clm_'+fixed_split_name+str(numiter)+'.mat')
        
        if retrain_models:
            for numiter in arange(1,2):
                if use_fixed_split:
                    if os.path.exists(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress2/'+'clm_'+fixed_split_name+str(numiter)+'.mat'):
                        io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress2/'+'clm_'+fixed_split_name+str(numiter)+'.mat')
                    else:
                        use_fixed_split=False
                learnedModels_=[None]*len(actions)
                testData_=[None]*len(actions)
                trainData_=[None]*len(actions)
                testObs_=[None]*len(actions)
                trainObs_=[None]*len(actions)
                for i in arange(0,len(actions)):
                    action=actions[i]
                    if use_fixed_split:
                        test_data=testData[action]
                        train_data=trainData[action]
                        test_obs=testObs[action]
                        train_obs=trainObs[action]
                    else:
                        load_data=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/'+'clm_'+action+fname_ext)
                        load_data=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress2/'+'clm_'+action+fname_ext)
                        data=[]
                        inputObs=[]
                        data=load_data['data'][0][0]
                        inputObs=load_data['inputObs'][0][0]
                        
                        load_data_test=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/'+'clm_'+'maneuver'+fname_ext)    
                        load_data_test=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/datafeaturess2/'+'clm_'+'maneuver'+fname_ext)    
                        
                        test_data=load_data_test['data'][0][0]
                        test_obs=load_data_test['inputObs'][0][0]
                        
                        
                        train_data=load_data['data'][0][0]
                        train_obs=load_data['inputObs'][0][0]
                        
                        
                        '''test_data0=array([array(data[z]).ravel() for z in folds['f'+str(i)]])
                        test_obs0=array([array(inputObs[z]).ravel() for z in folds['f'+str(i)]])
                        train_data0=array([array(data[z]).ravel() for z in train_idx])
                        train_obs0=array([array(inputObs[z]).ravel() for z in train_idx])'''
                        
                    model={}
                    model['type'] = 'gauss'
                    model['iotype'] = True
                    model['observationDimension'] = len(train_data[0])
                    model['inputDimension'] = len(train_obs[0])
                    if action=='end_action':
                        model['nstates'] =3
                    else:
                        if action=='rchange':
                            model['nstates'] =6
                        else:
                            if action=='lchange':
                                model['nstates'] = 5
                            else:
                                if action=='rturn':
                                    model['nstates'] = 3
                                else:
                                    if action=='lturn':
                                        model['nstates'] = 3
                                    else:
                                        if action=='combined':
                                            model['nstates'] = 6
                    print(action)
                    model=self.initializeHMMmodel(train_data,model['type'],model['nstates'],model['observationDimension'],model['iotype'],model['inputDimension'])
                    bparam={}
                    for k in arange(0,model['nstates']):
                        bparam['bp'+str(k)]=((1.0 / model['observationDimension']*zeros((model['observationDimension'],1))))
                    model['bparam']=bparam
                    model['action'] = action
                    model['learn_aparam'] = True
                    prior={}
                    prior['use'] = 1
                    prior['k0'] =1
                    prior['mu0'] =dot((1.0 / model['observationDimension']),ones((model['observationDimension'],1)))
                    prior['Psi'] =dot((1.0 / model['observationDimension']),eye(model['observationDimension'],model['observationDimension']))
                    model['prior']=prior
                    
                    model,ll=self.AIOhmmFit(model,train_data,train_obs,model)
                    learnedModels_[i]=model
                    trainData_[i]=train_data
                    trainObs_[i]=train_obs
                    
                for i in arange(0,len(actions)):
                    action=actions[i]
                    learnedModels[action]=learnedModels_[i]
                    trainData[action]=trainData_[i]
                    trainObs[action]=trainObs_[i]
                io.savemat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext,mdict={'learnedModels':learnedModels,'testData':test_data,'trainData':trainData,'testObs':test_obs,'trainObs':trainObs})
                io.savemat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress2/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext,mdict={'learnedModels':learnedModels,'testData':test_data,'trainData':trainData,'testObs':test_obs,'trainObs':trainObs})

        
        THRESH=0.7
        time_global=[]
        true_label=[]
        predicted_label=[]
        accuracy=[]
        for numiter in arange(1,2):
            loadedfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress2/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext)
            loadedfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress2/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext)
            prediction_local=[]
            true_label_local=[]
            test_data=[]
            test_obs=[]
            if len(loadedfile['testData'][0][0])==1:
                test_data_bis=loadedfile['testData'][0][0][0]
            else:
                test_data_bis=loadedfile['testData'][0][0]

            '''for l in arange(0,len(test_data_bis)):
                if len(test_data_bis[l])==1:
                    test_data.append(test_data_bis[l][0].reshape(13,len(test_data_bis[l][0])/13))
                else:
                    test_data.append(test_data_bis[l].reshape(13,len(test_data_bis[l])/13))'''



            if len(loadedfile['testData'][0][0])==1:
                test_obs_bis=loadedfile['testObs'][0][0][0]
            else:
                test_obs_bis=loadedfile['testObs'][0][0]

            '''for l in arange(0,len(test_obs_bis)):
                if len(test_obs_bis[l])==1:
                    test_obs.append(test_obs_bis[l][0].reshape(4,len(test_obs_bis[l][0])/4))
                else:
                    test_obs.append(test_obs_bis[l].reshape(4,len(test_obs_bis[l])/4))'''

            prediction_probability=self.timeSeriesPrediction(loadedfile['learnedModels'],test_data_bis,test_obs_bis,actions)
            action_predict,time_before=self.predictAction(prediction_probability,actions,delta_frames,THRESH)

            print(showName[action_predict])

            
        return showName[action_predict]

    #l'apprentissage du modele pour le cas de cross validation
    def AIOhmmTrain(self):
        train_model_for='turns'
        use_fixed_split=False
        learnedModels={}
        testData={}
        trainData={}
        testObs={}
        trainObs={}
        separate_actions=True
        retrain_models=True
        delta_frames=20
        if separate_actions:
            if train_model_for=='turns':
                actions=['rturn','lturn','end_action']
                showName=['Right Turn','Left Turn','Straight']
                fixed_split_name='fold_turns_'
                thresh=60
            else:
                if train_model_for=='lane':
                    actions=['lchange','rchange','end_action']
                    showName=['L Lane','R Lane','Straight']
                    fixed_split_name='fold_io2_'
                    thresh=130
                else:
                    if train_model_for=='all':
                        actions=['lchange','rchange','lturn','rturn','end_action']
                        showName=['L Lane','R Lane','L Turn','R Turn','Straight']
                        thresh=130
                        fixed_split_name=''
                    else:
                        actions=['lchange','rchange','end_action']
                        showName=['L Lane','R Lane','Straight']
                        train_model_for='turns'
                        fixed_split_name='fold_io2_'
                        thresh=130
        else:
            train_model_for='combined'
            actions=['combined','end_action']
        
        fname_ext='_f_13_ww_20_df_20.mat'

        folds_action={}
         #faire une validation croisee sur notre dataset pour chaque action ['lchange','rchange','lturn','rturn','end_action'] on divise le dossier sur 5
        for i in arange(0,len(actions)):
            action=actions[i]
            loadfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/'+'clm_'+action+fname_ext)
        for i in arange(0,len(actions)):
            action=actions[i]
            loadfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/'+'clm_'+action+fname_ext)
            
            size_fold=round(float(min(len(loadfile['data'][0][0]),thresh)) / 5.0)
            rand_list=random.permutation(min(len(loadfile['data'][0][0]),thresh))
            folds={}
            folds['f0']=rand_list[0:int(size_fold)]
            folds['f1']=rand_list[(int(size_fold)):(dot(2,int(size_fold)))]
            folds['f2']=rand_list[(dot(2,int(size_fold))):(dot(3,int(size_fold)))]
            folds['f3']=rand_list[(dot(3,int(size_fold))):(dot(4,int(size_fold)))]
            folds['f4']=rand_list[dot(4,int(size_fold)):(min(thresh,len(loadfile['data'][0][0])))];
            folds_action[action]=folds

        if retrain_models:
            for numiter in arange(1,2):
                if use_fixed_split:
                    if os.path.exists(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/'+'clm_'+fixed_split_name+str(numiter)+'.mat'):
                        io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/'+'clm_'+fixed_split_name+str(numiter)+'.mat',squeeze_me=True, struct_as_record=False)
        if retrain_models:
            for numiter in arange(1,2):
                if use_fixed_split:
                    if os.path.exists(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/'+'clm_'+fixed_split_name+str(numiter)+'.mat'):
                        io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/'+'clm_'+fixed_split_name+str(numiter)+'.mat',squeeze_me=True, struct_as_record=False)
                    else:
                        use_fixed_split=False
                learnedModels_=[None]*len(actions)
                testData_=[None]*len(actions)
                trainData_=[None]*len(actions)
                testObs_=[None]*len(actions)
                trainObs_=[None]*len(actions)
                for i in arange(0,len(actions)):
                    action=actions[i]
                    if use_fixed_split:
                        test_data=testData[action]
                        train_data=trainData[action]
                        test_obs=testObs[action]
                        train_obs=trainObs[action]
                    else:
                        folds=folds_action[action]
                        load_data=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/'+'clm_'+action+fname_ext)
                        load_data=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/'+'clm_'+action+fname_ext)
                        data=load_data['data'][0][0]
                        inputObs=load_data['inputObs'][0][0]
                        test_data=[data[z] for z in folds['f'+str(i)]]
                        test_obs=[inputObs[z] for z in folds['f'+str(i)]]
                        train_idx=arange(0,min(thresh,len(data)))
                        train_idx=delete(train_idx,folds['f'+str(i)])
                        train_data=[data[z] for z in train_idx]
                        train_obs=[inputObs[z] for z in train_idx]
                        
                        
                        test_data0=array([array(data[z]).ravel() for z in folds['f'+str(i)]])
                        test_obs0=array([array(inputObs[z]).ravel() for z in folds['f'+str(i)]])
                        train_data0=array([array(data[z]).ravel() for z in train_idx])
                        train_obs0=array([array(inputObs[z]).ravel() for z in train_idx])

                    model={}
                    model['type'] = 'gauss'
                    model['iotype'] = True
                    model['observationDimension'] = len(train_data[0])
                    model['inputDimension'] = len(train_obs[0])
                    if action=='end_action':
                        model['nstates'] =3
                    else:
                        if action=='rchange':
                            model['nstates'] =6
                        else:
                            if action=='lchange':
                                model['nstates'] = 5
                            else:
                                if action=='rturn':
                                    model['nstates'] = 3
                                else:
                                    if action=='lturn':
                                        model['nstates'] = 3
                                    else:
                                        if action=='combined':
                                            model['nstates'] = 6
                    print(action)
                    model=self.initializeHMMmodel(train_data,model['type'],model['nstates'],model['observationDimension'],model['iotype'],model['inputDimension'])
                    bparam={}
                    for k in arange(0,model['nstates']):
                        bparam['bp'+str(k)]=((1.0 / model['observationDimension']*zeros((model['observationDimension'],1))))
                    model['bparam']=bparam
                    model['action'] = action
                    model['learn_aparam'] = True
                    prior={}
                    prior['use'] = 1
                    prior['k0'] =1
                    prior['mu0'] =dot((1.0 / model['observationDimension']),ones((model['observationDimension'],1)))
                    prior['Psi'] =dot((1.0 / model['observationDimension']),eye(model['observationDimension'],model['observationDimension']))
                    model['prior']=prior
                    
                    model,ll=self.AIOhmmFit(model,train_data,train_obs,model)
                    learnedModels_[i]=model
                    testData_[i]=test_data0
                    trainData_[i]=train_data0
                    testObs_[i]=test_obs0
                    trainObs_[i]=train_obs0
                for i in arange(0,len(actions)):
                    action=actions[i]
                    learnedModels[action]=learnedModels_[i]
                    testData[action]=testData_[i]
                    trainData[action]=trainData_[i]
                    testObs[action]=testObs_[i]
                    trainObs[action]=trainObs_[i]
                io.savemat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext,mdict={'learnedModels':learnedModels,'testData':testData,'trainData':trainData,'testObs':testObs,'trainObs':trainObs,'folds_action':folds_action})
                io.savemat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext,mdict={'learnedModels':learnedModels,'testData':testData,'trainData':trainData,'testObs':testObs,'trainObs':trainObs,'folds_action':folds_action})
                
                
        threshlist=[]
        accuracylist=[]
        meantimelist=[]
        for THRESH in arange(0.1,1,0.1):
            time_global=[]
            true_label=[]
            predicted_label=[]
            accuracy=[]
            for numiter in arange(1,2):
                loadedfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/featuress/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext)
                loadedfile=io.loadmat(os.path.abspath(".")+'/app/plugins/maneuver_anticipation/data/featuress/learnedModel_'+train_model_for+'_aiohmm_'+str(numiter)+fname_ext)
                prediction_local=[]
                true_label_local=[]
                for i in arange(0,len(actions)):
                    action=actions[i]
                    test_data=[]
                    test_obs=[]
                    if len(loadedfile['testData'][action][0][0])==1:
                        test_data_bis=loadedfile['testData'][action][0][0][0]
                    else:
                        test_data_bis=loadedfile['testData'][action][0][0]
                        
                    for l in arange(0,len(test_data_bis)):
                        if len(test_data_bis[l])==1:
                            test_data.append(test_data_bis[l][0].reshape(13,len(test_data_bis[l][0])/13))
                        else:
                            test_data.append(test_data_bis[l].reshape(13,len(test_data_bis[l])/13))
                        
                    
                    
                    if len(loadedfile['testData'][action][0][0])==1:
                        test_obs_bis=loadedfile['testObs'][action][0][0][0]
                    else:
                        test_obs_bis=loadedfile['testObs'][action][0][0]
                    for l in arange(0,len(test_obs_bis)):
                        if len(test_obs_bis[l])==1:
                            test_obs.append(test_obs_bis[l][0].reshape(4,len(test_obs_bis[l][0])/4))
                        else:
                            test_obs.append(test_obs_bis[l].reshape(4,len(test_obs_bis[l])/4))
                
                    

                    for j in arange(0,len(test_data)):
                        true_label.extend([i])
                        true_label_local.extend([i])
                        prediction_probability=self.timeSeriesPrediction(loadedfile['learnedModels'],test_data[j],test_obs[j],actions)
                        action_predict,time_before=self.predictAction(prediction_probability,actions,delta_frames,THRESH)
                        predicted_label.extend([action_predict])
                        prediction_local.extend([action_predict])
                        time_global.extend([time_before])
                #print(str(len(nonzero(array(prediction_local) == true_label_local)[0]))+'/'+str(len(true_label_local)))
                accuracy.extend([dot(len(nonzero(array(prediction_local) == true_label_local)[0]),100.0) / len(true_label_local)])
            matching=nonzero(array(predicted_label) == true_label)[0]
            time_global=[time_global[i] for i in matching]
            delete(time_global,nonzero(array(time_global) == - 1)[0])
            #print('Mean time = '+str(mean(time_global)))
            confMat=matrix(self.confusionMatrix(predicted_label,true_label))
            #print('accuracy = '+str(mean(accuracy))+'('+str(std(accuracy))+')')
            #print(confMat)
            p_lchange=dot(confMat[0,0],100.0) / sum(confMat[:,0])
            p_rchange=dot(confMat[1,1],100.0) / sum(confMat[:,1])
            p_end_act=dot(confMat[2,2],100.0) / sum(confMat[:,2])
            #print('precision '+actions[0]+' = '+str(p_lchange))
            #print('precision '+actions[1]+' = '+str(p_rchange))
            #print('precision '+actions[2]+' = '+str(p_end_act))
            r_lchange=dot(confMat[0,0],100.0) / sum(confMat[0,:])
            r_rchange=dot(confMat[1,1],100.0) / sum(confMat[1,:])
            r_end_act=dot(confMat[2,2],100.0) / sum(confMat[2,:])
            #print('recall '+actions[0]+' = '+str(r_lchange))
            #print('recall '+actions[1]+' = '+str(r_rchange))
            #print('recall '+actions[2]+' = '+str(r_end_act))
            #print('Confusion Matrix: ')
            #print(confMat)
            threshlist.append(THRESH)
            accuracylist.append(mean(accuracy))
            meantimelist.append(mean(time_global))
        print(threshlist)
        print(accuracylist)
        print(meantimelist)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,gridspec_kw = {'width_ratios':[3, 5]}) 
        x = array(threshlist)
        y = array(accuracylist)
        ax1.plot(x, y)
        ax1.set_title('Effect of prediction threshold')
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Accuracy")
        x1 = array(meantimelist)
        ax2.scatter(x1, y)
        ax2.set_title('Effect of time-to-maneuver')
        ax2.set_xlabel("Time-to-maneuver")
        mpld3.show()
        flash('Prediction ended successfully !!','success')
        return render_template("maneuver.html")
        

    #fonction qui predit l'action elle retourne l'action le time to maneuver   
        
    def predictAction(self,probability_table,actions,delta_frames,THRESH):
        #THRESH = 0.7;
        time_before=- 1
        T=len(probability_table)
        action_taken=actions.index('end_action')
        for i in arange(0,T):
            val=max(probability_table[i,:])
            idx=probability_table[i,:].tolist().index(val)
            if val > THRESH and idx != action_taken and probability_table[i,action_taken] < THRESH:
                action_taken=idx
                time_before=dot(dot((T - idx),delta_frames),1.0) / 25.0
                return action_taken,time_before
        
        return action_taken,time_before
        
    if __name__ == '__main__':
        pass


    def timeSeriesPrediction(self,learnedModels,test_data,test_obs,actions):
        T=len(test_data[0])
        prediction_probability=zeros((T,len(actions)))
        for j in arange(0,T):
            test=[]
            obs=[]
            test.append(test_data.T[:][:j+1].T)
            obs.append(test_obs.T[:][:j+1].T)
            loglikelihood=zeros((1,len(actions)))[0].tolist()
            for i in arange(0,len(actions)):
                action=actions[i]
                model=learnedModels[action][0,0][0][0]
                
                ll=self.iomodelDataLoglikelihood(model,self.aiocalculateEvidence(model,test,obs),obs)
                
                loglikelihood[i]=ll[0]
            log_a_b=self.logTologOfSum(loglikelihood)
            prediction_probability[j,:]=exp(loglikelihood - log_a_b)
        
        return prediction_probability
        
    if __name__ == '__main__':
        pass

    def logTologOfSum(self,loglikelihood):
        if len(loglikelihood) == 2:
            v1=loglikelihood[0]
            v2=loglikelihood[1]
            log_a_b=self.logOfsum(v1,v2)
        else:
            v1=loglikelihood[0]
            v2=loglikelihood[1:]
            log_a_b=self.logOfsum(v1,self.logTologOfSum(v2))
        
        return log_a_b
        
    if __name__ == '__main__':
        pass

    def logOfsum(self,log_a, log_b):
        
        log_a_b = log(exp(log_a - log_b) + 1.0) + log_b
        
        return log_a_b
        
    if __name__ == '__main__':
        pass

    def confusionMatrix(self,predict,actual):
        lchange=nonzero(array(actual) == 0)[0]
        rchange=nonzero(array(actual) == 1)[0]
        nchange=nonzero(array(actual) == 2)[0]
        confMat=[]
        confMat.extend([[len(nonzero(array([predict[i] for i in lchange]) == 0)[0]),len(nonzero(array([predict[i] for i in lchange]) == 1)[0]),len(nonzero(array([predict[i] for i in lchange]) == 2)[0])]])
        confMat.extend([[len(nonzero(array([predict[i] for i in rchange]) == 0)[0]),len(nonzero(array([predict[i] for i in rchange]) == 1)[0]),len(nonzero(array([predict[i] for i in rchange]) == 2)[0])]])
        confMat.extend([[len(nonzero(array([predict[i] for i in nchange]) == 0)[0]),len(nonzero(array([predict[i] for i in nchange]) == 1)[0]),len(nonzero(array([predict[i] for i in nchange]) == 2)[0])]])
        return confMat
        
    if __name__ == '__main__':
        pass
    

    def AIOhmmFit(self,truemodel,data,inputObs,model):
        numiter=1
        ll=zeros((numiter,1))
        for i in arange(0,numiter):
            evidence=self.aiocalculateEvidence(model,data,inputObs)
            alph,bet,gam,xi,loglikelihood=self.FBWpass(model,evidence,inputObs)
            ll[i]=sum(loglikelihood)
            sufficient_statistics=self.estep(gam,model,data,inputObs)
            model=self.mstep(gam,xi,sufficient_statistics,model,inputObs,data)
            
        
        
        print(sum(loglikelihood))
        
        return model,ll
        
    if __name__ == '__main__':
        pass

    #Expectation_maximization algorithm,la fonction definie l'etape Maximisation

    def mstep(self,gam,xi,expected,model,inputObs,data_original):
        data={}
        N=len(gam)
        data['inputObs'] = inputObs
        data['nstates'] = model['nstates']
        data['inputDimension'] = model['inputDimension']
        data['xi'] = xi
        data['gam'] = gam
        funcObj=lambda parameters=None:self.completeObj0(parameters,data)[0]
        x0=reshape(model['piW'],(dot(model['inputDimension'],(model['nstates'] - 1)),1))
        parameters=minimize(funcObj,x0,method='L-BFGS-B',options={'maxiter': 500})
        model['piW'] = reshape(parameters['x'],(model['inputDimension'],model['nstates'] - 1)).T
        
        
        nstates=model['nstates']
        inputDimension=model['inputDimension']
        Wt=[None] *nstates
        data_complete=[None] *nstates
        for i in arange(0,model['nstates']):
            Wt[i]=model['W'][i,:,:]
            data['state'] = i
            data_complete[i]=data
        
        for i in arange(0,model['nstates']):
            funcObj=lambda parameters=None: self.completeObj(parameters,data_complete[i])[0]
            x0=reshape(reshape(Wt[i],(nstates - 1,inputDimension)).T,dot(inputDimension,(nstates - 1)),1)
            parameters=minimize(funcObj,x0,method='L-BFGS-B',options={'maxiter': 500})
            Wt[i]=reshape(parameters['x'],(inputDimension,nstates - 1)).T
        
        for i in arange(0,model['nstates']):
            model['W'][i,:,:]=Wt[i]
        
        if model['type']=='discrete':
            model['B'] = expected['observation_count'] / tile(expected['state_frequency'],(1,model['ostates']))
        else:
            if model['type']=='gauss':
                for i in arange(0,model['nstates']):
                    mu=expected['mean_vector'][:,i] / expected['state_frequency_const'][i]
                    xmu=zeros((model['observationDimension'],model['observationDimension']))
                    for x in arange(0,N):
                        model['a'] = model['aparam']['m'+str(i)]
                        model['b'] = model['bparam']['bp'+str(i)]
                        lgamma=gam[x]
                        data_sample=data_original[x]
                        T=len(lgamma[0])
                        modela=model['a']
                        modelb=model['b']
                        for s in arange(1,T):
                            modela=column_stack((modela,model['a']))
                        for s in arange(1,T-1):
                            modelb=column_stack((modelb,model['b']))
                        temp=[.0]
                        temp.extend(sum(multiply(modelb,data_sample[:,0:T-1]),0))
                        mult_const=(1.0 + sum(multiply(modela,inputObs[x]),0)) + temp
                        gamma_times_data=multiply(tile(lgamma[i,:],(model['observationDimension'],1)),data_original[x])
                        xmu=xmu + dot(gamma_times_data,(multiply(tile(array([mult_const]).T,(1,model['observationDimension'])),tile(mu.T,(T,1)))))
                    sigma=(expected['cov_matrix'][:,:,i] + dot(dot(expected['state_frequency_const'][i],matrix(mu).T),matrix(mu)) - xmu - xmu.T) / expected['state_frequency'][i]
                    if model['prior']['use'] == 1:
                        c1=dot(model['prior']['k0'],expected['state_frequency'][i]) / (model['prior']['k0'] + expected['state_frequency'][i])
                        sigma=model['prior']['Psi'] + sigma + dot(dot(c1,(model['prior']['mu0'].T - mu)).T,(model['prior']['mu0'].T - mu))
                        mu=(dot(model['prior']['k0'],model['prior']['mu0']).T + dot(expected['state_frequency'][i],mu)) / (model['prior']['k0'] + expected['state_frequency'][i])
                    model['mu'][i]=mu[0]
                    model['sigma'][i]=dot(0.5,(sigma + sigma.T))
                    
                    sig_inv_mu=linalg.solve((model['sigma'][i]),model['mu'][i])
                    mu_sig_inv_mu=dot(mu[0],sig_inv_mu)
                    dota0=dot(expected['cov_matrix_cross'][:,:,i],sig_inv_mu)
                    dota1=reshape(dota0,(len(dota0),1))
                    dota2=reshape(expected['mean_vector_input'][:,i],(len(expected['mean_vector_input'][:,i]),1))
                    dota3=dot(expected['cov_matrix_cross_lag_2'][:,:,i],model['bparam']['bp'+str(i)])
                    model['aparam']['m'+str(i)]=linalg.lstsq((expected['cov_matrix_input'][:,:,i]),(dot(dota1,(1.0 / mu_sig_inv_mu)) - dota2 - dota3))[0]
                    
                    dotb1=dot(expected['cov_matrix_lag'][:,:,i],reshape(sig_inv_mu,(len(sig_inv_mu),1)))
                    dotb2=reshape(expected['mean_vector_lag'][:,i],(len(expected['mean_vector_lag'][:,i]),1))
                    dotb3=dot(expected['cov_matrix_cross_lag'][:,:,i],reshape(model['aparam']['m'+str(i)],(len(model['aparam']['m'+str(i)]),1)))
                    dotb4=reshape(dotb3,(len(dotb3),1))
                    model['bparam']['bp'+str(i)]=linalg.lstsq((expected['cov_matrix_self_lag'][:,:,i]),(dot(dotb1,(1.0 / mu_sig_inv_mu))-dotb2-dotb4))[0]
                    
        return model
        
    if __name__ == '__main__':
        pass
    #Expectation_maximization algorithm,la fonction definie l'etape Expectation
    
    def estep(self,gam,model,data,inputObs):
        expected={}
        N=len(gam)
        expected['state_frequency'] = [0]*model['nstates']
        expected['state_frequency_const']= [0]*model['nstates']
        for i in arange(0,N):
            lgamma=gam[i]
            T=len(lgamma[0])
            data_sample=data[i]
            mult_const=zeros((model['nstates'],T))
            for j in arange(0,model['nstates']):
                model['a'] = model['aparam']['m'+str(j)]
                model['b'] = model['bparam']['bp'+str(j)]
                modela=model['a']
                for s in arange(1,T):
                    modela=column_stack((modela,model['a']))
                temp=[.0]
                temp.extend(matrix(sum(multiply(tile(reshape(model['b'],(len(model['b']),1)),(1,T- 1)),data_sample[:,0:T-1]),0)).tolist()[0])
                mult_const[j,:]=(1.0 + sum(multiply(modela,inputObs[i]),0)) + temp
            mult_const=mult_const ** 2
            expected['state_frequency_const'] = add(expected['state_frequency_const'],sum(multiply(lgamma,mult_const),1))
            expected['state_frequency'] = add(expected['state_frequency'],sum(lgamma,1))
        if model['type']=='gauss':
            expected['mean_vector'] = zeros((model['observationDimension'],model['nstates']))
            expected['mean_vector_input'] = zeros((model['inputDimension'],model['nstates']))
            expected['cov_matrix'] = zeros((model['observationDimension'],model['observationDimension'],model['nstates']))
            expected['cov_matrix_input'] = zeros((model['inputDimension'],model['inputDimension'],model['nstates']))
            expected['cov_matrix_cross'] = zeros((model['inputDimension'],model['observationDimension'],model['nstates']))
            expected['mean_vector_lag'] =zeros((model['observationDimension'],model['nstates']))
            expected['cov_matrix_cross_lag'] =zeros((model['observationDimension'],model['inputDimension'],model['nstates']))
            expected['cov_matrix_lag'] = zeros((model['observationDimension'],model['observationDimension'],model['nstates']))
            expected['cov_matrix_self_lag'] = zeros((model['observationDimension'],model['observationDimension'],model['nstates']))
            expected['cov_matrix_cross_lag_2'] = zeros((model['inputDimension'],model['observationDimension'],model['nstates']))
            for i in arange(0,N):
                lgamma=gam[i]
                T=len(lgamma[0])
                data_sample=data[i]
                input_sample=inputObs[i]
                for j in arange(0,model['nstates']):
                    model['a'] = model['aparam']['m'+str(j)]
                    model['b'] = model['bparam']['bp'+str(j)]
                    modela=model['a']
                    for s in arange(1,T):
                        modela=column_stack((modela,model['a']))
                    temp=[.0]
                    temp.extend(sum(multiply(tile(reshape(model['b'],(len(model['b']),1)),(1,T- 1)),data_sample[:,0:T-1]),0))
                    mult_const=(1.0 + sum(multiply(modela,inputObs[i]),0)) + temp
                    gamma_times_data_const=multiply(multiply(tile(mult_const,(model['observationDimension'],1)),tile(lgamma[j,:],(model['observationDimension'],1))),data_sample)
                    gamma_times_data=multiply(tile(lgamma[j,:],(model['observationDimension'],1)),data_sample)
                    gamma_times_input_data=multiply(tile(lgamma[j,:],(model['inputDimension'],1)),input_sample)
                    gamma_times_data_lag=multiply(tile(lgamma[j,1:-1],(model['observationDimension'],1)),data_sample[:,0:-2])
                    expected['mean_vector_lag'][:,j]=expected['mean_vector_lag'][:,j] + sum(gamma_times_data_lag,1)
                    expected['cov_matrix_cross_lag'][:,:,j]=expected['cov_matrix_cross_lag'][:,:,j] + dot(gamma_times_data_lag,input_sample[:,1:-1].T)
                    expected['cov_matrix_lag'][:,:,j]=expected['cov_matrix_lag'][:,:,j] + dot(gamma_times_data_lag,data_sample[:,1:-1].T)
                    expected['cov_matrix_self_lag'][:,:,j]=expected['cov_matrix_self_lag'][:,:,j] + dot(gamma_times_data_lag,data_sample[:,0:-2].T)
                    expected['cov_matrix_cross_lag_2'][:,:,j]=expected['cov_matrix_cross_lag_2'][:,:,j] + dot(gamma_times_input_data[:,1:-1],data_sample[:,0:-2].T)
                    expected['mean_vector'][:,j]=expected['mean_vector'][:,j] + sum(gamma_times_data_const,1)
                    expected['mean_vector_input'][:,j]=expected['mean_vector_input'][:,j] + sum(gamma_times_input_data,1)
                    expected['cov_matrix'][:,:,j]=expected['cov_matrix'][:,:,j] + dot(gamma_times_data,data[i].T)
                    expected['cov_matrix_input'][:,:,j]=expected['cov_matrix_input'][:,:,j] + dot(gamma_times_input_data,input_sample.T)
                    expected['cov_matrix_cross'][:,:,j]=expected['cov_matrix_cross'][:,:,j] + dot(gamma_times_input_data,data_sample.T)
        else:
            if model['type']=='discrete':
                expected['observation_count'] = zeros((model['nstates'],model['ostates']))
                for i in arange(0,N):
                    lgamma=gam[i]
                    for j in arange(0,model['ostates']):
                        expected['observation_count'][:,j]=expected['observation_count'][:,j] + sum(lgamma[:,nonzero(array(data[i]) == j)[0].tolist()],1) 
        
        return expected
        
    if __name__ == '__main__':
        pass   
    #The forward_backward algorithm pour calculer la probabilite d'une sequence observee dans le contexte des modeles de Markov caches.

    def FBWpass(self,model,evidence,inputObs):
        alph,loglikelihood,mult_const=self.ForwardPass(model,evidence,inputObs)
        bet=self.BackwardPass(model,evidence,mult_const,inputObs)
        gam=[None] *len(evidence)
        xi=[None] *len(evidence)
        for i in arange(0,len(evidence)):
            lgam=multiply(alph[i],bet[i])
            lgam=lgam / tile(mult_const[i],(model['nstates'],1))
            gam[i]=lgam
        for i in arange(0,len(evidence)):
            levidence=evidence[i]
            linputObs=inputObs[i]
            xi_sample=zeros((model['nstates'],model['nstates'],len(levidence[0]) - 1))
            for j in arange(0,len(levidence[0]) - 1):
                A=self.transitionMatrix(model['W'],linputObs[:,j + 1],model)
                xi_sample[:,:,j]=multiply(A,(multiply(matrix(alph[i][:,j]).T,matrix(multiply(levidence[:,j + 1],bet[i][:,j + 1])))))
                val=sum(sum(xi_sample[:,:,j]))
            xi[i]=xi_sample
        return alph,bet,gam,xi,loglikelihood
        
    if __name__ == '__main__':
        pass
    #Calcul progressif des probabilites 
    
    def ForwardPass(self,model,evidence,inputObs):
        alph=[None] *len(evidence)
        loglikelihood=[None] *len(evidence)
        mult_const=[None] *len(evidence)
        for i in arange(0,len(evidence)):
            levidence=evidence[i]
            linputObs=inputObs[i]
            alpha_sample=zeros((model['nstates'],len(levidence[0])))
            const=zeros((1,len(levidence[0])))[0].tolist()
            starting_prob=self.multiClassProbability(model['piW'],linputObs[:,0])
            alpha_sample[:,0],Z=self.normalize_local(multiply(levidence[:,0],starting_prob))
            Zlog=log(Z)
            const[0]=1.0 / Z
            for j in arange(1,len(levidence[0])):
                A=self.transitionMatrix(model['W'],linputObs[:,j],model)
                alpha_sample[:,j],Z=self.normalize_local(multiply(levidence[:,j],(dot(A.T,alpha_sample[:,j - 1]))))
                Zlog=Zlog + log(Z)
                const[j]=1.0 / Z
            
            alph[i]=alpha_sample
            loglikelihood[i]=Zlog
            mult_const[i]=const
        return alph,loglikelihood,mult_const
        
    if __name__ == '__main__':
        pass
    #loglikelihood pour maximiser la fonction 

    def iomodelDataLoglikelihood(self,model,evidence,inputObs):
        alph=[None] *len(evidence)
        loglikelihood=[None] *len(evidence)
        for i in arange(0,len(evidence)):
            levidence=evidence[i]
            linputObs=inputObs[i]
            alpha_sample=zeros((int(model['nstates']),len(levidence[0])))
            starting_prob=self.multiClassProbability(model['piW'],linputObs[:,0])
            alpha_sample[:,0],Z=self.normalize_local(multiply(levidence[:,0],starting_prob))
            Zlog=log(Z)
            for j in arange(1,len(levidence[0])):
                A=self.transitionMatrix(model['W'],linputObs[:,j],model)
                alpha_sample[:,j],Z=self.normalize_local(multiply(levidence[:,j],(dot(A.T,alpha_sample[:,j - 1]))))
                Zlog=Zlog + log(Z)
            
            alph[i]=alpha_sample
            loglikelihood[i]=Zlog
        return loglikelihood
        
    if __name__ == '__main__':
        pass
    #Calcul retrogressif des probabilites

    def BackwardPass(self,model,evidence,mult_const,inputObs):
        bet=[None] *len(evidence)
        for i in arange(0,len(evidence)):
            levidence=evidence[i]
            linputObs=inputObs[i]
            const=mult_const[i]
            beta_sample=zeros((model['nstates'],len(levidence[0])))
            beta_sample[:,-1]=1
            beta_sample[:,-1]=dot(const[-1],beta_sample[:,-1])
            for j in xrange(len(levidence[0]) - 2,-1,-1):
                A=self.transitionMatrix(model['W'],linputObs[:,j + 1],model)
                beta_sample[:,j]=dot(A,(multiply(levidence[:,j + 1],beta_sample[:,j + 1])))
                
                beta_sample[:,j]=dot(const[j],beta_sample[:,j])
            bet[i]=beta_sample
        return bet
        
    if __name__ == '__main__':
        pass

    #assigning each of the observations into one of the classes

    def multiClassProbability(self,W,U):
        
        potentials=exp(dot(W,U)).tolist()
        potentials.extend([1.0])
        Z=sum(potentials)
        probability=potentials / Z
        return probability
        
    if __name__ == '__main__':
        pass
    # the probability of transitioning from state i to state j
    
    def transitionMatrix(self,W,U,model):
        A=zeros((int(model['nstates']),int(model['nstates'])))
        for i in arange(0,model['nstates']):
            probability=self.multiClassProbability(W[i,:,:],U)
            A[i,:]=probability
        
        return A
        
    if __name__ == '__main__':
        pass
    #normaliser un vecteur( diviser chaque element par la somme du vecteur)
   
    def normalize_local(self,v):
        Z=sum(v)
        v=v / Z
        return v,Z
        
    if __name__ == '__main__':
        pass
    
    def completeObj(self,w,data):
        grad=self.gradient(w,data)
        obj=self.objective(w,data)
        return obj,grad
        
    if __name__ == '__main__':
        pass
 
    def completeObj0(self,w,data):
        grad=self.gradient0(w,data)
        obj=self.objective0(w,data)
        return obj,grad
        
    if __name__ == '__main__':
        pass

    #gradient of the objective function
    def gradient(self,w,data):
        grad=zeros((len(w),1))
        inputObs=data['inputObs']
        xi=data['xi']
        gam=data['gam']
        state=data['state']
        nstates=data['nstates']
        inputDimension=data['inputDimension']
        for i in arange(0,len(inputObs)):
            linputObs=inputObs[i]
            lxi=xi[i]
            lgamma=gam[i]
            for j in arange(0,len(linputObs[0]) - 1):
                probability=self.multiClassProbability(reshape(w,(inputDimension,nstates - 1)).T,linputObs[:,j + 1])
                lxi_=lxi[:,:,j]
                '''assert (sum(lxi_[state,0:]) - lgamma[state,j]) < 1e-08, '> 1e-08' '''
                grad=grad + kron(lxi_[state,0:-1].T - dot(lgamma[state,j],probability[0:-1]),linputObs[:,j + 1])
        
        grad=dot(- 1.0,grad)
        return grad
        
    if __name__ == '__main__':
        pass

    #return la fonction objective pour l'utiliser dans l'etape maximisation de EM algorithm
    def objective(self,w,data):
        obj=0.0
        inputObs=data['inputObs']
        xi=data['xi']
        state=data['state']
        nstates=data['nstates']
        inputDimension=data['inputDimension']
        for i in arange(0,len(inputObs)):
            linputObs=inputObs[i]
            lxi=xi[i]
            for j in arange(0,len(linputObs[0]) - 1):
                probability=self.multiClassProbability(reshape(w,(inputDimension,nstates - 1)).T,linputObs[:,j + 1])
                lxi_=lxi[:,:,j]
                obj=obj + sum(multiply(lxi_[state,:].T,log(probability)))
        
        obj=dot(- 1.0,obj)
        return obj
        
    if __name__ == '__main__':
        pass
    
    def gradient0(self,w,data):
        grad=zeros((len(w),1))
        inputObs=data['inputObs']
        gam=data['gam']
        nstates=data['nstates']
        inputDimension=data['inputDimension']
        for i in arange(0,len(inputObs)):
            linputObs=inputObs[i]
            lgamma=gam[i]
            probability=self.multiClassProbability(reshape(w,(inputDimension,nstates - 1)).T,linputObs[:,0])
            grad=grad + kron(lgamma[0:-2,1] - probability[0:-1 - 1],linputObs[:,0])
        
        grad=dot(- 1.0,grad)
        return grad
        
    if __name__ == '__main__':
        pass

    def objective0(self,w,data):
        obj=0.0
        inputObs=data['inputObs']
        gam=data['gam']
        nstates=data['nstates']
        inputDimension=data['inputDimension']
        for i in arange(0,len(inputObs)):
            linputObs=inputObs[i]
            lgamma=gam[i]
            probability=self.multiClassProbability(reshape(w,(inputDimension,nstates - 1)).T,array(linputObs)[:,0])
            obj=obj + sum(multiply(lgamma[:,1],log(probability)))
        
        obj=dot(- 1.0,obj)
        return obj
        
    if __name__ == '__main__':
        pass
  
    #calcul de l'evidence

    def aiocalculateEvidence(self,model,data,inputObs):
        
        evidence=[]
        if model['type']=='gauss':
            for i in arange(0,len(data)):
                data_sample=data[i]
                T=len(data_sample[0])
                evidence_sample=zeros((int(model['nstates']),T))
                for j in arange(0,model['nstates']):
                    if 'aparam' in model:
                        model['a'] = model['aparam']['m'+str(j)]
                        model['b'] = model['bparam']['bp'+str(j)]
                    else:
                        if 'a' not in model:
                            a=[0.5]
                            a.extend([- 0.5])
                            a.extend([1.0])
                            a.extend([0])
                            model['a'] =a
                    
                    modela=model['a']
                    for s in arange(1,T):
                        modela=column_stack((modela,model['a']))
                    modelb=model['b']
                    for s in arange(1,T-1):
                        modelb=column_stack((modelb,model['b']))
                    extnd=[.0]
                    extnd.extend(sum(multiply(modelb,data_sample[:,0:T-1]).tolist(),0).tolist())
                    mult_const=(1.0 + sum(multiply(modela,inputObs[i]),0) +extnd)
                    mu=model['mu'][j]
                    mult_const_=mult_const
                    for s in arange(1,int(model['observationDimension'])):
                        mult_const_=column_stack((mult_const_,mult_const))
                    
                    MU=multiply(mult_const_,tile(mu.T,(T,1)))
                    sigma=model['sigma'][j]
                    probability=[]
                    '''print('cov')
                        print(cov(data[i].T, rowvar=False))'''
                    min_eig = min(real(linalg.eigvals(sigma)))
                    if min_eig < 0:
                        sigma =sigma- 10*min_eig * eye(*sigma.shape)

                    for z in arange(0,len(data[i].T)):
                        muu=[multivariate_normal.pdf(data[i].T[z],MU[z],sigma,allow_singular=True)]
                        probability.extend(muu)
                    evidence_sample[j,:]=probability
                evidence.append(evidence_sample)
        else:
            if model['type']=='discrete':
                for i in arange(0,len(data)):
                    evidence.append(model['B'](arange(),data[i]))
        return evidence
        
    if __name__ == '__main__':
        pass
    
    #initialiser le modele de markov cache

    def initializeHMMmodel(self,train_data,type_,nstates,ostates,iotype,istates):
        model={}
        model['nstates'] =nstates
        model['type'] = type_
        model['iotype'] = iotype
        if type_=='discrete':
            model['ostates'] = ostates
            model['pi'] =(1./nstates)*ones((nstates,1))
            model['A'] = random.random((nstates,nstates))
            model['A'] = model['A'] / tile(sum(model['A'],axis=1),nstates).reshape((nstates,nstates)).T
            model['B'] = (1./ostates)*ones((ostates,1))
        else:
            if type_=='gauss' and logical_not(iotype):
                model['observationDimension'] = ostates
                model['pi'] =(1./nstates)*ones((nstates,1))
                model['A'] = random.random((nstates,nstates))
                model['A'] = model['A'] / tile(sum(model['A'],axis=1),nstates).reshape((nstates,nstates)).T
                mu={}
                sigma={}
                for i in arange(0,model['nstates']):
                    mu['m'+str(i)]=random.random((model['observationDimension'],1))
                    model['mu']=mu
                    sigma['s'+str(i)]=dot((1.0 / model['observationDimension']),eye(model['observationDimension'],model['observationDimension']))
                    model['sigma']=sigma
            else:
                if type_=='gauss' and iotype:
                    model['observationDimension'] = ostates
                    model['inputDimension'] = istates
                    model['piW'] = random.random((nstates - 1,istates))
                    model['W'] = random.random((nstates,nstates - 1,istates))
                    mu={}
                    sigma={}
                    aparam={}
                    a=[0.5]
                    a.extend([- 0.5])
                    a.extend([1.0])
                    a.extend([0])
                    for i in arange(0,model['nstates']):
                        aparam['m'+str(i)]=a
                        model['aparam']=aparam
                    mu,sigma=self.getGMMinitialization(train_data,nstates)
                    model['mu'] = mu
                    model['sigma'] = sigma
        
        return model
        
    if __name__ == '__main__':
        pass


    #Gaussian mixture model sert usuellement a estimer parametriquement la distribution de variables aleatoires 
    #Il s'agit alors de determiner la variance, la moyenne  de chaque gaussienne.
    #Ces parametres sont optimises selon un critere de maximum de vraisemblance pour approcher le plus possible la distribution recherchee.
    #Cette procedure se faititerativement via l'algorithme esperance-maximisation (EM).
    def getGMMinitialization(self,train_data,nstates):
        data_agg=train_data[0]
        count=0
        for i in arange(1,len(train_data)):
            data_agg=column_stack((data_agg,train_data[i]))
               

        data_agg=data_agg.T
        ftr_dim=len(data_agg[0])
        const_ftr=[]
        for i in arange(0,ftr_dim):
            std_dev=std(data_agg.T[i],ddof=1)
            if (std_dev < 0.01):
                const_ftr.append(i)
        for i in arange(0,len(const_ftr)):
            ftr_ind=const_ftr[i]
            data_agg.T[ftr_ind]=data_agg.T[ftr_ind] + 0.001*random.randn((len(data_agg)))
        nMixture = 3
        gmm = GaussianMixture(nMixture,max_iter=1000,reg_covar=0.1)
        gmm.fit(data_agg)
        mu=gmm.means_
        sigma_=gmm.covariances_
        return mu,sigma_
        
    if __name__ == '__main__':
        pass










