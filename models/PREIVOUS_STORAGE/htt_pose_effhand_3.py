import torch
import torch.nn.functional as torch_f
import cv2
from einops import repeat
import mediapipe as mp
import numpy as np
from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries 
import clip     # CLIP 따로 설치해야함
import os
from mmpose.apis import init_model
from mmpose.structures import merge_data_samples
from mmpose.apis.inference import inference_topdown
from mmpose.apis import init_model
from mmpose.utils import register_all_modules
import numpy as np
import torch
import cv2
import mmcv
register_all_modules()
import matplotlib.pyplot as plt  

class ResNet_(torch.nn.Module):
    def __init__(self,resnet_version=18):
        super().__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            self.base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            self.base_net = resnet.resnet50(pretrained=True)
        else:
            self.base_net=None
    
    
    def forward(self, image):
        features, res_layer5 = self.base_net(image)
        return features, res_layer5

# def preprocess_tensor_image(tensor_img: torch.Tensor):
#     """
#     Convert [C, H, W] torch tensor to numpy image (RGB, uint8)
#     """
#     img_np = tensor_img.detach().cpu().numpy()
#     img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
#     img_np = (img_np * 255).astype(np.uint8)
#     return img_np

# def load_hand_keypoints_from_csv(path: str, image_shape: tuple, original_shape=(848, 480)) -> torch.Tensor:
#     """
#     주어진 CSV 파일에서 손 keypoint를 불러오고 42x2 tensor로 padding함.
#     좌표는 원본 해상도 → 현재 학습 해상도 비율로 조정됨.

#     Args:
#         path (str): CSV 파일 경로
#         image_shape (tuple): 현재 이미지 해상도 (H, W)
#         original_shape (tuple): 원본 해상도 (W, H)

#     Returns:
#         torch.Tensor: shape (42, 2)
#     """
#     keypoints = []
#     try:
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 x = float(row['x'])
#                 y = float(row['y'])
#                 keypoints.append((x, y))
#     except Exception as e:
#         print(f"[ERROR] {path} load 실패: {e}")
#         keypoints = []

#     keypoints_tensor = torch.zeros((42, 2), dtype=torch.float32)
#     n = min(len(keypoints), 42)
#     if n > 0:
#         kp_tensor = torch.tensor(keypoints[:n])  # [N, 2]

#         # 원본 → 현재 이미지 사이즈로 스케일 조정
#         orig_w, orig_h = original_shape
#         curr_h, curr_w = image_shape
#         scale_x = curr_w / orig_w
#         scale_y = curr_h / orig_h
#         kp_tensor[:, 0] *= scale_x
#         kp_tensor[:, 1] *= scale_y

#         keypoints_tensor[:n] = kp_tensor

#     return keypoints_tensor

# def visualize_hand_keypoints(images, keypoints, save_dir="vis", prefix="sample"):
#     """
#     Args:
#         images: [B, 3, H, W] (tensor, normalized)
#         keypoints: [B, 42, 2] (tensor, image size 기준 좌표)
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     images = images.cpu()
#     keypoints = keypoints.cpu().numpy()

#     B, _, H, W = images.shape

#     for b in range(B):
#         img = images[b].numpy().transpose(1, 2, 0)  # [H, W, 3]
#         img = ((img * 1.0) + 0.5) * 255.0
#         img = np.clip(img, 0, 255).astype(np.uint8)
#         img = np.ascontiguousarray(img)  # ✅ OpenCV 호환을 위해 메모리 정렬

#         for i in range(keypoints.shape[1]):
#             x, y = keypoints[b, i]
#             if x > 0 and y > 0:
#                 cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

#         save_path = os.path.join(save_dir, f"{prefix}_{b}.jpg")
#         cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class TemporalNet(torch.nn.Module):
    def __init__(self,  is_single_hand,
                        transformer_d_model,
                        transformer_dropout,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_num_encoder_layers_action,
                        transformer_num_encoder_layers_pose,
                        transformer_normalize_before=True,

                        lambda_action_loss=None,
                        lambda_hand_2d=None,
                        lambda_hand_z=None,
                        ntokens_pose=1,
                        ntokens_action=1,
                        
                        dataset_info=None,
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        dim_grasping_feature=128,):

        super().__init__()
        
        self.ntokens_pose= ntokens_pose
        self.ntokens_action=ntokens_action

        self.pose_loss=loss_str2func()[pose_loss]
        
        self.lambda_hand_z=lambda_hand_z
        self.lambda_hand_2d=lambda_hand_2d        
        self.lambda_action_loss=lambda_action_loss
        self.lambda_handtype_loss=1.0

        # Hyperparameters
        self.lambda_hand_z=100 # 100
        self.lambda_hand_2d=1 # 1    
        self.lambda_action_loss=1 # 1

        self.is_single_hand=is_single_hand
        self.num_joints=21 if self.is_single_hand else 42

        
        # Feature Extraction
        self.meshregnet = ResNet_(resnet_version=18)

        # Egocentric Knowledge Module (Local Transformer)
        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model) 

        self.transformer_pose=Transformer_Encoder(d_model=transformer_d_model, 
                                nhead=transformer_nhead, 
                                num_encoder_layers=transformer_num_encoder_layers_pose,
                                dim_feedforward=transformer_dim_feedforward,
                                dropout=0.0, 
                                activation="relu", 
                                normalize_before=transformer_normalize_before)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Hand Pose Estimation
        # self.scale_factor = scale_factor 
        # self.trans_factor = trans_factor
        # self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*3,
        #                         act_hidden='leakyrelu',act_final='none')        
        # self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)
        
        # Object classification
        # self.num_objects=dataset_info.num_objects
        self.num_objects=63
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(42*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        # print(transformer_d_model)
        # self.concat_to_action_input=torch.nn.Linear(transformer_d_model*3,transformer_d_model)

        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)
        # self.concat_to_action_input=torch.nn.Linear(transformer_d_model*1,transformer_d_model)
        self.num_actions=dataset_info.num_actions
        self.action_token=torch.nn.Parameter(torch.randn(1,1,transformer_d_model))
        
        self.transformer_action=Transformer_Encoder(d_model=transformer_d_model, 
                            nhead=transformer_nhead, 
                            num_encoder_layers=transformer_num_encoder_layers_action,
                            dim_feedforward=transformer_dim_feedforward,
                            dropout=0.0,
                            activation="relu", 
                            normalize_before=transformer_normalize_before) 
        
        self.action_classification= ActionClassificationBranch(num_actions=self.num_actions, action_feature_dim=transformer_d_model)

        # Hand Type Prior
        self.num_handtypes=dataset_info.num_handtypes
        self.hand_type_classification = HandTypeClassificationBranch(num_types=self.num_handtypes, hand_feature_dim=transformer_d_model)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        # self.hand_pred_label_txt = ['No hand', 'large diameter', 'medium wrap', 'adducted thumb', 'prismatic finger', 'pinch', 'precision disc', 'tripod', 'fixed hook', 'index finger', 'extension type', 'writing tripod', 'parallel extension', 'abduction grip', 'lateral tripod', 'quadpod', 'stick', 'pincer', 'flattened palm', 'thumb in', 'curl index finger', 'spray', 'fist', 'bend index finger']
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)

        self.dataset_name = dataset_info.name

    def forward(self, batch_flatten, epoch=0, train=True, verbose=True):
        flatten_images, keypoints_csv = batch_flatten[TransQueries.IMAGE]
        flatten_images = flatten_images.cuda()
        keypoints_csv = keypoints_csv.cuda()
        
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images) 
        
        # ========= handkey extract Module ======
        B, C, H, W = flatten_images.shape

        # ======== Rescale keypoints to current image size ========
        original_w, original_h = 848, 480       # 이미지 원본
        scale_x = W / original_w
        scale_y = H / original_h

        # x, y 좌표만 사용, z는 버림
        
        # print("!!!!", keypoints_csv.shape)
        keypoints_tensor = keypoints_csv[:, :, :2].clone()  # [B, 42, 2]
        keypoints_tensor[:, :, 0] *= scale_x
        keypoints_tensor[:, :, 1] *= scale_y
        
        # (optional) 시각화
        # visualize_hand_keypoints(flatten_images, keypoints_tensor, save_dir="vis/hand_keypoints")

        # ======== Flatten 2D Keypoints and Feed into MLP ========
        mp_keypoints_tensor = keypoints_tensor.reshape(B, -1)  # [B, 42*2]
        # flatten_mp_feature = self.hand_pose3d_to_action_input(mp_keypoints_tensor).detach() #detach 되면 안되지 않나
        flatten_mp_feature = self.hand_pose3d_to_action_input(mp_keypoints_tensor)


        # ======== Egocentric Knowledge Module ========
        batch_seq_pin_feature=flatten_in_feature.contiguous().view(-1,self.ntokens_pose,flatten_in_feature.shape[-1])
        batch_seq_pin_pe=self.transformer_pe(batch_seq_pin_feature)
         
        batch_seq_pweights=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()
         
        batch_seq_pout_feature,_=self.transformer_pose(src=batch_seq_pin_feature, src_pos=batch_seq_pin_pe,
                            key_padding_mask=batch_seq_pmasks, verbose=False)
 
 
        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
        
        # # Hand Pose Estimation
        # flatten_hpose=self.image_to_hand_pose(flatten_pout_feature)
        # flatten_hpose=flatten_hpose.view(-1,self.num_joints,3)
        # flatten_hpose_25d_3d=self.postprocess_hand_pose(sample=batch_flatten,scaletrans=flatten_hpose,verbose=verbose) 

        # weights_hand_loss=batch_flatten['not_padding'].cuda().float()
        # hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,flatten_hpose_25d_3d=flatten_hpose_25d_3d,weights=weights_hand_loss,
        #                 total_loss=total_loss,verbose=verbose)        
        # results.update(hand_results)
        # losses.update(hand_losses)

        # Object Classification
        flatten_olabel_feature = self.image_to_olabel_embed(flatten_pout_feature)
        weights_olabel_loss=batch_flatten['not_padding'].cuda().float()
        olabel_results,total_loss,olabel_losses=self.predict_object(sample=batch_flatten,features=flatten_olabel_feature,
                        weights=weights_olabel_loss,total_loss=total_loss,verbose=verbose)
        results.update(olabel_results)
        losses.update(olabel_losses)

        # Hand Type Classification
        weights_hlabel_loss=batch_flatten['not_padding'].cuda().float()
        if self.dataset_name == 'h2o':
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype_h2o(
                sample=batch_flatten, 
                features=flatten_in_feature,
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        else:
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype(
                sample=batch_flatten, 
                features=flatten_in_feature,
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        results.update(hlabel_results)
        losses.update(hlabel_losses)
    
        # ======== Egocentric Action Module ========
        # flatten_hpose2d=torch.flatten(flatten_hpose[:,:,:2],1,2)
        # flatten_ain_feature_hpose=self.hand_pose3d_to_action_input(flatten_hpose2d) # flatten_ain_feature_hpose shape : (B * 128, 512)
        flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"]) # flatten_ain_feature_olabel shape : (B * 128, 512)
        
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt=torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        
        # flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_hpose,flatten_ain_feature_olabel),dim=1) # (B * 128, 1536)
        # print("flatten_pout_feature", flatten_pout_feature.shape)
        # print("flatten_ain_feature_olabel", flatten_ain_feature_olabel.shape)

        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_olabel),dim=1) # (B * 128, 512*2=1024) => object랑  concat        
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) # (B * 128, 512)

        #Concat handkeypoint flatten_mp_feature
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_mp_feature), dim=1) 
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) 


        # if train:
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1536)
        flatten_ain_feature=self.hlabel_concat_to_action_input(flatten_ain_feature)
        batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])
        
        # Concat trainable token
        batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
 
        batch_seq_weights_action=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_action)
        batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool() 
        batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)        
         
        batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
                                key_padding_mask=batch_seq_amasks, verbose=False)
        
        # Action Classification
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)     
        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 

        action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=batch_out_action_feature, weights=weights_action_loss,
                        total_loss=total_loss,verbose=verbose)
        # print("[Debug] pred & gt=>", action_results['action_pred_labels'], action_results['action_gt_labels'])

        results.update(action_results)
        losses.update(action_losses)
        if verbose:
            log_path = "./exp_loss_log/loss_log.txt"
            self.log_losses_to_file(log_path, losses, epoch)


        return total_loss, results, losses
    

    def predict_object(self, sample, features, weights, total_loss, verbose=False):
        olabel_feature = features
        out = self.obj_classification(olabel_feature)

        olabel_results, olabel_losses = {}, {}

        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        # print(obj_idx_list)
        num_classes = 63
        batch_size = 256

        # multi-hot label 생성
        olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)
        # print(enumerate(obj_idx_list))
        
        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                olabel_gts[i, obj_id] = 1.0     # one hot encoding
                
        logits = out["reg_outs"]                # shape: [B, C]

        olabel_results["obj_gt_labels"] = olabel_gts
        olabel_results["obj_pred_labels"] = (logits > 0).float()
        olabel_results["obj_reg_possibilities"] = torch.sigmoid(logits)

        # BCE loss 사용 (object multi label classification)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        olabel_loss = bce_loss_fn(logits, olabel_gts)  # [B, C]
        olabel_loss = torch.sum(olabel_loss, dim=1)  # sum over classes per sample
        olabel_loss = torch.mul(olabel_loss, weights.flatten())
        olabel_loss = torch.sum(olabel_loss) / torch.sum(weights)

        if total_loss is None:
            total_loss = self.lambda_action_loss * olabel_loss
        else:
            total_loss += self.lambda_action_loss * olabel_loss
            olabel_losses["olabel_loss"] = olabel_loss

        return olabel_results, total_loss, olabel_losses

    def predict_handtype(self, sample, features, weights, total_loss, verbose=False):   # h2o용(양손데이터)
        hand_feature = features
        out=self.hand_type_classification(hand_feature)
        
        hlabel_results, hlabel_losses={},{}
        hlabel_gts_left = sample['hand_label_left'].cuda()
        hlabel_gts_right = sample['hand_label_right'].cuda()

        # =========== Focus on Right hand ===========
        hlabel_gts = hlabel_gts_right

        hlabel_results["hand_gt_labels"]=hlabel_gts
        hlabel_results["hand_pred_labels"]=out["pred_labels"]
        hlabel_results["hand_reg_possibilities"]=out["reg_possibilities"]

        hlabel_loss = torch_f.cross_entropy(out["reg_outs"],hlabel_gts,reduction='none')
        hlabel_loss = torch.mul(torch.flatten(hlabel_loss),torch.flatten(weights))
            
        hlabel_loss=torch.sum(hlabel_loss)/torch.sum(weights)

        if total_loss is None:
            total_loss=self.lambda_handtype_loss*hlabel_loss
        else:
            total_loss+=self.lambda_handtype_loss*hlabel_loss
            hlabel_losses["hlabel_loss"]=hlabel_loss
        return hlabel_results, total_loss, hlabel_losses


    def predict_action(self,sample,features,weights,total_loss=None,verbose=False):
        action_feature=features
        out=self.action_classification(action_feature)
        
        action_results, action_losses={},{}
        action_gt_labels=sample[BaseQueries.ACTIONIDX].cuda()[0::self.ntokens_action].clone()
        action_results["action_gt_labels"]=action_gt_labels
        action_results["action_pred_labels"]=out["pred_labels"]
 
        action_results["action_reg_possibilities"]=out["reg_possibilities"]
        action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')  
        action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights)) 
        action_loss=torch.sum(action_loss)/torch.sum(weights) 

        if total_loss is None:
            total_loss=self.lambda_action_loss*action_loss
        else:
            total_loss+=self.lambda_action_loss*action_loss
        action_losses["action_loss"]=action_loss

        return action_results, total_loss, action_losses

    def log_losses_to_file(self, log_path, losses, epoch):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n--- Epoch {epoch} ---\n")
            for k, v in losses.items():
                f.write(f"{k}: {v.item():.6f}\n")
   



