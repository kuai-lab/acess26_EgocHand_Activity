import torch
import torch.nn.functional as torch_f
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

import scipy.ndimage
import sys
sys.path.insert(0, '')
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from fvcore.nn.flop_count import flop_count
from tqdm import tqdm
import time

from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.objclassbranch import ObjClassBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries 
import clip     # CLIP 따로 설치해야함
from HandFormer.HandFormer.models.ms_tcn_1D import MultiScale_TemporalConv as MS_TCN
from HandFormer.HandFormer.models.microaction_encoder_HTT import MicroactionEncoder_HTT
from HandFormer.HandFormer.models.hf_pose_motion_FINAL_HTT import HF_Pose      



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
 

class TemporalNet(torch.nn.Module):
    def __init__(self,  is_single_hand,
                        transformer_d_model,
                        transformer_dropout,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_num_encoder_layers_action,
                        transformer_num_encoder_layers_pose,
                        transformer_normalize_before=True,

                        num_classes =19 ,
                        lambda_action_loss=None,
                        lambda_hand_2d=None,
                        lambda_hand_z=None,

                        ntokens_pose=16,
                        ntokens_action=128,

                        embedding_dim_final=256,
                        dataset_info=None,
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        dim_grasping_feature=128):

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
                                    
        # Object classification
        self.num_actions=dataset_info.num_actions
        self.num_objects=63
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ObjClassBranch(num_obj=self.num_objects, feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*2 ,transformer_d_model)
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2, transformer_d_model)
        self.dataset_name = dataset_info.name

        # rotation_prior
        self.rotation_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )

        self.rotation_gate = torch.nn.Sequential(
            nn.Linear(transformer_d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid() #출력 0-1사이
        )

        self.final_fusion_layer=torch.nn.Linear(
            transformer_d_model+256, transformer_d_model
        )

        # Disable the transformer and classifier in the pose_net. Only microaction encoding will be used.
        self.lambda_contrastive_loss = 0.3
        num_action_classes=dataset_info.num_actions
        self.proxies=nn.Parameter(torch.randn(num_action_classes, 256))


    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        flatten_images = batch_flatten["rgb_image"].cuda()          # ([8, 3, 128, 270, 480])
        # flatten_images: [B, C, T, H, W]
        B, C, n_token_action, H, W = flatten_images.shape
        flatten_images = flatten_images.reshape(B*n_token_action, C, H, W)              # [B*T, C, H, W] 
        flatten_images = flatten_images.cuda()               # [1024, 3, 270, 480])

        pose_seq = batch_flatten["pose_keypoint"].cuda()     # [B, 3, 128, 21, 2])
        pose_seq = pose_seq.cuda()       
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {} ;results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images)
        rotation_features=self.compute_rotation_priors(
            pose_seq, sigma=3
        ) 

        rotation_features=rotation_features.cuda() # 2(B), 128(frames)
        rotation_embedding=self.rotation_encoder(rotation_features) #B, 256

        # ======== Egocentric Knowledge Module ========
        batch_seq_pin_feature=flatten_in_feature.contiguous().view(-1,self.ntokens_pose,flatten_in_feature.shape[-1])
        batch_seq_pin_pe=self.transformer_pe(batch_seq_pin_feature)
         
        batch_seq_pweights=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()
         
        batch_seq_pout_feature,_=self.transformer_pose(src=batch_seq_pin_feature, src_pos=batch_seq_pin_pe,
                            key_padding_mask=batch_seq_pmasks, verbose=False)
        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
    
        # Object Classification
        flatten_olabel_feature=self.image_to_olabel_embed(flatten_pout_feature)
        
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
        flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"]) # flatten_ain_feature_olabel shape : (B * 128, 512)
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt=torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_olabel),dim=1) # (B * 128, 512*2=1024) => object랑  concat        
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) # (B * 128, 512)
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1024)
        flatten_ain_feature=self.hlabel_concat_to_action_input(flatten_ain_feature) # b*128, 512


        seq_ain_feature = flatten_ain_feature.view(-1, n_token_action, flatten_ain_feature.shape[-1]) # b, 128, 512
        seq_summary=torch.mean(seq_ain_feature, dim=1)
        gate_summary=self.rotation_gate(seq_summary) #b, 512, rotation b, 256 
        gated_rotation_embedding = rotation_embedding*gate_summary # b, 256 랑 b, 256 해 -> b, 256 rotation_prior_token
        rotation_token = gated_rotation_embedding.unsqueeze(1).repeat(1, 128, 1)
        rotation_token = rotation_token.view(-1, 256)

        flatten_ain_feature=torch.cat((flatten_ain_feature, rotation_token), dim=1) #b*128, 512+256
        flatten_ain_feature=self.final_fusion_layer(flatten_ain_feature) #b*128, 512
        batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])

        # Concat trainable token
        batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
 
        batch_seq_weights_action=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_action)
        batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool() #2개->1개
        batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)        
         
        batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
                                key_padding_mask=batch_seq_amasks, verbose=False)
        
        # Action Classification
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)     
        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 

        action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=batch_out_action_feature, weights=weights_action_loss,
                        total_loss=total_loss,verbose=verbose)
        
        results.update(action_results)
        losses.update(action_losses)
        
        # --- ProxyNCA Loss 계산 ---
        if train:

            embeddings = torch_f.normalize(gated_rotation_embedding, p=2, dim=1)
            labels = action_results["action_gt_labels"]

            contrastive_loss = self._compute_proxynca_loss(embeddings, labels)

            total_loss += self.lambda_contrastive_loss * contrastive_loss
            losses["Proxynce_loss"] = contrastive_loss.detach()

    
        return total_loss, results, losses


#loss
    @staticmethod
    def compute_rotation_priors(pose_seq, sigma=3):
        """
        누적 회전량을 배치 단위로 계산
        Args:
            pose_seq: Tensor of shape (B, 3, T, V=21, M=2)
            sigma: smoothing sigma (for gaussian_filter1d)
        Returns:
            Tensor of shape (B, T) — 배치당 누적 회전량 시퀀스
        """
        import scipy.ndimage
        from scipy.spatial.transform import Rotation as ScipyRotation
        import numpy as np

        B, C, T, V, M = pose_seq.shape
        pose_seq = pose_seq.permute(0, 2, 3, 1, 4)[..., 1]  # → (B, T, V, C), 오른손만
        pose_seq_np = pose_seq.cpu().numpy()  # (B, T, V, 3)

        all_rotation_trends = []
        for b in range(B):
            keypoints_3d_seq = pose_seq_np[b]  # (T, V, 3)
            reference_grasp_points = None
            reference_palm_to_finger_vector = None

            signed_accumulated_angle_list = []
            signed_accum_angle = 0.0

            for t in range(T):
                kp = keypoints_3d_seq[t]
                palm_center = (kp[0] + kp[5] + kp[17]) / 3
                finger_center = (kp[4] + kp[8] + kp[12]) / 3
                palm_to_finger_vector = finger_center - palm_center

                current_grasp_points = np.array([kp[4], kp[8], kp[12]])

                if reference_grasp_points is None:
                    reference_grasp_points = current_grasp_points.copy()
                    reference_palm_to_finger_vector = palm_to_finger_vector.copy()
                    signed_accumulated_angle_list.append(0.0)
                    continue

                try:
                    delta_rot, _ = ScipyRotation.align_vectors(current_grasp_points, reference_grasp_points)
                    rotvec = delta_rot.as_rotvec()
                    angle_inc = np.linalg.norm(rotvec)
                except Exception:
                    angle_inc = 0.0
                    rotvec = np.zeros(3)

                sign = 0
                if angle_inc > 1e-6:
                    sign = np.sign(np.dot(rotvec, reference_palm_to_finger_vector))

                signed_delta = np.rad2deg(angle_inc) * sign
                signed_accum_angle += signed_delta
                signed_accumulated_angle_list.append(signed_accum_angle)

            smoothed = scipy.ndimage.gaussian_filter1d(signed_accumulated_angle_list, sigma=sigma)
            trend_tensor = torch.tensor(smoothed, dtype=torch.float32)
            all_rotation_trends.append(trend_tensor.unsqueeze(0))  # (1, T)

        return torch.cat(all_rotation_trends, dim=0) #.cuda()  # (B, T)


    def _compute_proxynca_loss(self, embeddings, labels):
        # 프록시와의 거리 계산 (유클리드 거리의 제곱)
        # self.proxies를 embeddings와 같은 디바이스로 이동
        proxies = self.proxies.to(embeddings.device) #num_claases, 256
        dists = torch.cdist(embeddings, proxies) ** 2

        # 자기 자신의 클래스 프록시(P_y)와의 거리는 따로 분리
        pos_dists = dists[torch.arange(len(labels)), labels]

        # 자기 자신 클래스를 제외한 나머지 프록시(P_z)와의 거리만 남김
        dists[torch.arange(len(labels)), labels] = float('inf')

        # 손실 계산 (논문 수식 기반)
        # logsumexp 트릭을 사용하여 수치적으로 안정적인 계산
        neg_log_probs = -torch.logsumexp(-dists, dim=1)

        # 최종 loss는 positive 거리와 negative log-probabilities의 합
        loss = (pos_dists + neg_log_probs).mean()
        return loss


    def predict_object(self,sample,features, weights, total_loss,verbose=False):
        olabel_feature=features
        out=self.obj_classification(olabel_feature)
        batch_size = features.shape[0] #2 X 128
        num_classes = self.num_objects
        olabel_results, olabel_losses={},{}

        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)

        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                olabel_gts[i, obj_id] = 1.0

        olabel_results["obj_gt_labels"]=olabel_gts
        olabel_results["obj_pred_labels"]=out["pred_labels"]
        olabel_results["obj_reg_possibilities"]=out["reg_possibilities"]

        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        olabel_loss = bce_loss_fn(out["reg_outs"], olabel_gts)  # [B, C]
        olabel_loss = torch.sum(olabel_loss, dim=1)  # sum over classes per sample
        olabel_loss = torch.mul(olabel_loss, weights.flatten())
        olabel_loss = torch.sum(olabel_loss) / torch.sum(weights)


        if total_loss is None:
            total_loss=self.lambda_action_loss*olabel_loss
        else:
            total_loss+=self.lambda_action_loss*olabel_loss
            olabel_losses["olabel_loss"]=olabel_loss
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
        print("pred_actions : ", out["pred_labels"])
        print("GT actions :", action_gt_labels)
        action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')  
        action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights)) 
        action_loss=torch.sum(action_loss)/torch.sum(weights) 

        if total_loss is None:
            total_loss=self.lambda_action_loss*action_loss
        else:
            total_loss+=self.lambda_action_loss*action_loss
        action_losses["action_loss"]=action_loss
        return action_results, total_loss, action_losses
