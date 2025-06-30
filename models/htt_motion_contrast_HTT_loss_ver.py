import torch
import torch.nn.functional as torch_f
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat

from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
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
                        ntokens_pose=1,
                        ntokens_action=120,
                        embedding_dim_final=256,
                        dataset_info=None,
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        dim_grasping_feature=128,
                        use_3d_pose=True,use_2d_pose=False, dropout=0, microaction_overlap=0.0, # [0.0, 0.99)
                        trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=3, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                        use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                        tf_heads=8, tf_layers=2,MIB_block=True):

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
        self.num_objects=63
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*2 ,transformer_d_model)
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2, transformer_d_model)
        self.dataset_name = dataset_info.name


        # TrjaectoryEncoder~MultimodalTokenizer 포함되어 있음(hf_pose.py)
        self.pose_net = HF_Pose(self.ntokens_pose, self.num_joints, num_classes,  
                 embedding_dim_final, use_2d_pose, dropout, 0.0, # microaction_overlap=0,
                 trajectory_atten_dim_per_head, trajectory_tcn_kernel_size, trajectory_tcn_stride, trajectory_tcn_dilations,
                 use_global_wrist_reference, include_orientation_in_global_wrist_ref, use_both_wrists, separate_hands,
                 tf_heads, tf_layers)

        # Disable the transformer and classifier in the pose_net. Only microaction encoding will be used.
        self.pose_net.pose_tf = nn.Identity()
        self.pose_net.classifier = nn.Identity()


    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        # import pdb;pdb.set_trace()
        flatten_images = batch_flatten["rgb_image"].cuda()          # ([8, 3, 128, 270, 480])
        # import pdb;pdb.set_trace()                                                                    
        # flatten_images: [B, C, T, H, W]
        B, C, T, H, W = flatten_images.shape
        flatten_images = flatten_images.reshape(B*T, C, H, W)              # [B*T, C, H, W] 
        flatten_images = flatten_images.cuda()               # [1024, 3, 270, 480])

        pose_seq = batch_flatten["pose_keypoint"].cuda()     # [8, 3, 128, 21, 2])
        pose_seq = pose_seq.cuda()       
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {} ;results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images)        # RGB Encoder=> (960, 512)
        pose_feat, rotation_prior_token = self.pose_net(pose_seq)     # [B, T, D]= [8, 128//16=8, 256] => (pose + rotation) token
        # combined_token = torch.cat([pose_feat, rotation_prior_token], dim=-1)
        
        ## ============================Contrastive Loss for rotation token========================== ###
        # Contrastive Loss for rotation token
        B_rot = rotation_prior_token.shape[0]
        B, T, D = pose_feat.shape   # 실제 pose_feat는 (B, num_mactions, 256)
        gt_bidirectional_labels = batch_flatten['bidirectional_label'].cuda()  # [B_total]
        gt_bidirectional_labels = gt_bidirectional_labels[:B_rot]
        gt_labels_expanded = gt_bidirectional_labels.unsqueeze(1).expand(-1, T).reshape(-1)
        rot_token_flat = rotation_prior_token.view(-1, D)
        valid_mask = (gt_labels_expanded != -1)
        assert rot_token_flat.shape[0] == valid_mask.shape[0], f"rot_token_flat={rot_token_flat.shape}, valid_mask={valid_mask.shape}"
        if valid_mask.sum() > 1:
            contrastive_loss = self.compute_contrastive_loss(
                rot_token_flat[valid_mask],
                gt_labels_expanded[valid_mask])
        else:
            contrastive_loss = torch.tensor(0.0, device=rotation_prior_token.device)
        if total_loss is None:
            total_loss = contrastive_loss
        else:
            total_loss += contrastive_loss
        losses.update({'rotation_prior_contrastive_loss': contrastive_loss})
        ## ============================================================================ ###

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
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1536)

        ### ======================================================================================= ##
        ## === Action Transfomer에 pose encoding된 피쳐(pose_token, wrist_token rot_token포함)추가 === ##'걍 둘다 추가안함
        # import pdb; pdb.set_trace()
        flatten_ain_feature=torch.cat((flatten_ain_feature_olabel, flatten_ain_feature_hlabel_txt), dim=1) #128, 1024
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
        
        results.update(action_results)
        losses.update(action_losses)
    
        return total_loss, results, losses
    
    ### ======================= Loss ============================= ###

    ###----------------------------------------------------------------\
    def compute_contrastive_loss(self, features, labels, temperature=0.07):
        # features: [N, D]
        # labels: [N]
        norm_feat = F.normalize(features, dim=-1)
        logits = torch.matmul(norm_feat, norm_feat.T) / temperature  # [N, N]
        
        labels = labels.view(-1, 1)
        
        # 유효한 mask (labels != -1)
        valid_mask = (labels != -1) & (labels.T != -1)  # [N, N]

        # positive mask (same label)
        pos_mask = (labels == labels.T) & valid_mask   # [N, N]
        # negative mask implicitly handled (all except pos and invalid)

        # Exclude self-comparison
        logits = logits - torch.eye(labels.size(0), device=features.device) * 1e9

        exp_logits = torch.exp(logits) * valid_mask.float()
        pos_sum = (exp_logits * pos_mask.float()).sum(1)
        all_sum = exp_logits.sum(1)

        loss = -torch.log((pos_sum + 1e-9) / (all_sum + 1e-9))
        return loss.mean()

    def recover_hand(self, flatten_sample, flatten_hpose_25d_3d, weights, total_loss,verbose=False):
        hand_results, hand_losses={},{}
        
        joints3d_gt = flatten_sample[BaseQueries.JOINTS3D].cuda()
        hand_results["gt_joints3d"]=joints3d_gt
        hand_results["pred_joints3d"]=flatten_hpose_25d_3d["rep3d"].detach().clone()
        hand_results["pred_joints2d"]=flatten_hpose_25d_3d["rep2d"]
        hand_results["pred_jointsz"]=flatten_hpose_25d_3d["rep_absz"]
 
            
        hpose_loss=0.
        
        joints25d_gt = flatten_sample[TransQueries.JOINTSABS25D].cuda()
        hand_losses=compute_hand_loss(est2d=flatten_hpose_25d_3d["rep2d"],
                                    gt2d=joints25d_gt[:,:,:2],
                                    estz=flatten_hpose_25d_3d["rep_absz"],
                                    gtz=joints25d_gt[:,:,2:3],
                                    est3d=flatten_hpose_25d_3d["rep3d"],
                                    gt3d= joints3d_gt,
                                    weights=weights,
                                    is_single_hand=self.is_single_hand,
                                    pose_loss=self.pose_loss,
                                    verbose=verbose)

        hpose_loss+=hand_losses["recov_joints2d"]*self.lambda_hand_2d + hand_losses["recov_joints_absz"]*self.lambda_hand_z + hand_losses["recov_joint_angle"]*self.lambda_hand_z/10+hand_losses["recov_joint_NCJ"]*self.lambda_hand_z

        if total_loss is None:
            total_loss= hpose_loss
        else:
            total_loss += hpose_loss
        return hand_results, total_loss, hand_losses

    ### ========================================================= ###
    ### ============== HTT에서 Rotation prior쓰는 함수(HF_Pose에서 이미 정의함) ============= ###
    def compute_rotation_priors(keypoints_3d_seq, window_size, overlap=0.5, sigma=3):
        """
        keypoints_3d_seq: (T, V, 3)
        window_size: microaction window size
        overlap: window overlap ratio
        sigma: smoothing sigma
        """
        T = keypoints_3d_seq.shape[0]
        step = int(window_size * (1 - overlap))
        
        # 기준 grasp points와 palm_to_finger_vector
        reference_grasp_points = None
        reference_palm_to_finger_vector = None
        
        signed_accumulated_angle_list = []
        signed_accum_angle = 0.0

        for t in range(0, T, step):
            end = min(t + window_size, T)
            window_kp = keypoints_3d_seq[t:end]  # (window_size, V, 3)
            
            for i in range(window_kp.shape[0]):
                kp = window_kp[i]
                palm_center = (kp[0] + kp[5] + kp[17]) / 3
                finger_center = (kp[4] + kp[8] + kp[12]) / 3
                palm_to_finger_vector = finger_center - palm_center
                current_grasp_points = np.array([kp[4], kp[8], kp[12]])
                if reference_grasp_points is None:
                    # 기준 설정
                    reference_grasp_points = current_grasp_points.copy()
                    reference_palm_to_finger_vector = palm_to_finger_vector.copy()
                    continue
                delta_rot, _ = ScipyRotation.align_vectors(current_grasp_points, reference_grasp_points)
                rotvec = delta_rot.as_rotvec()
                angle_inc = np.linalg.norm(rotvec)
                sign = 0
                if angle_inc > 1e-6:
                    sign = np.sign(np.dot(rotvec, reference_palm_to_finger_vector))
                signed_delta = np.rad2deg(angle_inc) * sign
                signed_accum_angle += signed_delta
                signed_accumulated_angle_list.append(signed_accum_angle)
        # smoothing
        smoothed = scipy.ndimage.gaussian_filter1d(signed_accumulated_angle_list, sigma=sigma)
        
        return torch.tensor(smoothed, dtype=torch.float32)
    ### ========================================================= ###
    ### ========================================================= ###


    def predict_object(self, sample, features, weights, total_loss, verbose=False):
        olabel_feature = features
        out = self.obj_classification(olabel_feature)

        olabel_results, olabel_losses = {}, {}

        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        # print(obj_idx_list)
        num_classes = 63
        batch_size = features.shape[0]

        # multi-hot label 생성
        olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)
        # print(enumerate(obj_idx_list))
        
        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                olabel_gts[i, obj_id] = 1.0     # one hot encoding
        # print("object_label gt: ", olabel_gts)

        logits = out["reg_outs"]                # shape: [B, C]

        olabel_results["obj_gt_labels"] = olabel_gts
        olabel_results["obj_pred_labels"] = (logits > 0).float()
        olabel_results["obj_reg_possibilities"] = torch.sigmoid(logits)

        # ✅ BCE loss 사용 (object multi label classification)
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

