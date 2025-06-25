import torch
import torch.nn.functional as torch_f
import torch.nn.functional as F
import torch.nn as nn
import cv2
from einops import repeat
import numpy as np
from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries 
import clip    
import os
from models.mlp import MultiLayerPerceptron
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from HandFormer.HandFormer.models.microaction_encoder import MicroactionEncoder
from HandFormer.HandFormer.models.hf_pose import HF_Pose
from HandFormer.HandFormer.models.mlp_1D import MLP
from HandFormer.HandFormer.models.ms_tcn_1D import MultiScale_TemporalConv as MS_TCN
from HandFormer.HandFormer.models.transformer_unimodal import Unimodal_TF
from HandFormer.HandFormer.models.transformer_bimodal import Bimodal_TF
import numpy as np
import torch
import cv2
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



class TemporalNetHandFormer(nn.Module):
    def __init__(self, 
                 dataset_info,
                 is_single_hand=False,
                 transformer_d_model=256,
                 transformer_dropout=0.0,
                 transformer_nhead=8,
                 transformer_dim_feedforward=1024,
                 transformer_num_encoder_layers_action=4,
                 transformer_num_encoder_layers_pose=2,
                 transformer_normalize_before=True,  
                 embedding_dim_final=256,
                 microaction_window_size=15,
                 ntokens_action=1,
                 rgb_input_feat_dim=512, num_classes =19,
                 use_3d_pose=True,use_2d_pose=False, dropout=0, microaction_overlap=0.0, # [0.0, 0.99)
                 trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=3, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                 use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                 tf_heads=8, tf_layers=2,MIB_block=True, modality='both', rgb_frames_to_use=-1 ):
        super().__init__()

        # 우리 데이터 맞춰서 한손, 양손, 손없을때 처리하는거 고도화해야함
        self.num_joints = 42 if not is_single_hand else 21
        self.seg_len = microaction_window_size
        self.use_2d_pose = use_2d_pose
        self.use_global_wrist_reference = use_global_wrist_reference
        self.include_orientation_in_global_wrist_ref = include_orientation_in_global_wrist_ref
        self.use_both_wrists = use_both_wrists
        self.separate_hands = separate_hands # If true, shared pose encoder applied on each hand separately, then aggregated.


        # progressively increasing the dimensions for microaction encoder and wrist TCN
        embedding_dims = [embedding_dim_final // i for i in [4, 2, 1]] # For 256 --> [64, 128, 256]
        layerwise_num_heads = [dim // trajectory_atten_dim_per_head for dim in embedding_dims] # For 256 --> [64//4, 128//4, 256//4] = [16, 32, 64]        
        coordinate_dim = 2 if self.use_2d_pose else 3 # 2D or 3D pose
        self.mactions_in_window = round(1.0/(1.0-microaction_overlap))
        self.MIB_block = MIB_block
        self.lambda_action_loss=1 # 1

        
        self.pose_enc = MicroactionEncoder(coordinate_dim, embedding_dim=embedding_dim_final, num_heads=layerwise_num_heads, dropout=dropout, \
                                            num_frames=microaction_window_size, num_joints=self.num_joints, num_hands=1 if self.separate_hands else 2, \
                                            stride=trajectory_tcn_stride, kernel_size=trajectory_tcn_kernel_size, dilations=trajectory_tcn_dilations,\
                                            global_wrist_ref=self.use_global_wrist_reference)
        
        if self.use_global_wrist_reference:
            wirst_tcn_input_dim = coordinate_dim * (2 if self.use_both_wrists else 1) # double if both hands are used
            wirst_tcn_input_dim = wirst_tcn_input_dim * (2 if self.include_orientation_in_global_wrist_ref else 1) # double if orientations are included
            self.wrist_data_bn = nn.BatchNorm1d(wirst_tcn_input_dim)
            self.wrist_tcn = nn.Sequential(
                MS_TCN(wirst_tcn_input_dim, embedding_dims[0], stride=trajectory_tcn_stride[0]),
                MS_TCN(embedding_dims[0], embedding_dims[0]),
                MS_TCN(embedding_dims[0], embedding_dims[0]),
                MS_TCN(embedding_dims[0], embedding_dims[1], stride=trajectory_tcn_stride[1]),
                MS_TCN(embedding_dims[1], embedding_dims[1]),
                MS_TCN(embedding_dims[1], embedding_dims[1]),
                MS_TCN(embedding_dims[1], embedding_dims[2], stride=trajectory_tcn_stride[2]),
                MS_TCN(embedding_dims[2], embedding_dims[2]),
                MS_TCN(embedding_dims[2], embedding_dims[2])
            )
            

        self.pose_net = HF_Pose(microaction_window_size, self.num_joints, num_classes, 
                 embedding_dim_final, use_2d_pose, dropout, 0.0, # microaction_overlap=0,
                 trajectory_atten_dim_per_head, trajectory_tcn_kernel_size, trajectory_tcn_stride, trajectory_tcn_dilations,
                 use_global_wrist_reference, include_orientation_in_global_wrist_ref, use_both_wrists, separate_hands,
                 tf_heads, tf_layers)

        # Disable the transformer and classifier in the pose_net. Only microaction encoding will be used.
        self.pose_net.pose_tf = nn.Identity()
        self.pose_net.classifier = nn.Identity()

        # RGB encoder
        self.meshregnet = ResNet_(resnet_version=18)

        self.num_actions = dataset_info.num_actions
        self.num_handtypes = dataset_info.num_handtypes
        self.num_objects = dataset_info.num_objects

        self.pose_proj = nn.Linear(embedding_dim_final, embedding_dim_final)
        # self.rgb_proj = nn.Linear(rgb_input_feat_dim, embedding_dim_final)
        self.rgb_proj = nn.Linear(512, embedding_dim_final)

        # self.res_pose_proj = MLP(256, [256], dropout=transformer_dropout, add_bn_layer=True)
        # self.res_rgb_proj = MLP(256, [256], dropout=transformer_dropout, add_bn_layer=True)
        # self.fuse_proj = MLP(embedding_dim_final * 2, [embedding_dim_final * 2], dropout=transformer_dropout, add_bn_layer=True)

        # Modality Interaction Block (MIB) MLPs
        self.res_pose_proj = MLP(embedding_dim_final, [embedding_dim_final], dropout=dropout, add_bn_layer=True) # Residual Connection for Pose Tokens
        self.res_rgb_proj = MLP(embedding_dim_final, [embedding_dim_final], dropout=dropout, add_bn_layer=True) # Residual Connection for RGB Tokens
        self.posergb_proj = MLP(embedding_dim_final*2, [embedding_dim_final*2], dropout=dropout, add_bn_layer=True) # Feaute Mixing
        

        self.mib_transformer = Bimodal_TF(
            transformer_d_model=transformer_d_model,
            num_heads_=transformer_nhead,
            num_layers_=transformer_num_encoder_layers_action,
            dropout=transformer_dropout,
            return_all_tokens=False,   # 필요에 따라 True 가능
            modality='both',           # pose, rgb, both 중 선택
            rgb_frames_to_use=-1       # or 1/2/4 등
        )

        self.feat_anticipation_mlp = nn.Sequential(
            MLP(512, [512, 1024], dropout=0.8, add_bn_layer=True),
            nn.Linear(1024,256)
        )

        # Classifiers
        self.classifier_action = nn.Linear(embedding_dim_final, self.num_actions)
        self.classifier_verb = nn.Linear(embedding_dim_final,  self.num_actions)
        self.classifier_noun = nn.Linear(embedding_dim_final,  self.num_objects)

        # Hand type classifier
        self.classifier_handtype = HandTypeClassificationBranch(num_types=self.num_handtypes, hand_feature_dim=transformer_d_model)

        # Object classifier
        self.obj_classifier = ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=transformer_d_model)
        self.ntokens_action=ntokens_action

    # end to end 액션 학습 loss 추가해야함 
    def forward(self, batch_flatten, epoch=0, train=True, verbose=False, return_atten_map=False):
        rgb_img = batch_flatten["rgb_image"].cuda()  # [B, 3, T, H, W]
        B, C, T, H, W = rgb_img.shape                # e.g)[2, 3, 128, 270, 480]
        # import pdb; pdb.set_trace()

        rgb_img = rgb_img.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        window_size = 15
        rgb_feat, _ = self.meshregnet(rgb_img)  # [B*T, D] resnet 통과 [B*T, 512] (ResNet 마지막 pooling layer output)
        rgb_feat = rgb_feat.view(B, T, -1)      # [B, 128, D]
        # print("dkdkdkd", rgb_feat.shape)

        valid_T = (T // window_size) * window_size        # 128 => 120(15 배수)
        # ↓↓↓ 평균 전에 projection 해야 차원 보존됨
        rgb_feat = rgb_feat[:, :valid_T, :].contiguous().view(B * valid_T, -1)     # [B, 120, 512]
        rgb_feat = self.rgb_proj(rgb_feat.view(B * valid_T, -1))     # [B*120, 256]
        rgb_feat = rgb_feat.view(B, valid_T, -1)                      # [B, 120, 256]
        # rgb_feat = rgb_feat.view(B, valid_T // window_size, window_size, -1).mean(dim=2)  # [B, 8, 256]  micro-action 기반 RGB 토큰 생성 방식
           # 논문 방식 (중간 프레임 선택)
        rgb_feat = rgb_feat.view(B, valid_T // window_size, window_size, -1)[:, :, window_size // 2, :]

        # print("rgb_tokens.shape:", rgb_feat.shape)        # [2, 8, 256] = [B, K:microaction 갯수(120//15), 256]
        rgb_feat = F.normalize(rgb_feat, dim=-1)    # rgb token

        # 검토 완

        keypoints_csv = batch_flatten["pose_keypoint"].cuda()  # [B, 3, T, J, M]
        scale_x = W / 848
        scale_y = H / 480
        pose_3d = keypoints_csv.clone()
        pose_3d[:, 0, :, :, :] *= scale_x
        pose_3d[:, 1, :, :, :] *= scale_y
        # print("pose_3d", pose_3d.shape)                       # [2, 3, 128, 21, 2]
        B, C, T, V, M = pose_3d.shape

        pose_tokens = self.pose_net(pose_3d)                  # [B, K(num_mactions), D]=[2,8,256]
        # import pdb; pdb.set_trace()
        pose_feat = self.pose_proj(pose_tokens)               # [B, K(num_maction), D_in] → [B, T, D_out]
        

        # print("after pose_proj:", pose_feat.shape)            # [2, 8, 256]

        pose_feat = F.normalize(pose_feat, dim=-1)
        # print("pose_feat.shape:", pose_feat.shape)  
        # print(" pose_feat last dim size:", pose_feat.shape[-1])
        # print("✅ pose_feat.view shape:", pose_feat.reshape(B*T,-1).shape)  # [256, 16] 
        concat_feat = torch.cat([pose_feat, rgb_feat], dim=-1)  #  [B, T, 512]

        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}

        
        # 🚨여기 문제
        # Pass all features through MIB MLPs
        # pose_feat = self.res_pose_proj(pose_feat.view(B*T, -1)).view(B, T, -1)
        B, T, D = pose_feat.shape   # 실제 pose_feat는 (B, num_mactions, 256)
        pose_feat = self.res_pose_proj(pose_feat.view(B*T, D)).view(B, T, -1)
        # pose_feat = self.res_pose_proj(pose_feat.contiguous().view(B * T, -1)).view(B, T, -1)
        rgb_feat = self.res_rgb_proj(rgb_feat.view(B*T, -1)).view(B, T, -1)
        fused_feat = self.posergb_proj(concat_feat.view(B*T, -1)).view(B, T, -1)
        # print("✅ fused_feat after proj:", fused_feat.shape)  # 반드시 [B, T, 512]=[8,8,512]
        # import pdb; pdb.set_trace()        # 오타: set_traace → set_trace
        # 🚨여기 문제 끝

        ### Detour for feature anticipation ###
        # Calculate l1 loss for feature anticipation
        rgb_feat_for_next_maction = self.feat_anticipation_mlp(fused_feat.view(B*T,-1))
        rgb_feat_for_next_maction = rgb_feat_for_next_maction.view(B,T,-1)
        # Reconstructed vs original RGB features
        reconst_rgb_feat = rgb_feat_for_next_maction[:, 0:T-1, :]
        original_rgb_feat = rgb_feat[:, 1:T, :] # Shifted by 1 frame as these are the frames we anticipated.
        # Anticipation loss -- l1 or mse
        l1_loss = F.l1_loss(reconst_rgb_feat, original_rgb_feat)
        # l1_loss = F.mse_loss(reconst_rgb_feat, original_rgb_feat)
        ### Detour ends ###
        l1_loss = F.l1_loss(reconst_rgb_feat, original_rgb_feat)
        losses.update({"l1_loss": l1_loss})

        # [8,8,256] 으로 바뀌는 모먼트
        if self.MIB_block: # Pass mixed features to transformer -- along with residual connections from pose and rgb features
            maction_feat_for_tf =  fused_feat + torch.cat([pose_feat, rgb_feat], dim=-1)
        else:
            maction_feat_for_tf =  torch.cat([pose_feat, rgb_feat], dim=-1) # Ignored mixed features (no PoseRGB feat)
       
        if return_atten_map:
            summary, summary_v, summary_n, atten_map = self.mib_transformer(maction_feat_for_tf, return_atten_map)
        else:
            summary, summary_v, summary_n = self.mib_transformer(maction_feat_for_tf)                
        
        out = self.classifier_action(summary)
        out_verb = self.classifier_verb(summary_v)
        out_noun = self.classifier_noun(summary_n)

        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 
        action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=out, weights=weights_action_loss,
        total_loss=total_loss,verbose=verbose)
        losses.update(action_losses)

        # import pdb; pdb.set_trace()
        if return_atten_map:
            return out, out_verb, out_noun, l1_loss.unsqueeze(0), atten_map
        else:
            return out, out_verb, out_noun, losses



###----------------------------------------------------------------

    # 여기서 안쓰임(object쪽 multi-label bce loss)
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


    # 여기서 안쓰임 (hand type cross entropy loss)
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



    # # 여기서 쓰이도록 수정해야함 (action cross entropy loss)
    # def predict_action(self,sample,features,weights,total_loss=None,verbose=False):
    #     action_feature=features
    #     out=self.action_classification(action_feature)
        
    #     action_results, action_losses={},{}
    #     action_gt_labels=sample[BaseQueries.ACTIONIDX].cuda()[0::self.ntokens_action].clone()
    #     action_results["action_gt_labels"]=action_gt_labels
    #     action_results["action_pred_labels"]=out["pred_labels"]
 
    #     action_results["action_reg_possibilities"]=out["reg_possibilities"]
    #     action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')  
    #     action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights)) 
    #     action_loss=torch.sum(action_loss)/torch.sum(weights) 

    #     if total_loss is None:
    #         total_loss=self.lambda_action_loss*action_loss
    #     else:
    #         total_loss+=self.lambda_action_loss*action_loss
    #     action_losses["action_loss"]=action_loss

    #     return action_results, total_loss, action_losses


    def predict_action(self, sample, features, weights, total_loss=None, verbose=False):
        # logits = self.classifier_action(features)  # [B, num_classes]
        logits = features  # 이미 classifier 지나간 경우
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)  # [B]

        # 만약 one-hot이면 index로 변환
        # if gt_all.ndim == 2 and gt_all.size(1) == logits.size(1):
        #     gt_all = torch.argmax(gt_all, dim=1)

        # 마이크로액션 단위 추출
        # self.ntokens_action = T // seg_len → 일반적으로 128//16 = 8
        # 즉 B=2, ntokens_action=8 → B*T = 16
        B = sample["pose_keypoint"].shape[0]
        gt_action_labels = sample[BaseQueries.ACTIONIDX].view(B, -1)[:, 0]  # [B]
        # assert gt_labels.shape[0] == B, f"gt_labels.shape={gt_labels.shape}, B={B}"
        device = features.device
        gt_action_labels = gt_action_labels.to(device)
        B = features.size(0)
        weights = weights.view(B, -1)[:, 0]  # [B*T] → [B]

        loss = F.cross_entropy(features, gt_action_labels, reduction='none')
        loss = loss * weights
        loss = torch.sum(loss) / (torch.sum(weights) + 1e-5)


        if total_loss is None:
            total_loss = self.lambda_action_loss * loss
        else:
            total_loss += self.lambda_action_loss * loss
        # import pdb;pdb.set_trace()

        action_results = {
            "action_gt_labels": gt_action_labels,
            "action_pred_labels": pred_labels,
            "action_reg_possibilities": probs,
        }
        # import pdb;pdb.set_trace()
        print("pred_label:", action_results["action_pred_labels"])
        print("gt_label :", action_results["action_gt_labels"])
        action_losses = {"action_loss": loss}
        return action_results, total_loss, action_losses


    

