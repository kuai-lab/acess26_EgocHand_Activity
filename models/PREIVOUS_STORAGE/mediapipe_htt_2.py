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

def compute_bone_length_loss(joints3d):
        """
        joints3d: (B, num_joints, 3)
        """
        # 예시용 bone 연결 정의 (21개 손가락 keypoints 기준)
        bone_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9,10), (10,11), (11,12),    # Middle
            (0,13), (13,14), (14,15), (15,16),   # Ring
            (0,17), (17,18), (18,19), (19,20)    # Pinky
        ]

        total_loss = 0.0

        for parent_idx, child_idx in bone_pairs:
            bone_vec = joints3d[:, parent_idx] - joints3d[:, child_idx]   # (B, 3)
            bone_length = bone_vec.norm(dim=-1)                          # (B,)
            target_length = bone_length.mean().detach()                  # 평균 길이 사용 (stop-gradient)
            total_loss += ((bone_length - target_length) ** 2).mean()

        return total_loss

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
                                    
       
        # Hand Pose Estimation
        self.scale_factor = scale_factor 
        self.trans_factor = trans_factor
        # self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*3,
                                # act_hidden='leakyrelu',act_final='none')  
        
        self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*3,
                act_hidden='leakyrelu',act_final='none')        
        self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)


        # Object classification
        # self.num_objects=dataset_info.num_objects
        self.num_objects=63
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(84,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        # print(transformer_d_model)
        # self.concat_to_action_input=torch.nn.Linear(transformer_d_model*3,transformer_d_model)

        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*3,transformer_d_model)
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
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)
        self.dataset_name = dataset_info.name

    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        flatten_images=batch_flatten[TransQueries.IMAGE].cuda()
        flatten_imgs=batch_flatten[BaseQueries.IMAGE].cuda()
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images) 


        # == new ==
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # 두 손까지 감지
            min_detection_confidence=0.5
        )

        B, C, H, W = flatten_imgs.shape #128, 3, 270, 480
        mp_keypoints = []

        for i in range(B):
            img_tensor = flatten_imgs[i].detach().cpu()  # (3, H, W)
            img = img_tensor.numpy().astype(np.uint8)

            res = mp_hands.process(img)

            # 기본값은 0으로
            kp = torch.zeros((42, 2), dtype=torch.float32)

            if res.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        x_px = lm.x * W
                        y_px = lm.y * H
                        kp[hand_idx * 21 + lm_idx] = torch.tensor([x_px, y_px], dtype=torch.float32)
                        # 왼손 → 0~20, 오른손 → 21~41에 들어감
            mp_keypoints.append(kp)

        mp_hands.close()

        # 최종 텐서 (B, 42, 3)
        mp_keypoints_tensor = torch.stack(mp_keypoints).cuda()

        # Normalize keypoints to [-1, 1]
        mp_keypoints_tensor[:, :, 0] = (mp_keypoints_tensor[:, :, 0] / W) * 2 - 1
        mp_keypoints_tensor[:, :, 1] = (mp_keypoints_tensor[:, :, 1] / H) * 2 - 1
        mp_2d_flat = mp_keypoints_tensor.view(B, -1).cuda()

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
        flatten_hpose2d = mp_keypoints_tensor
        # flatten_hpose_25d_3d=self.postprocess_hand_pose(sample=batch_flatten,scaletrans=flatten_hpose,verbose=verbose) 

        # weights_hand_loss=batch_flatten['not_padding'].cuda().float()
 
        # hand_results, total_loss, hand_losses = self.recover_hand(flatten_hpose_25d_3d=flatten_hpose_25d_3d,
        #                                                           weights=weights_hand_loss, total_loss=total_loss, verbose=verbose)
        # results.update(hand_results)
        # losses.update(hand_losses)

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
        # (1) hand pose: (B, 84) -> (B, 512) -> (B * 128, 512)
        flatten_hpose2d = mp_keypoints_tensor[:, :, :2].reshape(B, -1)  # (B, 84)
        flatten_ain_feature_hpose = self.hand_pose3d_to_action_input(flatten_hpose2d)  # (B, 512)
        flatten_ain_feature_hpose = repeat(flatten_ain_feature_hpose, 'b d -> (b t) d', t=self.ntokens_action)

        # (2) object label: (B, 512) -> (B * 128, 512)
        flatten_ain_feature_olabel = self.olabel_to_action_input(olabel_results["obj_reg_possibilities"])  # (B, 512)
        flatten_ain_feature_olabel = repeat(flatten_ain_feature_olabel, 'b d -> (b t) d', t=self.ntokens_action)

        # (3) hand type: (B, 512) -> (B * 128, 512)
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt = torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        flatten_ain_feature_hlabel_txt = repeat(flatten_ain_feature_hlabel_txt, 'b d -> (b t) d', t=self.ntokens_action)

        # (4) flatten_pout_feature: already (B * 128, 512)

        # (5) concatenate all: (B * 128, 512 * 3)
        flatten_pout_feature = flatten_pout_feature.unsqueeze(1).repeat(1, self.ntokens_action, 1).view(-1, 512)
        flatten_ain_feature = torch.cat((flatten_pout_feature, flatten_ain_feature_hpose, flatten_ain_feature_olabel), dim=1)
        flatten_ain_feature = self.concat_to_action_input(flatten_ain_feature)  # (B * 128, 512)

        # (6) concatenate hand label (B * 128, 512 * 2) -> (B * 128, 512)
        flatten_ain_feature = torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1)
        flatten_ain_feature = self.hlabel_concat_to_action_input(flatten_ain_feature)  # (B * 128, 512)

        # (7) reshape to transformer input: (B, 128, 512)
        batch_seq_ain_feature = flatten_ain_feature.contiguous().view(B, self.ntokens_action, -1)

        # (8) add global token
        batch_aglobal_tokens = repeat(self.action_token, '() n d -> b n d', b=B)
        batch_seq_ain_feature = torch.cat((batch_aglobal_tokens, batch_seq_ain_feature), dim=1)  # (B, 129, 512)
        batch_seq_ain_pe = self.transformer_pe(batch_seq_ain_feature)

        # not padding token 단위로 만들어주기
        not_padding = batch_flatten['not_padding'].cuda().float()
        not_padding = not_padding.repeat_interleave(self.ntokens_action)  # (B,) → (B×128,)
        batch_seq_weights_action = not_padding.view(-1, self.ntokens_action)

        # === Transformer forward ===
        # batch_seq_weights_action = batch_flatten['not_padding'].cuda().float().view(-1, self.ntokens_action)
        batch_seq_amasks_frames = (1 - batch_seq_weights_action).bool()
        batch_seq_amasks_global = torch.zeros_like(batch_seq_amasks_frames[:, :1]).bool()
        batch_seq_amasks = torch.cat((batch_seq_amasks_global, batch_seq_amasks_frames), dim=1)

        batch_seq_aout_feature, _ = self.transformer_action(
            src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
            key_padding_mask=batch_seq_amasks, verbose=False)

        # === Action Classification ===
        batch_out_action_feature = batch_seq_aout_feature[:, 0]  # (B, 512)
        weights_action_loss = torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action])
        action_results, total_loss, action_losses = self.predict_action(
            sample=batch_flatten, features=batch_out_action_feature, weights=weights_action_loss,
            total_loss=total_loss, verbose=verbose)

        results.update(action_results)
        losses.update(action_losses)

        # flatten_hpose2d = mp_keypoints_tensor[:, :, :2]  # 만약 z가 없다면 안전하게 2D만
        # flatten_hpose2d = flatten_hpose2d.reshape(flatten_hpose2d.size(0), -1)  # (B, 84)
        # flatten_ain_feature_hpose = self.hand_pose3d_to_action_input(flatten_hpose2d)
        # flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"]) # flatten_ain_feature_olabel shape : (B * 128, 512)
        
        # hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        # flatten_ain_feature_hlabel_txt=torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        
        # flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_hpose,flatten_ain_feature_olabel),dim=1) # (B * 128, 1536)
        # # flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_olabel),dim=1) # (B * 128, 512*2=1024) => object랑  concat

        # flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) # (B * 128, 512)

        # # if train:
        # flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1536)
        # flatten_ain_feature=self.hlabel_concat_to_action_input(flatten_ain_feature)
        # batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])
        
        # # Concat trainable token
        # batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        # batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        # batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
 
        # batch_seq_weights_action=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_action)
        # batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        # batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool() 
        # batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)     

        # batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
        #                         key_padding_mask=batch_seq_amasks, verbose=False)
        
        # # Action Classification
        # batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)     
        # weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 

        # action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=batch_out_action_feature, weights=weights_action_loss,
        #                 total_loss=total_loss,verbose=verbose)
        
        # results.update(action_results)
        # losses.update(action_losses)
        # == NaN DEBUGGING ==

    
        return total_loss, results, losses
    # -------------------------------------------------------
    # -------------------------------------------------------

    def check_nan(self, name, logits=None, labels=None, weights=None):
        if logits is not None and torch.isnan(logits).any():
            print(f"❌ {name}: logits에 NaN 있음")
        if labels is not None and torch.isnan(labels).any():
            print(f"❌ {name}: labels에 NaN 있음")
        if weights is not None:
            if torch.isnan(weights).any():
                print(f"❌ {name}: weights에 NaN 있음")
            if torch.sum(weights) == 0:
                print(f"❌ {name}: weights 합이 0")
    # POSE LOSS 인데 일단 생략
    def recover_hand(self, flatten_hpose_25d_3d, weights, total_loss, verbose=False):
        hand_results, hand_losses = {}, {}

        # 예측된 3D pose
        pred_joints3d = flatten_hpose_25d_3d["rep3d"]  # (B, 21, 3) 또는 (B, 42, 3)

        hand_results["pred_joints3d"] = pred_joints3d.detach().clone()
        hand_results["pred_joints2d"] = flatten_hpose_25d_3d["rep2d"]
        hand_results["pred_jointsz"] = flatten_hpose_25d_3d["rep_absz"]

        # === Self-supervised Loss: Bone length consistency ===
        hpose_loss = compute_bone_length_loss(pred_joints3d)

        if total_loss is None:
            total_loss = hpose_loss
        else:
            total_loss += hpose_loss

        hand_losses["bone_length_loss"] = hpose_loss

        return hand_results, total_loss, hand_losses



    # def predict_object(self, sample, features, weights, total_loss, verbose=False):
    #     olabel_feature = features
    #     out = self.obj_classification(olabel_feature)

    #     olabel_results, olabel_losses = {}, {}

    #     obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
    #     # print(obj_idx_list)
    #     num_classes = 63
    #     batch_size = features.shape[0]  # ← 이걸로 대체        # batch_size = 256    #(rgb)
    #     obj_idx_list = obj_idx_list[:batch_size]

    #     # multi-hot label 생성
    #     olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)
    #     # print(enumerate(obj_idx_list))
        
    #     for i, obj_ids in enumerate(obj_idx_list):
    #         for obj_id in obj_ids:
    #             olabel_gts[i, obj_id] = 1.0     # one hot encoding
    #     # print("object_label gt: ", olabel_gts)
    #     logits = out["reg_outs"]                # shape: [B, C]

    #     olabel_results["obj_gt_labels"] = olabel_gts
    #     olabel_results["obj_pred_labels"] = (logits > 0).float()
    #     olabel_results["obj_reg_possibilities"] = torch.sigmoid(logits)

    #     # ✅ BCE loss 사용 (object multi label classification)
    #     bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    #     olabel_loss = bce_loss_fn(logits, olabel_gts)  # [B, C]
    #     olabel_loss = torch.sum(olabel_loss, dim=1)  # sum over classes per sample
    #     olabel_loss = torch.mul(olabel_loss, weights.flatten())
    #     olabel_loss = torch.sum(olabel_loss) / torch.sum(weights)

    #     if total_loss is None:
    #         total_loss = self.lambda_action_loss * olabel_loss
    #     else:
    #         total_loss += self.lambda_action_loss * olabel_loss
    #         olabel_losses["olabel_loss"] = olabel_loss

    #     return olabel_results, total_loss, olabel_losses

    def predict_object(self, sample, features, weights, total_loss, verbose=False):
        olabel_feature = features
        out = self.obj_classification(olabel_feature)

        olabel_results, olabel_losses = {}, {}

        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        # print(obj_idx_list)
        num_classes = 63
        batch_size = features.shape[0]  # ← 이걸로 대체        # batch_size = 256    #(rgb)
        obj_idx_list = obj_idx_list[:batch_size]

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
        
        self.check_nan("olabel", logits=out["reg_outs"], labels=olabel_gts, weights=weights)

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
                
        self.check_nan("hlabel", logits=out["reg_outs"], labels=hlabel_gts, weights=weights)
        return hlabel_results, total_loss, hlabel_losses


    def predict_action(self,sample,features,weights,total_loss=None,verbose=False):
        action_feature=features
        out=self.action_classification(action_feature)
        
        action_results, action_losses={},{}
        action_gt_labels=sample[BaseQueries.ACTIONIDX].cuda().clone()
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

        self.check_nan("action", logits=out["reg_outs"], labels=action_gt_labels, weights=weights)
        return action_results, total_loss, action_losses

