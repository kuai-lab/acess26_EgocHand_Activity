import torch
import torch.nn.functional as torch_f

from einops import repeat

from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func, compute_hand_loss_2d #2d만 계산하게 
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries 
import clip


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
        self.num_joints=42

        
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
        self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*2,
                                act_hidden='leakyrelu',act_final='none')        
        self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)
        
        # Object classification
        self.num_objects=63
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=transformer_d_model)
        
        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
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
        # self.hand_pred_label_txt = ['No hand', 'large diameter', 'medium wrap', 'adducted thumb', 'prismatic finger', 'pinch', 'precision disc', 'tripod', 'fixed hook', 'index finger', 'extension type', 'writing tripod', 'parallel extension', 'abduction grip', 'lateral tripod', 'quadpod', 'stick', 'pincer', 'flattened palm', 'thumb in', 'curl index finger', 'spray', 'fist', 'bend index finger']
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)

        self.dataset_name = dataset_info.name

    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        # flatten_images=batch_flatten[TransQueries.IMAGE].cuda()
        flatten_images, keypoints_csv = batch_flatten[TransQueries.IMAGE]
        flatten_images = flatten_images.cuda()
        # keypoints_csv = keypoints_csv.cuda()

        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}

        # ======== Feature Extraction ========
        flatten_in_feature, _ =self.meshregnet(flatten_images) 
        
        # ======== Egocentric Knowledge Module ========
        batch_seq_pin_feature=flatten_in_feature.contiguous().view(-1,self.ntokens_pose,flatten_in_feature.shape[-1])
        batch_seq_pin_pe=self.transformer_pe(batch_seq_pin_feature)
         
        batch_seq_pweights=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()
         
        batch_seq_pout_feature,_=self.transformer_pose(src=batch_seq_pin_feature, src_pos=batch_seq_pin_pe,
                            key_padding_mask=batch_seq_pmasks, verbose=False)
 
 
        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
        
        # Hand Pose Estimation
        flatten_hpose=self.image_to_hand_pose(flatten_pout_feature) #[128,84]
        # print("@@@@@@", flatten_hpose.shape)
        flatten_hpose=flatten_hpose.view(-1,self.num_joints,2) #[128, 42, 2]
        # print("@@@@@@", flatten_hpose.shape)
        #3d 로 변환 안할거라 이거 필요 없음
        # flatten_hpose_25d_3d=self.postprocess_hand_pose(sample=batch_flatten,scaletrans=flatten_hpose,verbose=verbose) 

        weights_hand_loss=batch_flatten['not_padding'].cuda().float() #이진마스크로 손이 있는 부분만 반영하자
        # hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,flatten_hpose_25d_3d=flatten_hpose_25d_3d,weights=weights_hand_loss,
        #                 total_loss=total_loss,verbose=verbose)        
        hand_results,total_loss,hand_losses=self.recover_hand_2d(flatten_sample=batch_flatten,flatten_pose_2d=flatten_hpose,weights=weights_hand_loss,
                        total_loss=total_loss,verbose=verbose)
        
        results.update(hand_results)
        losses.update(hand_losses)

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
        flatten_hpose2d=torch.flatten(flatten_hpose[:,:,:2],1,2)
        flatten_ain_feature_hpose=self.hand_pose3d_to_action_input(flatten_hpose2d) # flatten_ain_feature_hpose shape : (B * 128, 512)
        flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"]) # flatten_ain_feature_olabel shape : (B * 128, 512)
        
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt=torch.nn.functional.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        
        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_hpose,flatten_ain_feature_olabel),dim=1) # (B * 128, 1536)
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature) # (B * 128, 512)

        # if train:
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1) # (B * 128, 1024)
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
    
    def recover_hand_2d(self,flatten_sample,flatten_pose_2d, weights,total_loss,verbose=False):
        """
        2D 손실만 계산하도록 단순화된 recover_hand
        """
        hand_results = {}
        # GT 2D
        flatten_images, keypoints_csv = flatten_sample[TransQueries.IMAGE]
        joints2d_gt = keypoints_csv.cuda()
        gt2d = joints2d_gt[:, :, :2]
        hand_results["gt_joints2d"] = gt2d
        pred2d = flatten_pose_2d

        # 2D 손실 계산
        hand_losses = compute_hand_loss_2d(
            est2d=pred2d,
            gt2d=gt2d,
            weights=weights,
            pose_loss=self.pose_loss,
            verbose=verbose
        )

        # λ_hand_2d 가중치 곱해 누적
        loss2d = hand_losses["recov_joints2d"]
        total_loss = loss2d if total_loss is None else total_loss + loss2d

        return hand_results, total_loss, hand_losses

    # Multi-Gpu
    def predict_object(self, sample, features, weights, total_loss, verbose=False):
        olabel_feature = features
        out = self.obj_classification(olabel_feature)

        olabel_results, olabel_losses = {}, {}

        obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
        num_classes = 63

        # 동적으로 batch size 추정
        actual_batch_size = len(obj_idx_list)
        olabel_gts = torch.zeros((actual_batch_size, num_classes), device=features.device)

        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                if obj_id >= num_classes:
                    raise ValueError(f"Invalid obj_id={obj_id}, exceeds num_classes={num_classes}")
                olabel_gts[i, obj_id] = 1.0

        logits = out["reg_outs"]  # [B, C]

        olabel_results["obj_gt_labels"] = olabel_gts
        olabel_results["obj_pred_labels"] = (logits > 0).float()
        olabel_results["obj_reg_possibilities"] = torch.sigmoid(logits)

        # BCE loss (multi-label classification)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        olabel_loss = bce_loss_fn(logits, olabel_gts)  # [B, C]
        olabel_loss = torch.sum(olabel_loss, dim=1)    # [B]
        
        # weight shape 안전처리
        weights = weights.flatten()
        if weights.shape[0] != olabel_loss.shape[0]:
            raise ValueError(f"weights.shape={weights.shape} doesn't match batch={olabel_loss.shape[0]}")

        olabel_loss = olabel_loss * weights
        olabel_loss = torch.sum(olabel_loss) / (torch.sum(weights) + 1e-6)

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

