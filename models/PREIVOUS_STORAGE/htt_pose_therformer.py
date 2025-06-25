import torch
import torch.nn.functional as F
import torch.nn.functional as torch_f
import matplotlib.pyplot as plt
import os
from einops import repeat
from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import To25DBranch, compute_hand_loss, loss_str2func
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries
import clip
from models.segmentation import ResCNN, ResNetFeatureExtractor_new, MaskLoss
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder

class TemporalNet(torch.nn.Module):
    def __init__(self, 
                 is_single_hand,
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
                 dim_grasping_feature=128):

        super().__init__()

        self.ntokens_pose = ntokens_pose
        self.ntokens_action = ntokens_action

        self.is_single_hand = is_single_hand
        self.num_joints = 21 if self.is_single_hand else 42

        self.pose_loss = loss_str2func()[pose_loss]

        self.lambda_hand_z = 100
        self.lambda_hand_2d = 1
        self.lambda_action_loss = 1
        self.lambda_handtype_loss = 1

        # --- Feature extraction ---
        self.rgb_backbone = resnet.resnet18(pretrained=True)
        self.thermal_backbone = ResNetFeatureExtractor_new(transformer_d_model=1024, temporal_d_model=512, resnet_version='resnet18')

        # --- Pose estimation for thermal ---
        encoder_layer = DeformableTransformerEncoderLayer(d_model=1024, n_levels=1, n_points=8)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 2)
        self.simplecnn = ResCNN(1024, 512)
        self.t_transformer_pe = PositionalEncoding(d_model=512)
        self.transformer_pe = PositionalEncoding(d_model=1024)
        self.transformer_pose = Transformer_Encoder(
            d_model=512, nhead=transformer_nhead, num_encoder_layers=transformer_num_encoder_layers_pose,
            dim_feedforward=transformer_dim_feedforward, dropout=0.0, activation="relu",
            normalize_before=transformer_normalize_before
        )

        self.image_to_hand_pose = MultiLayerPerceptron(
            base_neurons=[512, 512, 512], out_dim=self.num_joints*3,
            act_hidden='leakyrelu', act_final='none'
        )
        self.scale_factor = scale_factor
        self.trans_factor = trans_factor
        self.postprocess_hand_pose = To25DBranch(trans_factor=self.trans_factor, scale_factor=self.scale_factor)
        self.loss_fn = MaskLoss()

        # --- Object classification ---
        self.num_objects = 63
        self.image_to_olabel_embed = torch.nn.Linear(512, 512)
        self.obj_classification = ActionClassificationBranch(num_actions=self.num_objects, action_feature_dim=512)

        # --- Hand Type classification ---
        self.hand_type_classification = HandTypeClassificationBranch(num_types=dataset_info.num_handtypes, hand_feature_dim=512)

        # --- Egocentric Action module ---
        self.hand_pose3d_to_action_input = torch.nn.Linear(self.num_joints * 2, 512)
        self.olabel_to_action_input = torch.nn.Linear(self.num_objects, 512)
        self.concat_to_action_input = torch.nn.Linear(512 * 3, 512)

        self.action_token = torch.nn.Parameter(torch.randn(1,1,512))
        self.transformer_action = Transformer_Encoder(
            d_model=512, nhead=transformer_nhead, num_encoder_layers=transformer_num_encoder_layers_action,
            dim_feedforward=transformer_dim_feedforward, dropout=0.0,
            activation="relu", normalize_before=transformer_normalize_before
        )
        self.action_classification = ActionClassificationBranch(num_actions=dataset_info.num_actions, action_feature_dim=512)

        # --- CLIP for Hand Label Text Embedding ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.hand_pred_label_txt = 'None,Quadpod,small Diameter,Medium Diameter,Thumb up,Thumb-Middle Grip,Tip Pinch,Disk Grip,Dynamic Tripod,Fixed Hook,Fist,Large Diameter,parallel Extension,Thumb-2 Finger,Writing Tripod,Tripod,Hand Clench,Pincer Grip,Open Hand,Stirring,Spray-Trigger Grip,Index Finger Flexion,Thumb Tucked,Extended Index Curl,Relaxing Hand,Dynamic Flatten,Dynamic Pinch,Index Finger,Full Rotation,Dynamic Parallel Extension,Dynamic Lateral Pinch,Poking,Middle Rotation,Index Rotation,Adduction Grip,Dynamic Diameter,Palmar,Hammering,Extension Type'.split(',')
        self.tokenized_label = clip.tokenize(self.hand_pred_label_txt).to(self.device)
        self.hlabel_features = self.clip_model.encode_text(self.tokenized_label).detach()
        self.hlabel_concat_to_action_input = torch.nn.Linear(512*2, 512)

        self.dataset_name = dataset_info.name

    def forward(self, batch_flatten, epoch=0, train=True, verbose=False):
        rgb_input, thermal_input = batch_flatten[TransQueries.IMAGE]
        rgb_input = rgb_input.cuda()
        # thermal_input = thermal_input.cuda()    # (480,270)
        
        thermal_input = F.interpolate(rgb_input, size=(135, 240), mode='bilinear', align_corners=False)

        total_loss = torch.tensor([0.0], device='cuda')
        results = {}
        losses = {}

        # --- RGB feature ---
        rgb_feature, _ = self.rgb_backbone(rgb_input)

        # --- Thermal feature + Deformable Attention ---
        h_feature, pred_mask, _ = self.thermal_backbone(thermal_input)
        pred_mask = torch.sigmoid(pred_mask)

        
        pred_mask_np = pred_mask[0, 0].detach().cpu().numpy()  # (H, W) - 첫 번째 샘플의 마스크

        # 저장 경로 지정
        save_dir = "./debug_vis/mask_vis"
        os.makedirs(save_dir, exist_ok=True)


        # 시각화 및 저장
        plt.figure(figsize=(4, 4))
        plt.imshow(pred_mask_np, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Predicted Mask [mean={pred_mask_np.mean():.4f}]")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mask_epoch{epoch}_sample0.png"))
        plt.close()


        print("📏 pred_mask mean:", pred_mask.mean().item())
        if pred_mask.mean() < 0.05:
            print("⚠️ No hand detected → skip pose estimation")
            flatten_pout_feature = torch.zeros((batch_size * ntokens_pose, transformer_d_model), device='cuda')
       
        else:
            # --- Deformable Transformer ---
            src_flatten = h_feature.flatten(2).transpose(1,2)
            spatial_shapes = torch.as_tensor([(h_feature.shape[2], h_feature.shape[3])], dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            memory = self.encoder(
                src_flatten, spatial_shapes, level_start_index,
                pred_mask.squeeze(1).unsqueeze(0) > 0.9,
                self.transformer_pe(src_flatten),
                pred_mask.view(pred_mask.shape[0], -1) < 0.1
            )
            memory = memory.transpose(1, 2).contiguous().view(-1, h_feature.shape[1], h_feature.shape[2], h_feature.shape[3])
            
            spatial_low_dim_feature = self.simplecnn(memory)
            spatial_low_dim_feature = spatial_low_dim_feature.view(-1, self.ntokens_pose, spatial_low_dim_feature.shape[-1])
            
            pos_embed_t = self.t_transformer_pe(spatial_low_dim_feature)
            batch_seq_pweights = batch_flatten['not_padding'].cuda().float().view(-1, self.ntokens_pose)
            batch_seq_pmasks = (1 - batch_seq_pweights).bool()

            batch_seq_pout_feature, _ = self.transformer_pose(
                src=spatial_low_dim_feature, src_pos=pos_embed_t,
                key_padding_mask=batch_seq_pmasks, verbose=False
            )
            flatten_pout_feature = batch_seq_pout_feature.flatten(0, 1)

        # NaN 방지
        if torch.isnan(flatten_pout_feature).any():
            print("❌ NaN in flatten_pout_feature -> replacing with zeros")
            flatten_pout_feature = torch.nan_to_num(flatten_pout_feature, nan=0.0)
        # # --- Mask loss ---
        gt_mask = batch_flatten[TransQueries.MASK].cuda()
        
        gt_mask = (gt_mask > 0.5).float()
        gt_mask = gt_mask[:, :1, :, :]
        gt_mask = F.interpolate(gt_mask, size=pred_mask.shape[-2:], mode='nearest')
        mask_loss = self.loss_fn(pred_mask, gt_mask)
        losses.update({"mask_loss": mask_loss})
        total_loss += mask_loss

        # --- Deformable Transformer ---
        src_flatten = h_feature.flatten(2).transpose(1,2)
        spatial_shapes = torch.as_tensor([(h_feature.shape[2], h_feature.shape[3])], dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index,
            pred_mask.squeeze(1).unsqueeze(0) > 0.9,
            self.transformer_pe(src_flatten),
            pred_mask.view(pred_mask.shape[0], -1) < 0.1
        )
        memory = memory.transpose(1, 2).contiguous().view(-1, h_feature.shape[1], h_feature.shape[2], h_feature.shape[3])

        if torch.isnan(memory).any():
            print("🔥 NaN in memory from deformable encoder")
        spatial_low_dim_feature = self.simplecnn(memory)
        spatial_low_dim_feature = spatial_low_dim_feature.view(-1, self.ntokens_pose, spatial_low_dim_feature.shape[-1])
        if torch.isnan(spatial_low_dim_feature).any():
            print("🔥 NaN in spatial_low_dim_feature BEFORE transformer")
        print(spatial_low_dim_feature.min(), spatial_low_dim_feature.max())
        # --- Temporal Transformer (Pose) ---
        pos_embed_t = self.t_transformer_pe(spatial_low_dim_feature)
        batch_seq_pweights = batch_flatten['not_padding'].cuda().float().view(-1, self.ntokens_pose)
        batch_seq_pmasks = (1 - batch_seq_pweights).bool()
        print("mask valid ratio:", (batch_seq_pmasks == False).float().mean())


        batch_seq_pout_feature, _ = self.transformer_pose(
            src=spatial_low_dim_feature, src_pos=pos_embed_t,
            key_padding_mask=batch_seq_pmasks, verbose=False
        )
        flatten_pout_feature = torch.flatten(batch_seq_pout_feature, start_dim = 0, end_dim = 1)
        # NaN 방지
        if torch.isnan(flatten_pout_feature).any():
            print("❌ NaN in flatten_pout_feature -> skip downstream usage")
            flatten_pout_feature = torch.zeros_like(flatten_pout_feature)

        # BEFORE embedding
        if torch.isnan(flatten_pout_feature).any():
            print("❌ NaN in flatten_pout_feature")

        # --- Hand Pose Estimation ---
        flatten_hpose = self.image_to_hand_pose(flatten_pout_feature)
        flatten_hpose = flatten_hpose.view(-1, self.num_joints, 3)
        flatten_hpose_25d_3d = self.postprocess_hand_pose(sample=batch_flatten, scaletrans=flatten_hpose, verbose=verbose)

        # weights_hand_loss = batch_flatten['not_padding'].cuda().float()
        # hand_results, total_loss, hand_losses = self.recover_hand(
        #     flatten_sample=batch_flatten, 
        #     flatten_hpose_25d_3d=flatten_hpose_25d_3d,
        #     weights=weights_hand_loss,
        #     total_loss=total_loss,
        #     verbose=verbose
        # )
        # results.update(hand_results)
        # losses.update(hand_losses)

       # --- Object Classification (RGB feature) ---
        # 1. rgb_feature를 먼저 ntokens_pose 형태로 정리
        flatten_rgb_feature = rgb_feature.view(-1, self.ntokens_pose, rgb_feature.shape[-1])
        # print("rgb feature max:", flatten_rgb_feature.abs().max())

        # 2. TransformerPose 통과
        rgb_pos_embed = self.t_transformer_pe(flatten_rgb_feature)
   
        batch_seq_weights_rgb = batch_flatten['not_padding'].cuda().float().view(-1, self.ntokens_pose)
        batch_seq_masks_rgb = (1 - batch_seq_weights_rgb).bool()

        batch_seq_out_feature_rgb, _ = self.transformer_pose(
            src=flatten_rgb_feature, src_pos=rgb_pos_embed,
            key_padding_mask=batch_seq_masks_rgb, verbose=False
        )

        # 3. 그리고 flatten
        flatten_pout_feature_rgb = batch_seq_out_feature_rgb.flatten(0, 1)

        flatten_olabel_feature = self.image_to_olabel_embed(flatten_pout_feature_rgb)

        weights_olabel_loss = batch_flatten['not_padding'].cuda().float()
        olabel_results, total_loss, olabel_losses = self.predict_object(
            sample=batch_flatten,
            features=flatten_olabel_feature,
            weights=weights_olabel_loss,
            total_loss=total_loss,
            verbose=verbose
        )
        results.update(olabel_results)
        losses.update(olabel_losses)

        # --- Hand Type Classification (RGB feature) ---
        weights_hlabel_loss = batch_flatten['not_padding'].cuda().float()
        if self.dataset_name == 'h2o':
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype_h2o(
                sample=batch_flatten,
                features=flatten_pout_feature_rgb,  # <=== **RGB feature 사용!!**
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        else:
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype(
                sample=batch_flatten,
                features=flatten_pout_feature_rgb,  # <=== **RGB feature 사용!!**
                weights=weights_hlabel_loss,
                total_loss=total_loss,
                verbose=verbose
            )
        results.update(hlabel_results)
        losses.update(hlabel_losses)

        # --- Egocentric Action Module ---
        flatten_hpose2d = torch.flatten(flatten_hpose[:, :, :2], 1, 2)  # (B, joints*2=42)
        flatten_ain_feature_hpose = self.hand_pose3d_to_action_input(flatten_hpose2d)

        flatten_ain_feature_olabel = self.olabel_to_action_input(olabel_results["obj_reg_possibilities"])

        hand_pred_label_features = torch.stack([
            self.hlabel_features[int(idx)] for idx in hlabel_results["hand_pred_labels"]
        ])
        flatten_ain_feature_hlabel_txt = F.normalize(hand_pred_label_features).to(torch.cuda.current_device())

        # Concat (Pose + Object + Handtype)
        flatten_ain_feature = torch.cat(
            (flatten_pout_feature, flatten_ain_feature_hpose, flatten_ain_feature_olabel),
            dim=1
        )
        flatten_ain_feature = self.concat_to_action_input(flatten_ain_feature)

        flatten_ain_feature = torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1)
        flatten_ain_feature = self.hlabel_concat_to_action_input(flatten_ain_feature)

        batch_seq_ain_feature = flatten_ain_feature.contiguous().view(-1, self.ntokens_action, flatten_ain_feature.shape[-1])

        # Concat trainable action token
        batch_aglobal_tokens = repeat(self.action_token, '() n d -> b n d', b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature = torch.cat((batch_aglobal_tokens, batch_seq_ain_feature), dim=1)

        batch_seq_ain_pe = self.t_transformer_pe(batch_seq_ain_feature)

        batch_seq_weights_action = batch_flatten['not_padding'].cuda().float().view(-1, self.ntokens_action)
        batch_seq_amasks_frames = (1 - batch_seq_weights_action).bool()
        batch_seq_amasks_global = torch.zeros_like(batch_seq_amasks_frames[:, :1]).bool()
        batch_seq_amasks = torch.cat((batch_seq_amasks_global, batch_seq_amasks_frames), dim=1)

        batch_seq_aout_feature, _ = self.transformer_action(
            src=batch_seq_ain_feature,
            src_pos=batch_seq_ain_pe,
            key_padding_mask=batch_seq_amasks,
            verbose=False
        )

        # batch_out_action_feature = batch_seq_aout_feature[:, 0].flatten(1)
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)    
        # --- Action Classification ---
        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action]) 

        action_results, total_loss, action_losses = self.predict_action(
            sample=batch_flatten,
            features=batch_out_action_feature,
            weights=weights_action_loss,
            total_loss=total_loss,
            verbose=verbose
        )
        results.update(action_results)
        losses.update(action_losses)

        # --- Final Loss ---    
        return total_loss, results, losses
    

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

    # Therformer 버전 recover_hand (3D pose GT)
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

            if self.loss_norm:
                hpose_loss+=  hand_losses["recov_joints2d"]/(hand_losses["recov_joints2d"].detach() + 1e-9)+ hand_losses["recov_joints_absz"]/((hand_losses["recov_joints_absz"].detach() + 1e-9))
            else:
                hpose_loss+=hand_losses["recov_joints2d"]*self.lambda_hand_2d+ hand_losses["recov_joints_absz"]*self.lambda_hand_z + hand_losses["recov_joint3d"] *self.lambda_hand_z
            

            if total_loss is None:
                total_loss= hpose_loss
            else:
                total_loss += hpose_loss



    # def predict_object(self, sample, features, weights, total_loss, verbose=False):
    #     olabel_feature = features
    #     out = self.obj_classification(olabel_feature)

    #     olabel_results, olabel_losses = {}, {}

    #     obj_idx_list = sample[BaseQueries.OBJIDX]  # list of list[int], e.g. [[3], [1,4]]
    #     # print(obj_idx_list)
    #     num_classes = 63
    #     batch_size = features.size(0)

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
        obj_idx_list = sample[BaseQueries.OBJIDX]
        num_classes = 63
        batch_size = features.size(0)

        olabel_gts = torch.zeros((batch_size, num_classes), device=features.device)
        for i, obj_ids in enumerate(obj_idx_list):
            for obj_id in obj_ids:
                if 0 <= obj_id < num_classes:
                    olabel_gts[i, obj_id] = 1.0
                else:
                    print(f"[Warning] Invalid obj_id {obj_id} at index {i}")

        logits = out["reg_outs"]
        if torch.isnan(logits).any():
            print("❗ NaN detected in logits before BCE loss.")

        bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        olabel_loss = bce_loss_fn(logits, olabel_gts)  # [B, C]
        olabel_loss = torch.sum(olabel_loss, dim=1)  # per-sample sum

        weight_sum = torch.sum(weights)
        if weight_sum == 0:
            print("⚠️ Zero weights in olabel loss; skipping")
            olabel_loss = torch.tensor(0.0, device=logits.device)
        else:
            olabel_loss = torch.sum(olabel_loss * weights.flatten()) / weight_sum

        olabel_results["obj_gt_labels"] = olabel_gts
        olabel_results["obj_pred_labels"] = (logits > 0).float()
        olabel_results["obj_reg_possibilities"] = torch.sigmoid(logits)

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
        # print(f"[DEBUG] action_gt_labels shape: {action_gt_labels.shape}")
        # print(f"[DEBUG] action_gt_labels min/max: {action_gt_labels.min()} / {action_gt_labels.max()}")
        # print(f"[DEBUG] action logits shape: {out['reg_outs'].shape}")
        # print(f"[DEBUG] action weights sum: {torch.sum(weights)}")
        # print(f"[DEBUG] action weights contains NaN: {torch.isnan(weights).any()}")
        # print(f"[DEBUG] action logits contains NaN: {torch.isnan(out['reg_outs']).any()}")
        # ==================== DEBUG END ======================
        
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

