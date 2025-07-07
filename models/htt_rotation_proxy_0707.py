#Gated + proxynce loss

import torch
import torch.nn.functional as torch_f

from einops import repeat

from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.actionbranch import ActionClassificationBranch
from models.objclassbranch import ObjClassBranch
from models.handtypebranch import HandTypeClassificationBranch
from models.utils import  To25DBranch,compute_hand_loss,loss_str2func
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
        self.num_joints=21 if self.is_single_hand else 42


        # 1-1. 회전 특징을 위한 인코더 추가
        self.rotation_encoder = MultiLayerPerceptron(
            base_neurons=[500, 256, 128],
            out_dim=128,
            act_hidden='leakyrelu',
            act_final='none'
        )


        # 1-1. 회전 특징을 위한 인코더 추가
        self.direction_encoder = MultiLayerPerceptron(
            base_neurons=[500, 256, 128],
            out_dim=128,
            act_hidden='leakyrelu',
            act_final='none'
        )
        self.final_feature_fusion_layer = torch.nn.Linear(
            transformer_d_model + 256, transformer_d_model
        )
        
        # 게이팅 레이어
        self.rotation_gate = torch.nn.Sequential(
            torch.nn.Linear(transformer_d_model, 128),
            torch.nn.Sigmoid() # 출력을 0~1 사이로 만듦
        )
        # 게이팅 레이어
        self.direction_gate = torch.nn.Sequential(
            torch.nn.Linear(transformer_d_model, 128),
            torch.nn.Sigmoid() # 출력을 0~1 사이로 만듦
        )

        # 1. ProxyNCA Loss를 위한 프록시 파라미터 추가
        # self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2) # 기존 Triplet Loss 삭제
        self.lambda_contrastive_loss = 0.3 # 튜닝이 필요한 하이퍼파라미터

        num_action_classes = dataset_info.num_actions # e.g., 19
        embedding_dim = 128 # rotation_encoder의 출력 차원
        self.proxies = torch.nn.Parameter(torch.randn(num_action_classes, embedding_dim))

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
        self.num_objects=dataset_info.num_objects
        self.image_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.obj_classification=ObjClassBranch(num_obj=self.num_objects, feature_dim=transformer_d_model)

        # Feature to Action
        self.hand_pose3d_to_action_input=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_action_input=torch.nn.Linear(self.num_objects,transformer_d_model)

        # Egocentric Action Module (Global Transformer)
        self.concat_to_action_input=torch.nn.Linear(transformer_d_model*2,transformer_d_model)
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
        import pdb;pdb.set_trace()
        flatten_images=batch_flatten[TransQueries.IMAGE].cuda()
        rotation_features = batch_flatten[TransQueries.ROTATION_FEATURE].cuda()
        bbox_features = batch_flatten[TransQueries.DIRECTION_FEATURE].cuda()

        # import pdb;pdb.set_trace()
        # Loss
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

        # Object Classification
        flatten_olabel_feature=self.image_to_olabel_embed(flatten_pout_feature)
        weights_olabel_loss=batch_flatten['not_padding'].cuda().float()
        olabel_results,total_loss,olabel_losses=self.predict_object(sample=batch_flatten,features=flatten_olabel_feature,
                        weights=weights_olabel_loss,total_loss=total_loss,verbose=verbose)
        results.update(olabel_results)
        losses.update(olabel_losses)

        # Hand Type Classification
        weights_hlabel_loss=batch_flatten['not_padding'].cuda().float()
        # ... (기존과 동일) ...
        if self.dataset_name == 'h2o':
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype(
                sample=batch_flatten, features=flatten_in_feature, weights=weights_hlabel_loss,
                total_loss=total_loss, verbose=verbose
            )
        else:
            hlabel_results, total_loss, hlabel_losses = self.predict_handtype(
                sample=batch_flatten, features=flatten_in_feature, weights=weights_hlabel_loss,
                total_loss=total_loss, verbose=verbose
            )
        results.update(hlabel_results)
        losses.update(hlabel_losses)

        # Rotation contrastive
        rotation_embedding = self.rotation_encoder(rotation_features)   # mlp
        direction_embedding = self.direction_encoder(bbox_features)   # mlp

        # ======== Egocentric Action Module ========
        flatten_ain_feature_olabel=self.olabel_to_action_input(olabel_results["obj_reg_possibilities"])
        hand_pred_label_features = torch.stack([self.hlabel_features[int(value)] for value in hlabel_results['hand_pred_labels']])
        flatten_ain_feature_hlabel_txt=torch_f.normalize(hand_pred_label_features).to(torch.cuda.current_device())
        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_olabel),dim=1)
        flatten_ain_feature=self.concat_to_action_input(flatten_ain_feature)
        flatten_ain_feature=torch.cat((flatten_ain_feature, flatten_ain_feature_hlabel_txt), dim=1)
        flatten_ain_feature=self.hlabel_concat_to_action_input(flatten_ain_feature)

        # --- 게이트 적용 ---
        batch_size = batch_flatten[TransQueries.IMAGE].shape[0] // self.ntokens_action
        seq_ain_feature = flatten_ain_feature.view(batch_size, -1, flatten_ain_feature.shape[-1])
        seq_summary_feature = torch.mean(seq_ain_feature, dim=1)
        gate_values = self.rotation_gate(seq_summary_feature)
        gated_rotation_embedding = rotation_embedding * gate_values

        gate_values_2 = self.direction_gate(seq_summary_feature)
        gated_direction_embedding = direction_embedding * gate_values_2

        # --- 게이트 적용된 특징을 프레임 단위로 확장 및 융합 ---
        expanded_rotation_embedding = gated_rotation_embedding.unsqueeze(1).expand(-1, self.ntokens_action, -1)
        flatten_rotation_embedding = expanded_rotation_embedding.reshape(-1, 128)
        flatten_ain_feature = torch.cat((flatten_ain_feature, flatten_rotation_embedding), dim=1)
        # flatten_ain_feature = self.final_feature_fusion_layer(flatten_ain_feature)

         # --- 게이트 적용된 특징을 프레임 단위로 확장 및 융합 ---
        expanded_direction_embedding = gated_direction_embedding.unsqueeze(1).expand(-1, self.ntokens_action, -1)
        flatten_direction_embedding = expanded_direction_embedding.reshape(-1, 128)
        flatten_ain_feature = torch.cat((flatten_ain_feature, flatten_direction_embedding), dim=1)
        flatten_ain_feature = self.final_feature_fusion_layer(flatten_ain_feature)


        # --- 최종 Action Transformer 입력 준비 ---
        batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])
        batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
        batch_seq_weights_action=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_action)
        batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool()
        batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)

        batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
                                key_padding_mask=batch_seq_amasks, verbose=False)

        # --- Action Classification ---
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)
        weights_action_loss=torch.ones_like(batch_flatten['not_padding'].cuda().float()[0::self.ntokens_action])
        action_results, total_loss, action_losses=self.predict_action(sample=batch_flatten,features=batch_out_action_feature, weights=weights_action_loss,
                        total_loss=total_loss,verbose=verbose)
        results.update(action_results)
        losses.update(action_losses)

        # --- ProxyNCA Loss 계산 ---
        if train:
            # ✅ 게이트가 적용된 `gated_rotation_embedding`을 사용하도록 수정
            embeddings = torch_f.normalize(gated_rotation_embedding, p=2, dim=1)
            labels = action_results["action_gt_labels"]

            contrastive_loss = self._compute_proxynca_loss(embeddings, labels)

            total_loss += self.lambda_contrastive_loss * contrastive_loss
            losses["Proxynce_loss"] = contrastive_loss.detach()

        return total_loss, results, losses


    # ... (predict_object, predict_handtype, predict_action 등은 기존과 동일) ...

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

        # print("object_pred :", olabel_results["obj_pred_labels"])
        # print("object_gt: ", olabel_results["obj_gt_labels"])
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


    def predict_handtype(self, sample, features, weights, total_loss, verbose=False):
        hand_feature = features
        out=self.hand_type_classification(hand_feature)

        hlabel_results, hlabel_losses={},{}

        # 사용자 코드에 h2o 데이터셋 관련 로직이 없으므로, 일반적인 경우를 가정하여 수정합니다.
        # 'hand_label_right' 키가 sample에 존재해야 합니다.
        if 'hand_label_right' in sample:
            hlabel_gts = sample['hand_label_right'].cuda()
        else:
            # Fallback for other datasets if the key is different
            hlabel_gts = sample['hand_label'].cuda()


        hlabel_results["hand_gt_labels"]=hlabel_gts
        hlabel_results["hand_pred_labels"]=out["pred_labels"]
        hlabel_results["hand_reg_possibilities"]=out["reg_possibilities"]
        # print("hand_type_pred : ", out["pred_labels"])
        # print("hand_type_gt : ", hlabel_gts)
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
        print("action_pred : ", action_results["action_pred_labels"])
        print("action_gt : ",action_results["action_gt_labels"])
 
        action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')
        action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights))
        action_loss=torch.sum(action_loss)/torch.sum(weights)

        if total_loss is None:
            total_loss=self.lambda_action_loss*action_loss
        else:
            total_loss+=self.lambda_action_loss*action_loss
        action_losses["action_loss"]=action_loss
        return action_results, total_loss, action_losses


    # ✅ 2. ProxyNCA Loss 계산 함수 구현
    def _compute_proxynca_loss(self, embeddings, labels):
        """
        ProxyNCA Loss를 계산하는 함수.
        """
        # 프록시와의 거리 계산 (유클리드 거리의 제곱)
        # self.proxies를 embeddings와 같은 디바이스로 이동
        proxies = self.proxies.to(embeddings.device)
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