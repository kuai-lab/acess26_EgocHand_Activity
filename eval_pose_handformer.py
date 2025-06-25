import os
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from models.htt_handformer_type_loss_motion_final_contrastive_loss_concat import TemporalNetHandFormer
from netscripts import get_dataset_rgb_wilor_former_all
from datasets import collate
from libyana.modelutils import freeze
from libyana.randomutils import setseeds
from netscripts import reloadmodel, get_dataset_rgb_wilor_former_all
from netscripts import epochpass
from datasets.queries import BaseQueries, TransQueries 
from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def collate_fn(seq, extend_queries=[]):
    return collate.seq_extend_flatten_collate(seq, extend_queries)


def evaluate_handformer(model, dataloader, device, save_dir=None, epoch=0, save_score=True):
    model.eval()
    model.to(device)

    all_scores, all_labels, all_indices = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # pred_action, _, _, _ = model(batch)  # [B, num_classes]
            pred_action, _ , _, _, _ = model(batch)  # [B, num_classes]
            # probs = torch.softmax(pred_action, dim=-1)
            preds = torch.argmax(pred_action, dim=-1)
            print("prediction_action label:", preds)

            # print("preds:", preds)
            # print(preds.shape)

            B = batch["pose_keypoint"].shape[0]
            gt_action_labels = batch[BaseQueries.ACTIONIDX].view(B, -1)[:, 0]  # [B]
            print("GT action: ", gt_action_labels)
            # print(gt_action_labels.shape)

            all_scores.append(preds.cpu().numpy())
            all_labels.append(gt_action_labels.cpu().numpy())
            all_indices.extend(batch["sample_info"])

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # top1_acc = np.mean(np.argmax(all_scores, axis=-1) == all_labels)
    top1_acc = np.mean(all_scores == all_labels)
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")

    if save_score and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"epoch{epoch+1}_score.pkl"), "wb") as f:
            pickle.dump({idx: score for idx, score in zip(all_indices, all_scores)}, f)


    return top1_acc , all_labels, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', type=str, required=True)
    parser.add_argument('--dataset_folder', type=str, default='../data_MHAV/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--ntokens_pose', type=int, default=16)
    parser.add_argument('--ntokens_action', type=int, default=120)
    parser.add_argument('--spacing', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--enc_pose_layers', type=int, default=2)
    parser.add_argument('--enc_action_layers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./results_eval_handformer/')
    parser.add_argument('--manual_seed', type=int, default=42)
    parser.add_argument("--val_dataset", choices=["mhavhands", "fhbhands"], default="mhavhands",) 
    parser.add_argument("--val_split", default="test", choices=["test", "train", "val"])
    parser.add_argument("--center_idx", default=0, type=int)
    parser.add_argument(
        "--center_jittering", type=float, default=0.1, help="Controls magnitude of center jittering"
    )
    parser.add_argument(
        "--scale_jittering", type=float, default=0, help="Controls magnitude of scale jittering"
    )
    parser.add_argument('--png_name', type=str, required=True)

    args = parser.parse_args()
    setseeds.set_all_seeds(args.manual_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print("**** Lets eval on", args.val_dataset, args.val_split)
    val_dataset, _ = get_dataset_rgb_wilor_former_all.get_dataset_htt(
        args.val_dataset,
        dataset_folder=args.dataset_folder,
        split=args.val_split, 
        no_augm=True,
        scale_jittering=args.scale_jittering,
        center_jittering=args.center_jittering,
        ntokens_pose=args.ntokens_pose,
        ntokens_action=args.ntokens_action,
        spacing=args.spacing,
        is_shifting_window=True,
        split_type="actions"
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn= collate_fn,
    )

    dataset_info = val_dataset.pose_dataset

    model= TemporalNetHandFormer(dataset_info=dataset_info,
                is_single_hand=args.val_dataset!="h2ohands",
                transformer_num_encoder_layers_action=args.enc_action_layers,
                transformer_num_encoder_layers_pose=args.enc_pose_layers,
                transformer_d_model=args.hidden_dim,
                transformer_dropout=args.dropout,
                transformer_nhead=args.nheads,
                transformer_dim_feedforward=args.dim_feedforward,
                transformer_normalize_before=True,
                embedding_dim_final=256,
                microaction_window_size=15,
                rgb_input_feat_dim=2048,
                use_3d_pose=True)

    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("❌ Missing keys:", missing_keys)
        print("❗ Unexpected keys:", unexpected_keys)

    epoch=reloadmodel.reload_model(model,args.resume_path)
    use_multiple_gpu= torch.cuda.device_count() > 1
    if use_multiple_gpu:
        assert False, "Not implement- Eval with multiple gpus!"
        #model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()

    freeze.freeze_batchnorm_stats(model)

    model_params = filter(lambda p: p.requires_grad, model.parameters())   
    # print("params:", list(model.parameters())[0][0][:5])
    params = list(model.parameters())[0]
    print("param shape:", params.shape)
    print("first few values:", params.view(-1)[:5])  # 안전하게 flatten해서 확인
    optimizer=None
    # state_dict = torch.load(args.resume_path, map_location=device)
    # model.load_state_dict(state_dict)
    freeze.freeze_batchnorm_stats(model)

    # evaluate_handformer(model, val_loader, device, save_dir=args.save_dir, epoch=0, save_score=True)

    # confusion matrix plotting
    _, all_labels , all_scores = evaluate_handformer(model, val_loader, device, save_dir=args.save_dir, epoch=0, save_score=True)

    import seaborn as sns  # 좀 더 예쁘게 보려면 seaborn 사용 가능
    # pred, gt 합쳐서 unique label 얻기
    unique_labels = np.unique(np.concatenate([all_labels, all_scores]))    # confusion matrix (raw counts)
    cm = confusion_matrix(all_labels, all_scores, labels=unique_labels)

    # 정규화: 각 row를 전체로 나눠서 퍼센트
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # row-wise normalize

    # NaN 방지 (0으로 나누는 경우)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=unique_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', values_format=".1%")  # 소수점 1자리 퍼센트
    plt.title(f"Confusion Matrix (Epoch {epoch+1}) - Normalized")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"{args.png_name}_normalized.png"))
    plt.close()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
