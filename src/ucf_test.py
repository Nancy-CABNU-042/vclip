import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, args):
    
    model.to(device)
    model.eval()

    print("Eval setup: source={} semantic={} temporal={} eval_score={}".format(
        args.score_source,
        args.semantic_calib_type if args.use_semantic_calib else "identity",
        args.temporal_rescore_type if args.use_temporal_rescore else "identity",
        args.eval_score_type
    ))

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            outputs = model(visual, padding_mask, prompt_text, lengths)
            logits1 = outputs["logits1"].reshape(outputs["logits1"].shape[0] * outputs["logits1"].shape[1], outputs["logits1"].shape[2])
            logits2 = outputs["logits2"].reshape(outputs["logits2"].shape[0] * outputs["logits2"].shape[1], outputs["logits2"].shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            raw_score = outputs["raw_score"].reshape(-1)[:len_cur]
            calibrated_score = outputs["calibrated_score"].reshape(-1)[:len_cur]
            rescored_score = outputs["rescored_score"].reshape(-1)[:len_cur]
            if args.eval_score_type == "raw":
                final_score = raw_score
            elif args.eval_score_type == "calibrated":
                final_score = calibrated_score
            else:
                final_score = rescored_score

            if i == 0:
                ap1 = prob1
                ap2 = prob2
                ap_final = final_score
                #ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)
                ap_final = torch.cat([ap_final, final_score], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_final = ap_final.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()
    ap_final = ap_final.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    ROCF = roc_auc_score(gt, np.repeat(ap_final, 16))
    APF = average_precision_score(gt, np.repeat(ap_final, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)
    print("AUC_final: ", ROCF, " AP_final:", APF)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROCF, APF


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args.score_source, args.use_semantic_calib, args.semantic_calib_type, args.semantic_temperature, args.use_temporal_rescore, args.temporal_rescore_type, args.temporal_alpha, args.temporal_kernel_size, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, args)