import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ['true', '1', 'yes', 'y', 't']:
        return True
    if value in ['false', '0', 'no', 'n', 'f']:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=str2bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

parser.add_argument('--score-source', '--score_source', default='fused', choices=['logits1_only', 'logits2_only', 'fused'])

parser.add_argument('--use-semantic-calib', '--use_semantic_calib', default=False, type=str2bool)
parser.add_argument('--semantic-calib-type', '--semantic_calib_type', default='identity', choices=['identity', 'temperature'])
parser.add_argument('--semantic-temperature', '--semantic_temperature', default=1.0, type=float)

parser.add_argument('--use-temporal-rescore', '--use_temporal_rescore', default=False, type=str2bool)
parser.add_argument('--temporal-rescore-type', '--temporal_rescore_type', default='identity', choices=['identity', 'ema', 'conv1d'])
parser.add_argument('--temporal-alpha', '--temporal_alpha', default=0.7, type=float)
parser.add_argument('--temporal-kernel-size', '--temporal_kernel_size', default=3, type=int)

parser.add_argument('--eval-score-type', '--eval_score_type', default='rescored', choices=['raw', 'calibrated', 'rescored'])
