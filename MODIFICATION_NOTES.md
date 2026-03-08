# Modification Notes

## Changed Files

- `src/model.py`
  - Added `SemanticCalibration` module with `identity` and `temperature` modes.
  - Added `TemporalRescorer` module with `identity`, `ema`, and `conv1d` modes.
  - Added score conversion and fusion logic:
    - `score_1 = sigmoid(logits1)`
    - `score_2 = max(logits2, dim=-1)`
    - source selection via `score_source` (`logits1_only`, `logits2_only`, `fused`).
  - Applied post-score pipeline in order:
    - `raw_score -> calibrated_score -> rescored_score`.
  - Updated `forward` to return a dictionary including original logits and all new scores.

- `src/ucf_option.py`, `src/xd_option.py`
  - Added CLI flags for score source, semantic calibration, temporal re-scoring, and evaluation score type.
  - Added underscore aliases to support commands like `--score_source`.

- `src/ucf_train.py`, `src/xd_train.py`
  - Updated model output handling to read from output dict (`logits1`, `logits2`, `text_features_ori`).
  - Kept original loss structure unchanged.
  - Added startup logging for score source / semantic mode / temporal mode.
  - Passed new CLI-controlled module settings into `CLIPVAD` constructor.

- `src/ucf_test.py`, `src/xd_test.py`
  - Updated inference to consume model output dict.
  - Added selectable final evaluation score via `--eval_score_type` (`raw`, `calibrated`, `rescored`).
  - Preserved original branch metrics (`AUC1/AP1`, `AUC2/AP2`) and added final score metrics.
  - Added eval-time logging for score source / semantic mode / temporal mode / eval score type.

## Default Behavior

- Default score source is fused (`score_source=fused`).
- Semantic calibration defaults to identity behavior (`use_semantic_calib=False`, type `identity`).
- Temporal re-scoring defaults to identity behavior (`use_temporal_rescore=False`, type `identity`).
- Evaluation uses `rescored` by default, which equals near-original output when both modules are disabled.

## Example Commands

```bash
python ucf_train.py --score_source fused --use_semantic_calib False --use_temporal_rescore False
python ucf_train.py --score_source fused --use_semantic_calib True --semantic_calib_type temperature --semantic_temperature 1.0 --use_temporal_rescore False
python ucf_train.py --score_source fused --use_semantic_calib False --use_temporal_rescore True --temporal_rescore_type ema --temporal_alpha 0.7
python ucf_train.py --score_source fused --use_semantic_calib True --semantic_calib_type temperature --semantic_temperature 1.0 --use_temporal_rescore True --temporal_rescore_type ema --temporal_alpha 0.7
python ucf_train.py --score_source logits1_only --use_semantic_calib False --use_temporal_rescore False
python ucf_train.py --score_source logits2_only --use_semantic_calib False --use_temporal_rescore False
```


## Targeted Fix Pass

### Score Fusion Scaling Fix
- Updated score conversion in `src/model.py` to avoid fusion domination by branch-2 scale:
  - from: `score_2 = torch.max(logits2, dim=-1).values`
  - to: `score_2 = torch.sigmoid(torch.max(logits2, dim=-1).values)`
- Kept fusion formula unchanged: `raw_score = 0.5 * score_1 + 0.5 * score_2`.
- Added optional debug-only one-time score statistics logging for both `score_1` and `score_2` (min/max/mean/std), controlled by environment variable `VADCLIP_DEBUG=1`.

### Boolean Parsing Fix
- Replaced unsafe `argparse` boolean parsing (`type=bool`) with a safe `str2bool` helper in:
  - `src/ucf_option.py`
  - `src/xd_option.py`
- Applied to:
  - `--use-checkpoint`
  - `--use-semantic-calib` / `--use_semantic_calib`
  - `--use-temporal-rescore` / `--use_temporal_rescore`

### Example (explicit True/False)
```bash
python ucf_train.py --score_source fused --use_semantic_calib True --semantic_calib_type temperature --semantic_temperature 1.0 --use_temporal_rescore False
```
