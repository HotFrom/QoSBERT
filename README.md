A reference implementation for QoSBERT-style service‑quality prediction built on the CodeGen language‑model backbone.  The project demonstrates how to transform user–service metadata into semantic text, fine‑tune a PLM with uncertainty estimation, and reproduce the results reported in our TSC submission “QoSBERT: An Uncertainty‑Aware Approach Based on Pre‑trained Language Models for Service Quality Prediction.”

✨ Key Features

Component

Description

Semantic templating

Converts numerical user/service metadata into natural‑language descriptions compatible with PLMs.

Monte‑Carlo Dropout

Provides calibrated confidence intervals for each QoS prediction.

Multi‑view pooling

Mean + Max + Attention pooling over the top‑K transformer layers to capture rich semantics.

Federated‑ready

Code structure keeps data loaders and model separately, making later migration to FL frameworks trivial.

🗂️ Repository Layout
.
├── dataset/                  # Pre‑processed WS‑Dream splits
│   ├── data_train_0.20_simplified.txt
│   ├── data_test_0.20_simplified.txt
│   └── ...
├── saved_models/             # Checkpoints & logs (created automatically)
├── src/
│   ├── run.py                # Main training / eval driver (HF Trainer wrapper)
│   ├── data_utils.py         # JSONL → tokenised Dataset objects
│   ├── modeling_qos.py       # Gaussian head + MC‑Dropout utilities
│   └── ...
└── README.md                 # ← You are here
📦 Requirements

Package

Version tested

Python

3.10 +

PyTorch

≥ 2.2.0 (CUDA 11.8)

Transformers

≥ 4.40.0 (from source recommended)

Datasets

≥ 2.18.0

Accelerate

≥ 0.29.0
Quick‑Start Training
# four GPUs (IDs 5,6,7,4) with gradient‑accum 1 gives an effective batch of 1024
CUDA_VISIBLE_DEVICES=5,6,7,4 \
nohup python src/run.py \
  --output_dir ./saved_models/qos_codegen \
  --model_type codegen \
  --tokenizer_name /home/wangziliang/codegen \
  --model_name_or_path /home/wangziliang/codegen \
  --do_train --do_eval --do_test \
  --train_data_file dataset/data_train_0.20_simplified.txt \
  --eval_data_file  dataset/data_validation_0.20_simplified.txt \
  --test_data_file  dataset/data_test_0.20_simplified.txt \
  --epoch 20 \
  --block_size 64 \
  --train_batch_size 256 \
  --eval_batch_size 256 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 42 \
  > qos_train_logmlp20.txt 2>&1 &
> 

If you find this repo useful, please cite:

@article{wang2025qosbert,
  title   = {QoSBERT: An Uncertainty-Aware Approach Based on Pre-trained Language Models for Service Quality Prediction},
  author  = {Wang, Ziliang and Zhang, Xiaohong and Li, Ze Shi and Yan, Meng},
  journal = {IEEE Transactions on Services Computing},
  year    = {2025},
  volume  = {18},
  number  = {6},
  pages   = {4096--4108},
  doi     = {10.1109/TSC.2025.3620138}
}

pdf:https://drive.google.com/file/d/1HE4nF_sYEWx8WSvFxedx8TMbVWy6higK/view?usp=drive_link
