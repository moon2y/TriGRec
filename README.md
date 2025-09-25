# TriGRec

TriGRec is a two-stage recommendation pipeline:
1) Learn user representations (specific/common/cross) from multi-domain sequences,
2) Train/evaluate the recommender that consumes those embeddings.

-------------------------------------------------------------------------------
## Repository Structure
-------------------------------------------------------------------------------

TriGRec/
├─ data/                                  # your pickled datasets
├─ state_dict/
│  ├─ rep/                                # user-representation checkpoints
│  └─ rec/                                # recommender checkpoints
├─ recommendation_model/
│  ├─ train_eval.py                       # entry (uses config.py)
│  ├─ data.py                             # MakeSequenceDataSet, BERTRecDataSet
│  ├─ model.py                            # BERT4Rec & CrossAttention gating
│  └─ config.py                           # BASE settings + --override parser
└─ user_representation_module/
   ├─ train.py                            # entry (uses config.py)
   ├─ inference.py                        # embedding export (imports BERT4Rec)
   ├─ data.py                             # BERTRecDataSet, InferenceDataset
   ├─ model.py                            # BERT4Rec, GRL, Discriminator, losses
   └─ config.py                           # TRAIN/INFER settings + --override


-------------------------------------------------------------------------------
## Data Format
-------------------------------------------------------------------------------
- Pickled dicts: user_id -> [item_id, item_id, ...]
- Example files (used by defaults):
   - data/food_all.pkl
   - data/kitchen_train.pkl
   - data/cross_k_train.pkl
   - data/food_train.pkl
   - data/kitchen_all.pkl
   - data/cross_f_train.pkl


-------------------------------------------------------------------------------
## Quickstart (run from repository root with module mode)
-------------------------------------------------------------------------------
1) User Representation — Training
   Learns Specific/Common/Cross encoders and saves to state_dict/rep/...

```bash
python -m user_representation_module.train
```

   Tweak a few settings on the fly:
```bash
python -m user_representation_module.train --override "device=cuda:0,lr=0.0005,batch_size=64,num_epochs=50"
```

2) User Representation — Embedding Export
   Loads trained encoders and exports spe/cross embeddings to .pkl and .npy.

```bash
python -m user_representation_module.inference
```

   Choose epoch and domain for 'specific' embeddings:
```bash
python -m user_representation_module.inference --override "EPOCH_TO_LOAD=80,SPE_DOMAIN=B"
```

   Outputs:
   user_representation_module/embeddings/<domain>/
     spe_emb_ep{EPOCH}_{A|B}.pkl | .npy
     cross_emb_ep{EPOCH}_{A|B}.pkl | .npy

3) Recommender — Train/Eval
   Uses sequences + exported embeddings to train BERT4Rec with cross-attention.

```bash
python -m recommendation_model.train_eval
```

   Example overrides:
```bash
python -m recommendation_model.train_eval --override "device=cuda:0,batch_size=1024,spe_emb_path=user_representation_module/embeddings/food/spe_emb_ep100_A.npy,cross_emb_path=user_representation_module/embeddings/food/cross_emb_ep100_A.npy"
```

   Checkpoint:
   state_dict/rec/food/maxlen10.pt

-------------------------------------------------------------------------------
## Configuration
-------------------------------------------------------------------------------
Each module maintains defaults in config.py:
- user_representation_module/config.py:
  TRAIN (for training) and INFER (for embedding export)
- recommendation_model/config.py:
  BASE (for train/eval)

Override any key at runtime:
```bash
python -m user_representation_module.train --override "device=cuda:0,lr=0.0005"
python -m user_representation_module.inference --override "EPOCH_TO_LOAD=100,SPE_DOMAIN=A"
python -m recommendation_model.train_eval --override "seed=2025,batch_size=1024"
```

-------------------------------------------------------------------------------
## End-to-End Pipeline
-------------------------------------------------------------------------------
1. Train user encoders (user_representation_module/train.py)
   - Learns specific/common/cross encoders (plus prediction/discriminator heads).
   - Saves to state_dict/rep/<domain>/...pt at intervals.

2. Export embeddings (user_representation_module/inference.py)
   - Loads encoders from state_dict/rep/...
   - Exports spe and cross user embeddings to .pkl and .npy.

3. Train & evaluate recommender (recommendation_model/train_eval.py)
   - Trains BERT4Rec with cross-attention gating.
   - Saves model to state_dict/rec/...
