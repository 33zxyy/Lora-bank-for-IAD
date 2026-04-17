import sys
import os
sys.path.append(os.getcwd())
from share import *
from utils.util import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.mvtecad_dataloader import MVTecDataset_cad
from cdm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


def main(args):
    setup_seed(args.seed)

    log_name = f'mvtec_setting{args.setting}'

    model = create_model('models/cdad_mvtec.yaml').cpu()

    weights = load_state_dict(args.resume_path, location='cpu')
    # Keep backward-compatible behavior for base initialization.
    if args.start_task == 0 and args.resume_path.endswith('base.ckpt'):
        # For fresh starts, allow changing LoRA rank without shape mismatch by
        # skipping persisted LoRA expert tensors from base checkpoints.
        model_state = model.state_dict()
        select_weights = {}
        for key, value in weights.items():
            if key not in model_state:
                continue
            if '.experts.' in key or '.expert_gates.' in key:
                continue
            if model_state[key].shape != value.shape:
                continue
            select_weights[key] = value
        missing, unexpected = model.load_state_dict(select_weights, strict=False)
        print("missing:", len(missing), "unexpected:", len(unexpected))
    else:
        model.load_state_dict(weights, strict=False)

    model.learning_rate = args.learning_rate

    train_dataset, task_num = MVTecDataset_cad('train', args.data_path, args.setting)
    test_dataset, _ = MVTecDataset_cad('test', args.data_path, args.setting)

    if args.start_task < 0 or args.start_task >= task_num:
        raise ValueError(f"start_task must be in [0, {task_num - 1}], got {args.start_task}")

    for i in range(args.start_task, task_num):

        model.set_log_name(log_name + f'/task{i}')

        ckpt_callback_val = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f'./incre_val/{log_name}/',
            filename=f'task{i}_best',
            mode='max')

        trainer = pl.Trainer(gpus=1, precision=32,
                    callbacks=[ckpt_callback_val, ],
                    num_sanity_val_steps=0,
                    accumulate_grad_batches=1,     # Do not change!!!
                    max_epochs=args.max_epoch,
                    check_val_every_n_epoch=args.check_v,
                    enable_progress_bar=False
                    )


        train_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=True)
        gpm_dataloader = DataLoader(train_dataset[i], num_workers=8, batch_size=args.gpm_batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset[i], num_workers=8, batch_size=args.batch_size, shuffle=False)

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

        best_model_path = ckpt_callback_val.best_model_path
        if not best_model_path:
            raise RuntimeError(f"No best checkpoint found for task {i}.")
        print(f"[Task {i}] load best checkpoint for test: {best_model_path}")
        model.load_state_dict(load_state_dict(best_model_path, location='cuda'), strict=False)

        # test is used to process gradient projection
        trainer.test(model, dataloaders=gpm_dataloader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CDAD")

    parser.add_argument("--resume_path", default='./models/base.ckpt')

    parser.add_argument("--data_path", default="./data/mvtec_anomaly_detection", type=str)

    parser.add_argument("--setting", default=1, type=int)

    parser.add_argument("--start_task", default=0, type=int,
                        help="Task index to start/resume from, e.g., 1 means start from task1.")

    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--batch_size", default=12, type=int)

    parser.add_argument("--gpm_batch_size", default=1, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)

    parser.add_argument("--max_epoch", default=500, type=int)

    parser.add_argument("--check_v", default=25, type=int)

    args = parser.parse_args()

    main(args)
