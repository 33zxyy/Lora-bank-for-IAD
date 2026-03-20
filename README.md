# [CVPR 2025] One-for-More
**One-for-More: Continual Diffusion Model for Anomaly Detection**


[Xiaofan Li<sup>1</sup>](https://scholar.google.com/citations?user=WFppW4IAAAAJ&hl=zh-CN),
[Xin Tan<sup>1</sup>](https://scholar.google.com/citations?user=UY4NCdcAAAAJ&hl=zh-CN),
Zhuo Chen<sup>2</sup>,
[Zhizhong Zhang<sup>1</sup>](https://scholar.google.com/citations?user=CXZciFAAAAAJ&hl=zh-CN),
Ruixin Zhang<sup>4</sup>,
Rizen Guo<sup>5</sup>,
Guannan Jiang<sup>6</sup>,
Yulong Chen<sup>3</sup>,
[Yanyun Qu<sup>2</sup>](https://scholar.google.com/citations?user=idiP90sAAAAJ&hl=zh-CN),
[Lizhuang Ma<sup>1,3</sup>](https://scholar.google.com/citations?user=yd58y_0AAAAJ&hl=zh-CN),
[Yuan Xie<sup>1</sup>](https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=zh-CN)

<sup>1</sup>East China Normal University,
<sup>2</sup>Xiamen University, <sup>3</sup>Shanghai Jiao Tong University,
<sup>4</sup>Youtu Lab, Tencent, <sup>5</sup>WeChatPay Lab33, Tencent,<sup>6</sup>CATL,

[[📖 Paper](https://arxiv.org/abs/2502.19848)] 
<!-- [[`Project Page`](https://lewandofskee.github.io/projects/diad/)] -->


## 💥 News
- **[2025.03.17]** We release the code for training and test 🚀
- **[2025.02.27]** We release the [arXiv paper](hhttps://arxiv.org/abs/2502.19848) 🚀
- **[2025.02.28]** The paper is accepted by CVPR 2025 🎉

## Abstract
With the rise of generative models, there is a growing interest in unifying all tasks within a generative framework. Anomaly detection methods also fall into this scope and utilize diffusion models to generate or reconstruct normal samples when given arbitrary anomaly images. However, our study found that the diffusion model suffers from severe **faithfulness hallucination** and **catastrophic forgetting**, which can't meet the unpredictable pattern increments. To mitigate the above problems, we propose a continual diffusion model that uses gradient projection to achieve stable continual learning. Gradient projection deploys a regularization on the model updating by modifying the gradient towards the direction protecting the learned knowledge. But as a double-edged sword, it also requires huge memory costs brought by the Markov process. Hence, we propose an iterative singular value decomposition method based on the transitive property of linear representation, which consumes tiny memory and incurs almost no performance loss. Finally, considering the risk of ``over-fitting'' to normal images of the diffusion model, we propose an anomaly-masked network to enhance the condition mechanism of the diffusion model. For continual anomaly detection, ours achieves first place in 17/18 settings on MVTec and VisA.

## 1. Installation

First create a new conda environment

    $ conda create -n cdad python==3.8.5
    $ conda activate cdad
    $ bash install.bash
## 2.Dataset
### 2.1 MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move them to `./data/mvtec_anomaly_detection/`. The MVTec-AD dataset directory should be as follows. 

```
|-- data
    |-- mvtec_anomaly_detection
        |-- bottle
            |-- ground_truth
                |-- broken_large
                    |-- 000_mask.png
                |-- broken_small
                    |-- 000_mask.png
                |-- contamination
                    |-- 000_mask.png
            |-- test
                |-- broken_large
                    |-- 000.png
                |-- broken_small
                    |-- 000.png
                |-- contamination
                    |-- 000.png
                |-- good
                    |-- 000.png
            |-- train
                |-- good
                    |-- 000.png
```

### 2.2 VisA
- **Create the VisA dataset directory**. Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file and move them to `./VisA/`. The VisA dataset directory should be as follows. 

```
|-- data
    |-- VisA
        |-- candle
            |-- Data
                |-- Images
                    |-- Anomaly
                        |-- 000.JPG
                    |-- Normal
                        |-- 0000.JPG
                |-- Masks
                    |--Anomaly 
                        |-- 000.png        
```


## 3. Build the model
First download the checkpoint of AutoEncoder and diffusion model, we use the pre-trained stable diffusion v1.5.

    $ wget https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
    $ unzip kl-f8.zip
    $ wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
    

Then run the code to get the output model `./models/base.ckpt`.

    $ python build_base_model.py


## 4. Train
The incremental settings for the MVTec and VisA datasets are shown in the table.
| Dataset | Setting ID |   Incremental setting   |
|:-------:|:--:|:-----------------------:|
|  MVTec  |  1 |    14 - 1 with 1 Step   |
|  MVTec  |  2 |    10 - 5 with 1 Step   |
|  MVTec  |  3 |    3 ✖️ 5 with 5 Steps   |
|  MVTec  |  4 | 10 - 1 ✖️ 5 with 5 Steps |
|   VisA  |  1 |    11 - 1 with 1 Step   |
|   VisA  |  2 |    8 - 4 with 1 Step    |
|   VisA  |  3 |   8 - 1 ✖️ with 4 Steps   |

The training scripts are as follows:

    $ python scripts/train_mvtec.py --setting [ID]
    $ python scripts/train_visa.py --setting [ID]

The images are saved under `./log_image/`
The training logs are saved under `./log`

## 5. Test
The trained checkpoints of `MVTec, setting [s], task [t]` are saved under `./incre_val/mvtec_setting[s]/task[t]_best.ckpt`. For evaluation and visualization, run the following code:

    $ python scripts/test_mvtec.py --setting [s] --task [t]
    $ python scripts/test_visa.py --setting [s] --task [t]

The test results are saved under `./Test/`


## 6. Experimental Details Checklist (for papers)
Besides the pre-trained Stable Diffusion checkpoint, the following values can be directly reported from this repo.

### 6.1 Data protocol (filled)
- Dataset and root path:
  - MVTec-AD: `./data/mvtec_anomaly_detection`
  - VisA: `./data/VisA`
- Incremental setting id: `--setting` (see the setting table in Section 4).
- Test checkpoint path format:
  - MVTec: `./incre_val/mvtec_setting[s]/task[t]_best.ckpt`
  - VisA: `./incre_val/visa_setting[s]/task[t]_best.ckpt`

### 6.2 Optimization and training schedule (filled)
- Random seed: `--seed=1`.
- Batch size: `--batch_size=12`.
- GPM statistics batch size: `--gpm_batch_size=1`.
- Learning rate: `--learning_rate=1e-5`.
- Max epoch: `--max_epoch=500`.
- Validation frequency:
  - MVTec: `--check_v=25`
  - VisA: `--check_v=5`
- Training precision/device: PyTorch Lightning `gpus=1`, `precision=32`.

### 6.3 Diffusion/CDAD model hyperparameters (filled)
- Diffusion process: `timesteps=1000`, `linear_start=0.00085`, `linear_end=0.012`, `num_timesteps_cond=1`.
- Model scale: `image_size=64`, `channels=4`, `scale_factor=0.18215`.
- Feature distance: `distance='eucl'`.
- Feature layers for anomaly scoring:
  - MVTec: `layers=[1,2,3]`
  - VisA: `layers=[0,2,4]`
- AMN/control settings: `neighbor_size=[7,7]`, `hint_channels=3`, `model_channels=320`.

### 6.4 Continual additive LoRA expert settings (filled)
- `lora_rank=4`
- `lora_alpha=1.0`
- `orth_lambda=1e-4`
- `novelty_threshold=0.2`
- Historical experts are frozen during continual updates; inference uses `fuse_lora_experts()` (if available) before testing.

### 6.5 Evaluation protocol (filled)
- Checkpoint selection: `ModelCheckpoint(monitor='val_acc', mode='max', filename='task{i}_best')`.
- Metrics reported in test scripts: APSP-AUC and PRO-AUC.
- Anomaly map post-processing: Gaussian smoothing `sigma=5`.
- Suggested reporting format: per-task + average across tasks (and optionally forgetting/BWT).

### 6.6 Recommended table fields in your paper
`dataset | setting id | task id | seed | lr | batch size | gpm batch size | max epoch | check_v | LoRA rank | lora alpha | orth lambda | novelty threshold | checkpoint rule | metrics`.



## Citation

```
@article{li2025one,
  title={One-for-More: Continual Diffusion Model for Anomaly Detection},
  author={Li, Xiaofan and Tan, Xin and Chen, Zhuo and Zhang, Zhizhong and Zhang, Ruixin and Guo, Rizen and Jiang, Guanna and Chen, Yulong and Qu, Yanyun and Ma, Lizhuang and others},
  journal={arXiv preprint arXiv:2502.19848},
  year={2025}
}
```
## Acknowledgements
We thank the great works [DiAD](https://github.com/lewandofskee/DiAD), [GPM](https://github.com/sahagobinda/GPM) and [ControlNet](https://github.com/lllyasviel/ControlNet) for providing assistance for our research.
