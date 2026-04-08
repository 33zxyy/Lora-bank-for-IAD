# One-for-More 新人上手指南

本文面向第一次接触本仓库的同学，帮助你先建立“全局地图”，再进入训练/评测与核心算法细节。

## 1. 仓库在做什么（一句话）

这是一个基于 Stable Diffusion 1.5 的**持续异常检测**项目：
- 通过扩散模型重建输入；
- 通过特征差异生成 anomaly map；
- 通过 GPM + iSVD 的梯度投影缓解持续学习中的灾难性遗忘。

## 2. 代码结构总览（按“职责”划分）

### 2.1 入口层（你最先会运行的文件）

- `README.md`：安装、数据组织、训练与测试命令说明。
- `build_base_model.py`：把 SD1.5 + VAE 权重拼装成项目可训练的 `models/base.ckpt`。
- `scripts/train_mvtec.py` / `scripts/train_visa.py`：按 setting 多 task 持续训练。
- `scripts/test_mvtec.py` / `scripts/test_visa.py`：指定 setting + task 做评测并导出可视化和指标。

> 建议：新人第一周先把这 5 个文件完整读一遍，就能跑通“从构建到评测”的主流程。

### 2.2 模型层（持续学习方法核心）

- `cdm/model.py`：统一模型创建与 checkpoint 加载。
- `cdm/sd_amn.py`：在 Latent Diffusion 基础上实现训练/验证主逻辑、重建与 anomaly map 评估。
- `cdm/gpm.py`：实现 GPM + iSVD（通过 hook 收集激活，SVD 更新投影子空间，训练时投影梯度）。
- `cdm/amn.py` / `cdm/vit.py` / `cdm/mha.py`：异常掩码网络、注意力与相关结构实现。
- `cdm/param.py`：定义哪些参数允许训练、哪些层参与投影保护。

### 2.3 数据层（持续任务切分 + 合成异常）

- `data/mvtecad_dataloader.py`：按 setting 切 task，训练时注入 NSA patch 异常，输出 `jpg/txt/hint/mask`。
- `data/visa_dataloader.py`：VisA 对应逻辑，流程与 MVTec 版本一致。
- `data/nsa.py`：NSA 异常 patch 生成函数。
- `data/MVTec-AD/*`、`data/VisA/*`：任务切分索引（json/csv）与数据目录。

### 2.4 配置与工具层

- `models/cdad_mvtec.yaml` / `models/cdad_visa.yaml`：模型结构、层选择、距离度量等关键配置。
- `share.py` + `config.py`：全局开关（如节省显存模式）与启动时设置。
- `utils/eval_helper.py`：像素级/图像级指标计算与汇总。
- `utils/util.py`：anomaly map 计算、日志图像保存、随机种子等常用函数。
- `ldm/` 与 `taming/`：来自 latent diffusion / taming-transformers 的底层组件。

## 3. “跑起来”时最重要的 6 个概念

1. **base.ckpt 的来源**：
   训练前先执行 `build_base_model.py`，把预训练 SD 权重映射到本项目模型参数空间。

2. **setting / task 机制**：
   setting 决定“如何增量切分类别”；每个 task 顺序训练，模拟类别逐步到达。

3. **hint 与重建**：
   数据集返回 `hint` 作为条件输入；模型生成/重建后与输入在 backbone 特征空间比较得到异常图。

4. **验证指标触发 best ckpt**：
   训练脚本中按 `val_acc`（组合指标）保存 `task{i}_best.ckpt`。

5. **test 阶段的双重作用**：
   训练脚本中的 `trainer.test(...)` 不只是“评估”，更是触发 GPM/iSVD 更新投影空间的重要步骤。

6. **最终输出位置**：
   - checkpoint：`incre_val/...`
   - 可视化：`log_image/` 或 `TEST/.../image*`
   - 指标 csv：`logs/...` 或 `TEST/.../log/...`

## 4. 新人建议的学习路径（按周）

### 第 0 阶段：先能跑（1 天）
- 按 `README.md` 完成环境、数据目录、基础模型构建。
- 先跑一个最小 setting + task 的训练和测试，确认路径与依赖无误。

### 第 1 阶段：读主链路（2~3 天）
- 按顺序阅读：
  1) `scripts/train_*.py`
  2) `data/*_dataloader.py`
  3) `cdm/model.py`
  4) `cdm/sd_amn.py`
- 目标：能画出“batch -> 模型 -> 指标 -> ckpt”的数据流图。

### 第 2 阶段：吃透持续学习机制（3~5 天）
- 精读 `cdm/gpm.py` + `cdm/param.py`。
- 重点搞清：
  - hook 采集哪些层；
  - SVD 如何截断；
  - 训练时梯度如何投影并保护旧任务子空间。

### 第 3 阶段：可控实验（1 周）
- 固定随机种子，改 1 个变量做小实验（如 layer 选择、distance、batch size）。
- 把每次实验记录成“配置-结果-结论”三列表，避免重复试错。

## 5. 工程实践建议

- **先小后大**：先用单 task 验证流程，再扩到全 setting。
- **保持可复现**：每次改动记录命令、seed、配置、commit id。
- **读代码抓主干**：先看 `scripts/` 和 `cdm/`，再深入 `ldm/` 细节。
- **注意目录大小写**：仓库中同时出现 `TEST/`、`log_image/` 等目录，脚本中路径大小写需严格一致。
- **先确认数据索引文件**：很多报错来源于 `data/MVTec-AD/<setting>/...json` 或 `data/VisA/<setting>/...csv` 缺失。

---

如果你是导师，可把本文当成 onboarding 模板：
- 第一次 1:1：讲结构图；
- 第二次 1:1：讲 gpm.py 的投影逻辑；
- 第三次 1:1：review 复现实验日志。
