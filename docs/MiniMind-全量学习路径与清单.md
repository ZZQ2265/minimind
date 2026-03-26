# MiniMind 全量学习路径与核对清单

> **面向目标**：按本清单逐项完成，可系统覆盖本仓库**模型结构、数据管线、各阶段训练、推理与工程化**的全部内容。
> **硬件说明**：4060 / 云 GPU 均可；具体 batch、序列长以 `README.md` 与脚本参数为准。

---

## 🎯 学习路线总览

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 0: 环境准备 → 安装依赖、理解目录结构                          │
├─────────────────────────────────────────────────────────────────┤
│  阶段 1: 体验先行 → 下载模型、运行推理、建立直觉                     │
├─────────────────────────────────────────────────────────────────┤
│  阶段 2: 理解模型 → 模型架构、分词器、数据处理                       │
├─────────────────────────────────────────────────────────────────┤
│  阶段 3: 训练实践 → 预训练 → SFT → LoRA → 强化学习                 │
├─────────────────────────────────────────────────────────────────┤
│  阶段 4: 工程化 → API服务、模型转换、评测                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 使用方式

- **顺序**：按阶段 0→4 推进；同一阶段内可并行阅读「理论 / 代码 / README 对应小节」
- **勾选**：每完成一项，将 `[ ]` 改为 `[x]`
- **权威文档**：以仓库根目录 `README.md` 为准；本清单与源码文件一一对应

---

## 📋 前置知识要求

| 知识领域 | 要求程度 | 说明 |
|----------|----------|------|
| Python 编程 | 熟练 | 能够阅读和理解 PyTorch 代码 |
| PyTorch 框架 | 中等 | 理解张量、自动求导、DataLoader |
| 深度学习基础 | 中等 | 神经网络、反向传播、优化器 |
| Transformer | 了解 | Attention 机制、位置编码概念 |

---

## 阶段 0：环境与仓库全局

> **目标**：搭建开发环境，了解项目整体结构

| 序号 | 内容 | 核对 |
|------|------|------|
| 0.1 | 阅读 `README.md` 前言、项目目标、模型列表与更新日志 | [ ] |
| 0.2 | 安装依赖：`pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple` | [ ] |
| 0.3 | 验证 CUDA：运行 `python -c "import torch; print(torch.cuda.is_available())"` | [ ] |
| 0.4 | 理解目录结构（见下方表格） | [ ] |
| 0.5 | 阅读 `dataset/dataset.md`，知悉数据集下载地址与放置位置 | [ ] |

### 目录结构说明

```
📁 minimind/
├── 📁 model/                    # 模型定义
│   ├── model_minimind.py        # ← 核心：Transformer 架构
│   ├── model_lora.py            # LoRA 低秩适配
│   ├── tokenizer.json           # 分词器词表
│   └── tokenizer_config.json    # 分词器配置
├── 📁 dataset/                  # 数据处理
│   ├── lm_dataset.py            # ← 数据加载与预处理
│   └── dataset.md               # 数据说明文档
├── 📁 trainer/                  # 训练脚本
│   ├── trainer_utils.py         # 训练公共工具
│   ├── train_pretrain.py        # 预训练
│   ├── train_full_sft.py        # 监督微调
│   ├── train_lora.py            # LoRA 微调
│   ├── train_dpo.py             # DPO 强化学习
│   ├── train_ppo.py             # PPO 强化学习
│   ├── train_grpo.py            # GRPO 强化学习
│   ├── train_spo.py             # SPO 强化学习
│   ├── train_reason.py          # 推理模型训练
│   ├── train_distillation.py    # 知识蒸馏
│   └── train_tokenizer.py       # 分词器训练
├── 📁 scripts/                  # 工具脚本
│   ├── web_demo.py              # Streamlit Web 界面
│   ├── serve_openai_api.py      # OpenAI 兼容 API 服务
│   ├── chat_openai_api.py       # API 客户端示例
│   └── convert_model.py         # 模型格式转换
├── 📁 docs/                     # 文档
├── eval_llm.py                  # 模型推理入口
├── requirements.txt             # 依赖列表
└── README.md                    # 项目文档
```

---

## 阶段 1：推理先行（建立端到端直觉）

> **目标**：先体验模型效果，建立直观感受，验证环境配置

> 💡 **为什么先下载模型？**
> - ✅ 先看效果，再学原理
> - ✅ 验证环境配置是否正确
> - ✅ 降低入门门槛（不需要等2小时训练）
> - ✅ 提供对比基准（知道训练好的模型应该是什么效果）

| 序号 | 内容 | 核对 |
|------|------|------|
| 1.1 | 下载预训练模型：`git clone https://huggingface.co/jingyaogong/MiniMind2` | [ ] |
| 1.2 | 运行推理：`python eval_llm.py --load_from ./MiniMind2` | [ ] |
| 1.3 | **精读** `eval_llm.py`：理解模型加载与推理流程 | [ ] |
| 1.4 | 理解命令行参数：`--weight`、`--temperature`、`--top_p`、`--max_new_tokens` | [ ] |
| 1.5 | （可选）启动 Web 界面：`streamlit run scripts/web_demo.py` | [ ] |
| 1.6 | （可选）尝试 ollama / vllm 等第三方推理框架 | [ ] |

### eval_llm.py 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--load_from` | 模型路径（model=原生torch权重，其他=transformers格式） | model |
| `--weight` | 权重名称前缀（pretrain/full_sft/dpo/ppo_actor/grpo等） | full_sft |
| `--temperature` | 生成温度（越高越随机） | 0.85 |
| `--top_p` | nucleus采样阈值 | 0.85 |
| `--max_new_tokens` | 最大生成长度 | 8192 |
| `--lora_weight` | LoRA权重名称 | None |
| `--inference_rope_scaling` | 启用RoPE长度外推 | False |

---

## 阶段 2：理解模型架构与数据

> **目标**：深入理解 LLM 的核心组件

### 2.1 模型架构 (`model/model_minimind.py`) ⭐ 必精读

| 序号 | 内容 | 核对 |
|------|------|------|
| 2.1.1 | 理解 `MiniMindConfig`：`hidden_size`、`num_hidden_layers`、GQA | [ ] |
| 2.1.2 | 理解 **RMSNorm** 归一化（与 LayerNorm 的区别） | [ ] |
| 2.1.3 | 理解 **RoPE 旋转位置编码**（频率预计算、位置外推） | [ ] |
| 2.1.4 | 理解 **SwiGLU 激活函数**（与 ReLU 的区别） | [ ] |
| 2.1.5 | 理解 **Attention** 机制（Q/K/V、KV-Cache、GQA） | [ ] |
| 2.1.6 | 理解 **Transformer Block** 结构 | [ ] |
| 2.1.7 | 理解 **MoE（混合专家）**：`aux_loss` 负载均衡 | [ ] |
| 2.1.8 | 理解 `MiniMindForCausalLM` 的 `forward` 与输出 | [ ] |

### 2.2 分词器 (`model/tokenizer.*`)

| 序号 | 内容 | 核对 |
|------|------|------|
| 2.2.1 | 阅读 README「Tokenizer」小节：词表大小 6400 的设计取舍 | [ ] |
| 2.2.2 | 浏览 `tokenizer.json` 与 `tokenizer_config.json` | [ ] |
| 2.2.3 | （可选）阅读 `trainer/train_tokenizer.py`：自训练分词器流程 | [ ] |

### 2.3 数据处理 (`dataset/lm_dataset.py`) ⭐ 必精读

| 序号 | 内容 | 核对 |
|------|------|------|
| 2.3.1 | 理解 `PretrainDataset`：`text` → `input_ids`/`labels` | [ ] |
| 2.3.2 | 理解 `SFTDataset`：`conversations` 格式与 `apply_chat_template` | [ ] |
| 2.3.3 | 理解 `generate_labels`：只计算 assistant 段的 loss | [ ] |
| 2.3.4 | 理解 padding 与 `-100` 掩码的作用 | [ ] |
| 2.3.5 | 理解 `DPODataset`：`chosen`/`rejected` 格式 | [ ] |
| 2.3.6 | 理解 `RLAIFDataset`：强化学习数据格式 | [ ] |

---

## 阶段 3：训练实践

> **目标**：从0开始训练自己的模型

### 3.0 训练公共基础设施 (`trainer/trainer_utils.py`)

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.0.1 | 理解 `get_lr`：学习率调度 | [ ] |
| 3.0.2 | 理解 `init_distributed_mode`：DDP 多卡训练 | [ ] |
| 3.0.3 | 理解断点续训：`--from_resume 1` 与 `./checkpoints/*_resume.pth` | [ ] |
| 3.0.4 | 理解 wandb / SwanLab 训练可视化 | [ ] |
| 3.0.5 | 理解混合精度、梯度裁剪、梯度累积 | [ ] |

### 3.1 预训练 (Pretrain) - `trainer/train_pretrain.py`

> **目的**：让模型学习知识（无监督学习，词语接龙）

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.1.1 | **精读** `train_pretrain.py`：参数列表、训练循环、loss（含 aux_loss） | [ ] |
| 3.1.2 | 下载 `pretrain_hq.jsonl` 到 `./dataset/` 目录 | [ ] |
| 3.1.3 | 运行预训练：`python train_pretrain.py`（或短跑验证） | [ ] |
| 3.1.4 | 理解输出权重命名：`pretrain_{hidden_size}.pth` | [ ] |

### 3.2 监督微调 (SFT) - `trainer/train_full_sft.py`

> **目的**：让模型学会对话（指令微调）

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.2.1 | **精读** `train_full_sft.py`：与预训练权重的衔接 | [ ] |
| 3.2.2 | 下载 `sft_mini_512.jsonl` 到 `./dataset/` 目录 | [ ] |
| 3.2.3 | 运行 SFT：`python train_full_sft.py` | [ ] |
| 3.2.4 | 用 `python eval_llm.py --weight full_sft` 验证对话效果 | [ ] |

### 3.3 LoRA 微调 - `trainer/train_lora.py`

> **目的**：参数高效微调，适合垂直领域适配

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.3.1 | **精读** `model/model_lora.py`：LoRA 如何挂到线性层 | [ ] |
| 3.3.2 | **精读** `train_lora.py`：训练流程 | [ ] |
| 3.3.3 | 准备 `lora_identity.jsonl` 或自有数据 | [ ] |
| 3.3.4 | 运行 LoRA：`python train_lora.py` | [ ] |
| 3.3.5 | 用 `python eval_llm.py --lora_weight lora_xxx` 测试效果 | [ ] |

### 3.4 知识蒸馏 - `trainer/train_distillation.py`

> **目的**：用大模型的知识指导小模型

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.4.1 | **精读** `train_distillation.py`：软标签与 KL-Loss | [ ] |
| 3.4.2 | 理解白盒蒸馏与黑盒蒸馏的区别 | [ ] |

### 3.5 推理模型训练 - `trainer/train_reason.py`

> **目的**：训练具备思维链能力的推理模型

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.5.1 | **精读** `train_reason.py`：思考标签、标签位置 loss | [ ] |
| 3.5.2 | 下载 `r1_mix_1024.jsonl` 数据集 | [ ] |
| 3.5.3 | 理解推理模板：`<think\>` 思考过程 `<answer\>` 最终回答 | [ ] |

### 3.6 强化学习 - DPO

> **目的**：基于人类偏好优化模型

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.6.1 | **精读** `train_dpo.py`：DPO 损失函数 | [ ] |
| 3.6.2 | 下载 `dpo.jsonl` 数据集 | [ ] |
| 3.6.3 | 运行 DPO：`python train_dpo.py` | [ ] |

### 3.7 强化学习 - PPO

> **目的**：在线强化学习，实时优化策略

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.7.1 | **精读** `train_ppo.py`：Actor、Critic、GAE | [ ] |
| 3.7.2 | 下载 `rlaif-mini.jsonl` 数据集 | [ ] |
| 3.7.3 | 下载奖励模型 `internlm2-1_8b-reward` 到同级目录 | [ ] |
| 3.7.4 | 运行 PPO：`python train_ppo.py` | [ ] |

### 3.8 强化学习 - GRPO

> **目的**：分组相对策略优化（DeepSeek-R1 使用的算法）

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.8.1 | **精读** `train_grpo.py`：组内归一化优势函数 | [ ] |
| 3.8.2 | 理解 GRPO 与 PPO 的区别（无需 Critic 网络） | [ ] |
| 3.8.3 | 运行 GRPO：`python train_grpo.py` | [ ] |

### 3.9 强化学习 - SPO（实验性）

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.9.1 | **精读** `train_spo.py` | [ ] |
| 3.9.2 | 阅读 README 中 SPO 实验性说明 | [ ] |

---

## 阶段 4：工程化与评测

> **目标**：将模型部署到实际应用

### 4.1 API 服务

| 序号 | 内容 | 核对 |
|------|------|------|
| 4.1.1 | **精读** `scripts/serve_openai_api.py`：OpenAI 兼容服务端 | [ ] |
| 4.1.2 | **精读** `scripts/chat_openai_api.py`：客户端调用示例 | [ ] |
| 4.1.3 | 启动 API 服务并测试 | [ ] |

### 4.2 模型转换

| 序号 | 内容 | 核对 |
|------|------|------|
| 4.2.1 | **精读** `scripts/convert_model.py`：torch ↔ transformers 格式转换 | [ ] |
| 4.2.2 | （可选）转换为 GGUF 格式用于 llama.cpp | [ ] |
| 4.2.3 | （可选）部署到 ollama | [ ] |

### 4.3 评测

| 序号 | 内容 | 核对 |
|------|------|------|
| 4.3.1 | 阅读 README 中 C-Eval、C-MMLU 评测章节 | [ ] |
| 4.3.2 | 理解 YaRN / RoPE 长度外推 | [ ] |
| 4.3.3 | 调整生成参数进行效果对比实验 | [ ] |

---

## 🚀 快速实践指南

### 最小闭环（2小时 + 3块钱）

```bash
# 1. 下载数据集（约 2.8GB）
# 从 ModelScope 下载: pretrain_hq.jsonl + sft_mini_512.jsonl
# 放到 ./dataset/ 目录

# 2. 预训练（约1.1小时）
python train_pretrain.py

# 3. 监督微调（约1小时）
python train_full_sft.py

# 4. 测试效果
python eval_llm.py --weight full_sft
```

### 推荐数据集组合

| 场景 | 数据集 | 大小 | 推荐设置 |
|------|--------|------|----------|
| 快速体验 | `pretrain_hq.jsonl` + `sft_mini_512.jsonl` | 2.8GB | `max_seq_len≈340` |
| 中等规模 | + `sft_1024.jsonl` | 8.4GB | `max_seq_len≈650` |
| 完整复现 | + `sft_2048.jsonl` + `dpo.jsonl` | 18GB+ | 按README建议 |

---

## 📁 附录 A：源码文件全覆盖表

| 路径 | 说明 | 优先级 |
|------|------|--------|
| `model/model_minimind.py` | 配置 + 模型主体 | ⭐⭐⭐ |
| `model/model_lora.py` | LoRA | ⭐⭐ |
| `model/tokenizer.json` | 分词器词表 | ⭐ |
| `dataset/lm_dataset.py` | 数据处理 | ⭐⭐⭐ |
| `trainer/trainer_utils.py` | 训练工具 | ⭐⭐ |
| `trainer/train_pretrain.py` | 预训练 | ⭐⭐⭐ |
| `trainer/train_full_sft.py` | 全参 SFT | ⭐⭐⭐ |
| `trainer/train_lora.py` | LoRA | ⭐⭐ |
| `trainer/train_distillation.py` | 蒸馏 | ⭐ |
| `trainer/train_reason.py` | Reason | ⭐ |
| `trainer/train_dpo.py` | DPO | ⭐⭐ |
| `trainer/train_ppo.py` | PPO | ⭐ |
| `trainer/train_grpo.py` | GRPO | ⭐ |
| `trainer/train_spo.py` | SPO | ⭐ |
| `eval_llm.py` | 推理 | ⭐⭐⭐ |
| `scripts/web_demo.py` | Web UI | ⭐ |
| `scripts/serve_openai_api.py` | API 服务 | ⭐ |
| `scripts/convert_model.py` | 模型转换 | ⭐ |

---

## 📖 附录 B：核心概念速查

| 概念 | 文件位置 | 简要说明 |
|------|----------|----------|
| RMSNorm | `model_minimind.py` | Root Mean Square 归一化，比 LayerNorm 更高效 |
| RoPE | `model_minimind.py` | 旋转位置编码，支持长度外推 |
| SwiGLU | `model_minimind.py` | 门控线性单元激活函数 |
| GQA | `model_minimind.py` | 分组查询注意力，减少 KV Cache |
| MoE | `model_minimind.py` | 混合专家，增加参数但不增加计算量 |
| LoRA | `model_lora.py` | 低秩适配，只训练少量参数 |
| DPO | `train_dpo.py` | 直接偏好优化，无需奖励模型 |
| PPO | `train_ppo.py` | 近端策略优化，需要 Actor+Critic |
| GRPO | `train_grpo.py` | 分组相对策略优化，无需 Critic |

---

*文档说明：与仓库 `README.md` 及当前源码结构对齐；若上游仓库增减脚本，请以实际文件为准。*
