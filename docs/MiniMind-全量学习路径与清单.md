# MiniMind 全量学习路径与核对清单

> 面向目标：按本清单逐项完成，可系统覆盖本仓库**模型结构、数据管线、各阶段训练、推理与工程化**的全部内容。  
> 硬件说明：4060 / 云 GPU 均可；具体 batch、序列长以 `README.md` 与脚本参数为准，本清单不绑定显存数字。

---

## 使用方式

- **顺序**：大体按阶段 0→N 推进；同一阶段内可并行阅读「理论 / 代码 / README 对应小节」。
- **勾选**：每完成一项，将 `[ ]` 改为 `[x]`。
- **权威文档**：以仓库根目录 `README.md`（及 `README_en.md`）为准；本清单与源码文件一一对应，避免遗漏模块。

---

## 阶段 0：环境与仓库全局

| 序号 | 内容 | 核对 |
|------|------|------|
| 0.1 | 阅读 `README.md` 前言、项目目标、模型列表与更新日志（了解版本演进与命名） | [ ] |
| 0.2 | 安装依赖：`requirements.txt`，确认 `torch` 能使用 CUDA（`README` 快速开始中有检测方式） | [ ] |
| 0.3 | 理解目录：`model/`（模型）、`dataset/`（数据与 `lm_dataset.py`）、`trainer/`（训练脚本与 `trainer_utils.py`）、`scripts/`（工具与演示）、根目录 `eval_llm.py` | [ ] |
| 0.4 | 阅读 `dataset/dataset.md`（数据放置位置约定） | [ ] |
| 0.5 | 浏览 `LICENSE`、`CODE_OF_CONDUCT.md`（开源与协作约定，可选但建议扫一眼） | [ ] |
| 0.6 | 知悉 `README` 中数据集下载地址（ModelScope / HuggingFace），以及 `dataset/` 下各 `*.jsonl` 文件名与用途对照表 | [ ] |

---

## 阶段 1：推理先行（建立端到端直觉）

| 序号 | 内容 | 核对 |
|------|------|------|
| 1.1 | 按 `README` 下载 HuggingFace/ModelScope 格式的 **MiniMind2**（或其它发布权重）到本地 | [ ] |
| 1.2 | 运行 `python eval_llm.py --load_from ./MiniMind2`（或你本机路径），理解命令行参数：`--weight`、`--save_dir`、`--hidden_size`、`--use_moe`、`--lora_weight`、`--inference_rope_scaling` 等 | [ ] |
| 1.3 | **精读** `eval_llm.py`：`init_model` 中「`model` 关键字 + 原生 `.pth`」与「`from_pretrained`」两条路径 | [ ] |
| 1.4 | （可选）`scripts/web_demo.py` + Streamlit 本地聊天界面 | [ ] |
| 1.5 | （可选）`README` 中 **ollama / vllm** 等第三方推理方式，理解「训练仓库产物」与「生态部署」的关系 | [ ] |

---

## 阶段 2：Tokenizer 与文本→张量

| 序号 | 内容 | 核对 |
|------|------|------|
| 2.1 | 阅读 `README`「Tokenizer」小节：词表大小、minimind_tokenizer 设计取舍 | [ ] |
| 2.2 | 浏览 `model/tokenizer_config.json`、`model/tokenizer.json`（知道词表与特殊符号从哪来） | [ ] |
| 2.3 | **精读** `trainer/train_tokenizer.py`：自训练分词器的流程（README 标明仅供学习，非必跑） | [ ] |
| 2.4 | 在 `dataset/lm_dataset.py` 中跟踪：`PretrainDataset` 如何把 `text` 变成 `input_ids`/`labels`，以及 `pad` 与 `-100` 掩码 | [ ] |

---

## 阶段 3：模型核心（必精读）

| 序号 | 内容 | 核对 |
|------|------|------|
| 3.1 | **精读** `model/model_minimind.py`：`MiniMindConfig`（`hidden_size`、`num_hidden_layers`、GQA `num_key_value_heads`、RoPE、`rope_scaling`/YaRN、`use_moe` 与各 MoE 字段） | [ ] |
| 3.2 | 同文件：RMSNorm、RoPE 频率预计算、`Attention`、`MLP`、Transformer Block、`MiniMindForCausalLM` 的 `forward` 与 `CausalLMOutputWithPast` | [ ] |
| 3.3 | 理解 **MoE** 分支：何时产生 `aux_loss`，与预训练循环中 `loss + aux_loss` 的对应关系 | [ ] |
| 3.4 | **精读** `model/model_lora.py`：LoRA 如何挂到线性层、`apply_lora` / `load_lora` 与 `eval_llm.py` 的衔接 | [ ] |
| 3.5 | （可选）对照 `README` / 论文链接（如 MobileLLM）理解「小模型宽度与深度」取舍 | [ ] |

---

## 阶段 4：训练公共基础设施

| 序号 | 内容 | 核对 |
|------|------|------|
| 4.1 | **精读** `trainer/trainer_utils.py`：`get_lr`、`init_distributed_mode`、`setup_seed`、`lm_checkpoint`（含 `resume`）、`init_model`、`SkipBatchSampler`、`Logger`、`get_model_params` | [ ] |
| 4.2 | 理解 **DDP**：`README` 中 `torchrun --nproc_per_node N train_xxx.py`；单卡即直接 `python train_xxx.py` | [ ] |
| 4.3 | 理解 **断点续训**：各脚本 `--from_resume 1` 与 `./checkpoints/*_resume.pth`（见 `README`） | [ ] |
| 4.4 | 理解 **wandb / SwanLab**：`--use_wandb` 与 `README` 中关于国内网络与 SwanLab 的说明 | [ ] |
| 4.5 | 理解混合精度、`grad_clip`、`accumulation_steps` 在 `train_pretrain.py` 等中的典型写法 | [ ] |

---

## 阶段 5：数据管线（`lm_dataset.py` 全类）

| 序号 | 内容 | 核对 |
|------|------|------|
| 5.1 | `PretrainDataset`：`pretrain_hq.jsonl` 格式与 `max_length` | [ ] |
| 5.2 | `SFTDataset`：`conversations` 格式、`apply_chat_template`、`generate_labels`（只算 assistant 段 loss） | [ ] |
| 5.3 | `pre_processing_chat` / `post_processing_chat`：system 注入、Reason 相关字符串处理 | [ ] |
| 5.4 | `DPODataset`：`chosen`/`rejected` 与 DPO 所需拼接方式 | [ ] |
| 5.5 | `RLAIFDataset`：RL 脚本所用对话字段与采样 | [ ] |
| 5.6 | 对照 `README`「数据介绍」：`pretrain_hq`、`sft_*`、`dpo.jsonl`、`r1_mix_1024.jsonl`、`rlaif-mini.jsonl`、`lora_*.jsonl` 的推荐 `max_seq_len` | [ ] |

---

## 阶段 6：预训练（Pretrain）

| 序号 | 内容 | 核对 |
|------|------|------|
| 6.1 | **精读** `trainer/train_pretrain.py`：参数列表、`train_epoch`、loss（含 `aux_loss`）、保存 `.pth` | [ ] |
| 6.2 | 下载并配置 `pretrain_hq.jsonl`，至少跑通一小段步数或确认数据管线无报错 | [ ] |
| 6.3 | 能解释输出权重命名：`pretrain_{hidden_size}.pth`（及 MoE 后缀，若启用） | [ ] |

---

## 阶段 7：监督微调（Full SFT）

| 序号 | 内容 | 核对 |
|------|------|------|
| 7.1 | **精读** `trainer/train_full_sft.py`：数据路径、`SFTDataset`、与预训练权重的衔接 | [ ] |
| 7.2 | 使用 `sft_mini_512.jsonl` 或 `README` 推荐组合跑通 SFT（或短跑验证） | [ ] |
| 7.3 | 用 `eval_llm.py --weight full_sft` 验证自训权重对话效果 | [ ] |

---

## 阶段 8：LoRA 微调

| 序号 | 内容 | 核对 |
|------|------|------|
| 8.1 | **精读** `trainer/train_lora.py` + `model/model_lora.py`（README：不依赖 peft 封装） | [ ] |
| 8.2 | 准备 `lora_identity.jsonl` / `lora_medical.jsonl`（或同格式自有数据），跑通 LoRA | [ ] |
| 8.3 | `eval_llm.py` 加载 `--lora_weight` 测试合并后的表现 | [ ] |

---

## 阶段 9：白盒蒸馏

| 序号 | 内容 | 核对 |
|------|------|------|
| 9.1 | **精读** `trainer/train_distillation.py`（README：同系列无大教师时多为学习参考） | [ ] |
| 9.2 | 理解蒸馏损失与 SFT 的差异、数据/教师 logits 依赖 | [ ] |

---

## 阶段 10：Reason（推理链 / 蒸馏式推理）

| 序号 | 内容 | 核对 |
|------|------|------|
| 10.1 | **精读** `trainer/train_reason.py`：模板、`empty_think_ratio`、标签位置 loss 等技巧（见 `README` 说明） | [ ] |
| 10.2 | 使用 `r1_mix_1024.jsonl` 等数据，理解 `max_seq_len` 建议值 | [ ] |
| 10.3 | `eval_llm.py` 中 `--weight reason`（与其它 weight 前缀） | [ ] |

---

## 阶段 11：DPO（偏好优化）

| 序号 | 内容 | 核对 |
|------|------|------|
| 11.1 | **精读** `trainer/train_dpo.py`：DPO 损失、参考模型与策略模型 | [ ] |
| 11.2 | 使用 `dpo.jsonl`，对照 `README` 运行与评测 | [ ] |

---

## 阶段 12：强化学习 — PPO

| 序号 | 内容 | 核对 |
|------|------|------|
| 12.1 | **精读** `trainer/train_ppo.py`：奖励、价值网络、rollout 等与 `RLAIFDataset` 的配合 | [ ] |
| 12.2 | 使用 `rlaif-mini.jsonl`，阅读 `README` 中 PPO 章节与注意事项 | [ ] |
| 12.3 | `eval_llm.py --weight ppo_actor` | [ ] |

---

## 阶段 13：强化学习 — GRPO

| 序号 | 内容 | 核对 |
|------|------|------|
| 13.1 | **精读** `trainer/train_grpo.py` | [ ] |
| 13.2 | 对照 `README` 中 GRPO 说明与超参 | [ ] |
| 13.3 | `eval_llm.py --weight grpo` | [ ] |

---

## 阶段 14：强化学习 — SPO（实验性）

| 序号 | 内容 | 核对 |
|------|------|------|
| 14.1 | **精读** `trainer/train_spo.py` | [ ] |
| 14.2 | 阅读 `README` 中 **实验性说明**（小模型、实现与论文差异） | [ ] |
| 14.3 | `eval_llm.py --weight spo` | [ ] |

---

## 阶段 15：工程化与对外接口

| 序号 | 内容 | 核对 |
|------|------|------|
| 15.1 | **精读** `scripts/chat_openai_api.py`：客户端调用方式 | [ ] |
| 15.2 | **精读** `scripts/serve_openai_api.py`：OpenAI 兼容服务端，与 ChatUI 集成思路（`README`） | [ ] |
| 15.3 | **精读** `scripts/convert_model.py`：格式转换与第三方推理栈（llama.cpp 等）衔接 | [ ] |
| 15.4 | 回顾 `scripts/web_demo.py` 与 Streamlit 部署 | [ ] |

---

## 阶段 16：评测与扩展阅读

| 序号 | 内容 | 核对 |
|------|------|------|
| 16.1 | `README` 中 **C-Eval、C-MMLU、OpenBookQA** 等评测与 **YaRN / RoPE 外推** 相关章节 | [ ] |
| 16.2 | `eval_llm.py` 中生成参数：`temperature`、`top_p`、`max_new_tokens`、`historys` | [ ] |
| 16.3 | （可选）阅读 `README_en.md` 与英文资料对照 | [ ] |
| 16.4 | （可选）孪生项目 **MiniMind-V**（多模态）仓库，与本仓库边界区分 | [ ] |

---

## 附录 A：源码文件全覆盖表（用于查漏）

以下为当前仓库内与「学习 MiniMind」直接相关的文件，**建议至少浏览一遍文件头与入口**：

| 路径 | 说明 |
|------|------|
| `model/model_minimind.py` | 配置 + 模型主体 |
| `model/model_lora.py` | LoRA |
| `model/tokenizer.json` / `tokenizer_config.json` | 分词器资源 |
| `dataset/lm_dataset.py` | 四类 Dataset + 前后处理 |
| `dataset/dataset.md` | 数据目录说明 |
| `trainer/trainer_utils.py` | 训练公共工具 |
| `trainer/train_pretrain.py` | 预训练 |
| `trainer/train_full_sft.py` | 全参 SFT |
| `trainer/train_lora.py` | LoRA |
| `trainer/train_distillation.py` | 蒸馏 |
| `trainer/train_reason.py` | Reason |
| `trainer/train_dpo.py` | DPO |
| `trainer/train_ppo.py` | PPO |
| `trainer/train_grpo.py` | GRPO |
| `trainer/train_spo.py` | SPO |
| `trainer/train_tokenizer.py` | 训练 tokenizer |
| `eval_llm.py` | 推理与对话 |
| `scripts/web_demo.py` | Web UI |
| `scripts/chat_openai_api.py` | API 客户端示例 |
| `scripts/serve_openai_api.py` | API 服务端 |
| `scripts/convert_model.py` | 模型转换 |
| `README.md` / `README_en.md` | 总文档 |
| `requirements.txt` | 依赖 |

---

## 附录 B：建议的「最小闭环」与「全量闭环」

- **最小闭环（时间紧）**  
  阶段 0 → 1 → 3（模型）→ 4 → 5（数据类）→ 6（预训练短跑）→ 7（SFT）→ `eval_llm.py` 验证。

- **全量闭环（本项目不遗漏）**  
  附录 A 中每一行 + 本清单各阶段 `[x]` 全部完成。

---

*文档生成说明：与仓库 `README.md` 及当前源码结构对齐；若上游仓库增减脚本，请以实际文件为准并自行在附录 A 中增补一行。*
