# eval_llm.py 代码详解（MiniMind 推理脚本）

本文档用于系统讲解 `eval_llm.py` 的结构、执行流程、关键参数与实际用法，帮助你从“会运行”到“看懂并能改”。

---

## 1. 文件定位：这个脚本在项目中做什么？

`eval_llm.py` 是 MiniMind 项目的**推理与交互入口脚本**。  
它的职责可以概括为四件事：

1. **读取命令行参数**（决定模型类型、权重、采样策略、设备等）
2. **初始化 tokenizer 与模型**（支持原生权重和 HuggingFace 格式）
3. **组织输入与对话历史**（自动测试或手动聊天）
4. **调用 `generate` 生成回复并统计速度**

换句话说，它是一个“可配置的命令行聊天程序”，专门服务于 MiniMind 模型推理评估。

---

## 2. 顶层依赖与作用

脚本导入的模块分为四类：

### 2.1 Python 标准库

- `time`：计时，用于计算 tokens/s
- `argparse`：解析命令行参数
- `random`：预留随机种子功能（当前主逻辑固定种子）
- `warnings`：忽略警告信息，让终端输出更干净

### 2.2 PyTorch 与 Transformers

- `torch`：模型加载、设备迁移（CPU/CUDA）
- `AutoTokenizer`：加载分词器
- `AutoModelForCausalLM`：加载 HF 格式自回归模型
- `TextStreamer`：流式打印生成内容（边生成边显示）

### 2.3 MiniMind 项目内模型实现

- `MiniMindConfig`：MiniMind 模型配置
- `MiniMindForCausalLM`：MiniMind 因果语言模型类
- `model_lora` 中函数：给模型注入 LoRA 并加载 LoRA 权重

### 2.4 训练/工具函数

- `setup_seed`：设置随机种子，控制可复现
- `get_model_params`：打印/统计模型参数信息

---

## 3. `init_model(args)`：模型初始化逻辑详解

这个函数是脚本的核心初始化入口，最终返回：

- `model.eval().to(args.device)`
- `tokenizer`

### 3.1 分词器加载

```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
```

它假设 `args.load_from` 所指路径可被 `from_pretrained` 识别（本地目录或模型名）。

### 3.2 两种模型加载路径（关键分支）

#### 路径A：`if 'model' in args.load_from`

这条分支表示“按项目原生方式构建模型并加载 `.pth` 权重”：

1. 用 `MiniMindConfig(...)` 构建配置（`hidden_size`、`num_hidden_layers`、`use_moe`、`inference_rope_scaling`）
2. 实例化 `MiniMindForCausalLM`
3. 拼接 checkpoint 路径  
   `./{save_dir}/{weight}_{hidden_size}{_moe可选}.pth`
4. `load_state_dict` 严格加载权重（`strict=True`）
5. 如指定 LoRA，则：
   - `apply_lora(model)`
   - `load_lora(...)`

适合你自己训练出的原生权重工作流。

#### 路径B：`else`

这条分支走 HuggingFace 标准加载：

```python
AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
```

适合已导出为 HF 格式的模型目录（例如你当前运行的 `--load_from ./MiniMind2`）。

### 3.3 参数统计与推理模式

最后调用：

- `get_model_params(model, model.config)`：通常用于打印参数规模
- `model.eval()`：关闭训练态行为（如 dropout）
- `.to(device)`：迁移到 CPU 或 GPU

---

## 4. `main()` 的整体流程（从启动到输出）

### 4.1 参数定义阶段

脚本通过 `argparse` 提供了较全面的推理开关。重点参数如下：

- **模型来源相关**
  - `--load_from`：模型加载路径/模式
  - `--save_dir`：原生权重目录
  - `--weight`：权重前缀（`pretrain/full_sft/rlhf/reason/...`）
  - `--lora_weight`：LoRA 权重名

- **结构相关**
  - `--hidden_size`
  - `--num_hidden_layers`
  - `--use_moe`
  - `--inference_rope_scaling`

- **生成相关**
  - `--max_new_tokens`
  - `--temperature`
  - `--top_p`
  - `--historys`（注意拼写为 historys）

- **运行与展示**
  - `--show_speed`
  - `--device`

### 4.2 预设问题（自动模式）

脚本内置 8 条中文 prompt，覆盖常识、编程、解释、建议等场景，用于“自动测试”模型基础能力。

### 4.3 模式选择

程序启动后会让你选择：

- `[0] 自动测试`：按预设 prompts 依次跑
- `[1] 手动输入`：交互式聊天，输入空字符串结束

### 4.4 对话历史维护

每轮都会：

1. 截取最近 `historys` 条历史（若为0则不保留）
2. 追加当前 `user` 消息

这决定了模型能否“记住上下文”。

### 4.5 输入模板构建

脚本用 `tokenizer.apply_chat_template(...)` 组织对话格式（非 pretrain 权重时）：

- `add_generation_prompt=True`：提示模型“该你回答了”
- 若 `weight == 'reason'`，附加 `enable_thinking=True`

如果是 `pretrain` 权重，不用 chat template，而是：

```python
tokenizer.bos_token + prompt
```

这符合预训练续写式输入范式。

### 4.6 Tokenize 与设备迁移

```python
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

得到 `input_ids` 与 `attention_mask`，并迁移到目标设备。

### 4.7 生成阶段

调用 `model.generate(...)`，关键设置：

- `do_sample=True`：采样而非贪心
- `top_p` + `temperature`：控制随机性与多样性
- `max_new_tokens`：本轮最大生成长度
- `streamer=TextStreamer(...)`：流式打印输出

### 4.8 后处理与统计

1. 截去输入部分，仅保留新生成 token
2. `decode(skip_special_tokens=True)` 得到文本
3. 把 assistant 回复加入 `conversation`
4. 计算并打印速度：`gen_tokens / elapsed_time`

---

## 5. 关键参数的“直觉解释”

### 5.1 `temperature`

- 越低：更稳、更保守、重复率可能更高
- 越高：更发散、更有创造性、也更可能跑偏
- 常用范围：`0.6 ~ 1.0`

### 5.2 `top_p`

- 控制采样时保留的概率质量
- 越小：词表候选更少，输出更“收敛”
- 越大：候选更丰富，输出更“开放”

### 5.3 `historys`

- 值越大：上下文记忆越强，但输入更长、速度更慢、也更占显存
- 值为0：每轮独立问答，不保留上下文

### 5.4 `max_new_tokens`

- 限制“最多能说多长”
- 不是模型真实长上下文能力指标
- 设置过大可能导致长时间生成或无效拖长

---

## 6. 为什么你的命令可运行？

你当前运行的是：

```bash
python eval_llm.py --load_from ./MiniMind2
```

这会触发 **HF 加载分支**（`else` 分支），因为 `./MiniMind2` 不包含 `'model'` 子串（通常如此），然后使用：

```python
AutoModelForCausalLM.from_pretrained("./MiniMind2", trust_remote_code=True)
```

因此你不需要手动指定 `.pth` 路径，也不依赖 `save_dir/weight` 拼接规则。

---

## 7. 常见使用范式（建议）

### 7.1 快速体验（HF导出模型）

```bash
python eval_llm.py --load_from ./MiniMind2
```

### 7.2 降低随机性，便于复测

```bash
python eval_llm.py --load_from ./MiniMind2 --temperature 0.3 --top_p 0.8
```

### 7.3 启用多轮上下文

```bash
python eval_llm.py --load_from ./MiniMind2 --historys 6
```

### 7.4 显式指定 CPU（无GPU时）

```bash
python eval_llm.py --load_from ./MiniMind2 --device cpu
```

---

## 8. 代码中的可优化点（阅读后进阶）

以下是你后续可以考虑的改进方向：

1. **`historys` 命名**  
   建议改为 `history` 或 `max_history_turns`，语义更清晰。

2. **随机种子策略**  
   当前每轮都固定 `setup_seed(2026)`，会增强可复现，但降低多样性。  
   可改为可配置：固定/随机二选一。

3. **异常处理**  
   模型加载失败、权重路径不存在、tokenizer不匹配等可补充更友好的报错。

4. **生成控制扩展**  
   可增加 `repetition_penalty`、`top_k`、`no_repeat_ngram_size` 为命令行参数。

5. **批量评测支持**  
   目前是单条交互生成，可扩展到批处理 benchmark。

---

## 9. 一句话总结

`eval_llm.py` 本质上是一个“**可切换权重来源 + 可配置采样策略 + 支持对话历史 + 流式输出 + 速度统计**”的 MiniMind 推理脚本，既可用于快速体验模型，也可用于基础评测与调参验证。

