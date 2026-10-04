# MCC-OPD 数据

从 [GitHub Release](https://github.com/RZ-Q/MCC-OPD/releases/tag/data-v1) 下载
`MCC-OPD-data.zip`，在代码仓库的父目录解压，两者合并为 `MCC-OPD/`。
默认训练配置即可使用其中的 `data/locomo/train.parquet` 和 `data/locomo/val.parquet`。
LoCoMo、MuSiQue 和 LongMemEval 已准备好，不需要重新检索或调用 API。
REALTALK 按下方命令从官方数据制作，包含同样的三题 reference 订正。

| 文件 | 题数 | 内容 |
|---|---:|---|
| `locomo/train.parquet` | 140 | 论文训练输入，含 Full prompt、去除证据后的 CF prompt 及其 Qwen3 token IDs |
| `locomo/val.parquet` | 77 | 论文验证输入 |
| `locomo/test.jsonl.gz` | 1192 | 完整对话，保留图像 caption 与修订后的问答 |
| `realtalk/test.jsonl.gz`（本地制作） | 728 | 完整对话、caption、秒级时间与多 reference |
| `musique/test.jsonl.gz` | 2172 | 修订后的 MuSiQue-Ans，全段落输入与 v4 prompt |
| `longmemeval/test.jsonl.gz` | 499 | 固定 R60 检索输入，含五题证据修订 |

训练与验证 parquet 与论文输入逐字节一致，遵循 verl 的 `prompt`、`reward_model`、
`extra_info` 格式。`extra_info.c_minus_e` 是 CF 文本，`vad_cf_input_ids` 是训练接口沿用的
CF token IDs 字段名；不是额外的 VAD 算法。tokenizer 为 Qwen3，关闭内置 thinking。
LoCoMo 按问题划分，三个 split 题号互斥，共享十段对话。

测试文件每行是一道题，包含 `question_id`、`question`、`answer`、`messages` 和原分类／证据。
直接对 `messages` 使用 Qwen3 chat template（`add_generation_prompt=True`、
`enable_thinking=False`）。`context_window` 和 `max_new_tokens` 保留各轨实际预算。
MuSiQue 另含 `answer_aliases`、全部 `paragraphs` 和 `question_decomposition`；REALTALK
保留 `accepted_answers`；LongMemEval 保留 `retrieved_memories` 和 `abstention`。
LongMemEval 合并原 60／25／414 条输入，对仅使用 LoCoMo 训练的模型统一作为测试集。
文件校验值和来源校验值见 `manifest.json`。

## 订正前后对比

每轨同时提供 `before.jsonl.gz` 和 `changes.jsonl.gz`（REALTALK 由制作脚本生成）。前者保留订正前的题面、参考答案、
证据及来源题号；后者按 `question_id` 对齐全部来源题目，标明 `unchanged`、`revised`、
`removed` 或 `added`，并为变化字段同时列出 `before` / `after`。`released_split` 指向
上表中的最终文件。修改后的字段值取自与训练 parquet、测试 `messages` 对应的实际数据。

| 对比 | 订正前 | 最终交付 |
|---|---:|---:|
| LoCoMo 原始非 adversarial QA | 1540 | train 140 / val 77 / test 1192 |
| REALTALK 原始 QA | 728 | test 728 |
| MuSiQue-Ans 官方 dev | 2417 | test 2172 |
| LongMemEval 固定 R60 输入 | 500 | test 499 |

LoCoMo 和 REALTALK 的 `source-original/` 保留上游原始对话，便于逐 turn 核对正文、caption
和时间；LoCoMo 的类别 5 不在本次比较范围。MuSiQue 的 `before` 包含官方全部段落及问题分解，
可直接检查段落、证据标记、问答和别名的变化。LongMemEval 的 `before` 是修订前已经冻结的
R60 检索输入，包含原始检索条目和历史文本；检索选择本身不算作本轮数据订正。

各轨 `revisions/` 保留现有处置依据：LoCoMo 的初始排除清单、QA／证据修订和训练 reference
修订；REALTALK 的三题 reference 决策；MuSiQue 的 test dispositions 与后续 intent decisions；
LongMemEval 的 QA、排除和五题证据补充记录。MuSiQue 记录只导出论文使用的 test 部分。
这些记录保留当时的来源路径，当前可直接使用的数据以上表和 `changes.jsonl.gz` 为准。
Judge 的题级 guidance 随评测代码提供。

## REALTALK 本地制作

在代码仓库根目录执行：

```bash
git clone https://github.com/danny911kr/REALTALK.git tmp/REALTALK
git -C tmp/REALTALK checkout b903e06a9770bf4e5fe9018c3e132889666d3b4a
python data/prepare_realtalk.py --source-dir tmp/REALTALK/data --output-dir data/realtalk
```

脚本仅使用 Python 标准库，保留原始对话、caption、时间和题号，应用
`reference-fixes.json` 中的三题订正。公开数据包不重托管 REALTALK；
上游尚未明确数据许可，相关询问见[官方 issue #4](https://github.com/danny911kr/REALTALK/issues/4)。

## 来源与归属

- **LoCoMo**：[snap-research/locomo](https://github.com/snap-research/locomo)，版本
  `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。请引用 *Evaluating Very Long-Term
  Conversational Memory of LLM Agents*。随上游项目提供的 CC BY-NC 4.0 许可见
  `licenses/LoCoMo.txt`。本包包含本文使用的划分、caption 和问答修订。
- **REALTALK**：[danny911kr/REALTALK](https://github.com/danny911kr/REALTALK)，版本
  `b903e06a9770bf4e5fe9018c3e132889666d3b4a`。请引用 Lee et al., *REALTALK: A 21-Day
  Real-World Dataset for Long-Term Conversation*。本包使用 `full-caption-time-r1-20260924`
  的输入与 reference；数据归原作者所有，使用条件以该来源为准。
- **MuSiQue**：[StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)，data v1.0。
  请引用 Trivedi et al., *MuSiQue: Multihop Questions via Single-hop Question Composition*。
  随上游项目提供的 CC BY 4.0 许可见 `licenses/MuSiQue.txt`。本包测试集为本文修订的
  `musique-consolidated-20260916-r2`，保留 2172 题与接受的答案别名。
- **LongMemEval**：[xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)，
  [上游代码](https://github.com/xiaowu0162/LongMemEval)。请引用 Wu et al., *LongMemEval:
  Benchmarking Chat Assistants on Long-Term Interactive Memory*。官方数据集页标注 MIT，
  许可文本见 `licenses/LongMemEval.txt`。本包固定本文实际使用的 R60
  证据，不需要再次运行检索。
