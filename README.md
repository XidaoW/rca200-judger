# Brise RCA Score

`brise-rca-score` 是一个可独立安装和运行的根因分析（Root Cause Analysis，RCA）评分器。评分器读取本地 Ground Truth 与 Agent 诊断结果，对诊断质量进行标准化评估，并输出定因、定界、分析过程和综合得分。

评分器支持单 Case、批量、多根因和离线回放场景。运行时不依赖 SLS、PostgreSQL 或 Brise Server；Live 模式仅通过 DashScope OpenAI-compatible API 调用模型，Decision Replay 模式不产生任何网络请求。

## 核心能力

- **定因评分（F）**：评估 Agent 判断的故障类型与 Ground Truth 的匹配程度。
- **定界评分（E）**：评估 Agent 定位的故障实体、服务、Pod 或宿主节点是否准确。
- **过程评分（P）**：评估诊断过程中的因果一致性、逻辑完整性和证据充分性。
- **综合评分（O）**：根据 F/E/P 权重、可信度门禁和多根因惩罚计算最终分数。
- **多根因评估**：支持 Ground Truth 根因与 Agent Claim Atom 的确定性 assignment、causal rescue 和逐根因聚合。
- **Decision Replay**：使用已冻结的维度判定重新计算评分 kernel，用于回归验证和结果复现。
- **批量执行**：支持 Case 过滤、并行评分、失败隔离和汇总文件输出。
- **模型协议兼容**：处理推理模型的结构化响应、JSON 校正重试和 hidden reasoning 行为。

## 系统组成

```text
CLI / Python API
        │
        ▼
输入合同与本地文件加载
        │
        ├── 单根因评分编排
        ├── 多根因评分编排
        └── Decision Replay
        │
        ▼
定因 F / 定界 E / 过程 P 评分模块
        │
        ▼
综合评分 Kernel、可信度门禁与结果输出
```

主要模块：

| 模块 | 职责 |
|---|---|
| `api.py` | 公共 `score_case` API 与评分模式路由 |
| `contracts.py` | Ground Truth、Diagnosis、Agent Output 和评分结果合同 |
| `inputs.py` | 本地输入读取、校验、哈希和 Agent 输出转换 |
| `resolution.py` | 故障类型、实体和 GT reasoning 上下文解析 |
| `single_root.py` | 单根因 F/E/P 编排与结果组装 |
| `multi_root_engine.py` | 多根因 assignment、rescue、聚合和惩罚计算 |
| `process_runtime.py` | Process Judge、结构化重试和 Decision Replay |
| `scoring/` | 定因、定界、过程和综合分计算逻辑 |
| `llm/` | DashScope HTTP 客户端与模型协议适配 |
| `topology/base.py` | 拓扑协议、节点/边合同和测试 Stub |
| `topology/json_graph.py` | 本地 `topology.json` 加载、索引和图关系计算 |
| `local_cli.py` | 命令行参数、批量任务和结果持久化 |

更详细的模块边界见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 环境要求

- Python 3.10 或更高版本
- `pydantic>=2`
- `pyyaml`
- Live 模式需要有效的 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_ENDPOINT`

## 安装

在仓库根目录执行：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

验证安装：

```bash
rca-score --help
```

开发环境安装：

```bash
pip install '.[dev]'
```

构建 wheel 和源码包：

```bash
python -m build
```

构建结果写入本地 `dist/` 目录。`dist/` 不应提交到 Git，应通过统一制品库分发。

## 快速开始

### 单 Case Live 评分

```bash
export DASHSCOPE_API_KEY='<your-key>'
export DASHSCOPE_ENDPOINT='<your-openai-compatible-endpoint>'

rca-score score \
  --gt-file examples/single-root/private-gt-bundle.json \
  --agent-output examples/single-root/agent_output.json \
  --model qwen3.7-plus \
  --temperature 0 \
  --output score.json
```

常用可选参数：

```text
--topology-file <topology.json>
--trajectory-file <trajectory.json>
--credibility RELIABLE|QUESTIONABLE|UNRELIABLE
--timeout <seconds>
--vote-count <n>
```

### 多根因评分

Ground Truth 可以通过多个 `outcome.root_causes[]` 描述多个根因，并通过 `reasoning.steps[].rc_refs` 将推理步骤绑定到对应根因。

```bash
rca-score score \
  --gt-file examples/multi-root/private-gt-bundle.json \
  --agent-output examples/multi-root/agent_output.json \
  --model qwen3.7-plus \
  --output multi-score.json
```

### Decision Replay

Decision Replay 使用已保存的维度判定重新执行评分 kernel，不调用 DashScope：

```bash
env -u DASHSCOPE_API_KEY rca-score score \
  --gt-file examples/single-root/private-gt-bundle.json \
  --agent-output examples/single-root/agent_output.json \
  --replay-decision-file examples/single-root/decision-snapshot.json \
  --output replay-score.json
```

仓库中的预期输出位于：

```text
examples/single-root/expected-decision-replay.json
```

### 批量评分

批量任务目录结构：

```text
run-root/
├── c001/
│   ├── private-gt-bundle.json
│   ├── agent_output.json
│   ├── diagnosis.json              # 可选
│   ├── topology.json               # 可选
│   ├── trajectory.json             # 可选
│   └── brise-rca-score.json        # Replay 模式可选
├── c002/
└── ...
```

执行 Live 批量评分：

```bash
rca-score batch \
  --run-root /path/to/run-root \
  --dataset-root /path/to/local-dataset \
  --output-dir /path/to/new-empty-output \
  --workers 4 \
  --model qwen3.7-plus
```

执行批量 Decision Replay。每个 Case 目录需要同时包含 `diagnosis.json` 和 `brise-rca-score.json`：

```bash
env -u DASHSCOPE_API_KEY rca-score batch \
  --run-root /path/to/run-root \
  --output-dir /path/to/new-empty-output \
  --replay-decisions
```

批量参数：

```text
--cases c001,c002,c003
--limit 10
--workers 1..8
--process-context legacy|trajectory
```

`--output-dir` 必须不存在或为空，防止失败重跑时覆盖既有评分产物。

## Python API

```python
from pathlib import Path

from brise_rca_score.api import score_case
from brise_rca_score.llm.dashscope import DashScopeLLM

llm = DashScopeLLM(default_model="qwen3.7-plus", timeout_s=180)
result = score_case(
    gt_file=Path("examples/single-root/private-gt-bundle.json"),
    agent_output_file=Path("examples/single-root/agent_output.json"),
    llm=llm,
    model="qwen3.7-plus",
    temperature=0.0,
    credibility="RELIABLE",
)

print(result.model_dump_json(indent=2))
```

兼容入口仍可使用：

```python
from brise_rca_score.offline import score_case
```

## 输入合同

### Ground Truth Bundle

`--gt-file` 必须是 JSON 文件，核心结构如下：

```json
{
  "ground_truth": {
    "outcome": {
      "root_causes": [
        {
          "rc_id": "rc1",
          "fault_id": "F002-threadExhaustion",
          "entity_group": {
            "required_entity_ids": ["entity-inventory"],
            "alias_entity_ids": [],
            "entity_refs": [
              {
                "entity_id": "entity-inventory",
                "entity_name": "inventory",
                "entity_domain": "apm",
                "entity_type": "apm.service"
              }
            ]
          },
          "conclusion": "inventory worker thread pool is exhausted"
        }
      ]
    },
    "reasoning": {
      "steps": [
        {
          "step": 1,
          "rc_refs": ["rc1"],
          "fault_id": "F002-threadExhaustion",
          "step_type": "cause",
          "target": "inventory",
          "title": "Thread pool exhaustion",
          "description": "active threads reach the pool limit"
        }
      ]
    }
  },
  "workspace": "example-workspace",
  "region_id": "example-region",
  "anchor_time": "2026-01-01T00:00:00+08:00"
}
```

### 被评分输出

必须提供以下两种输入之一：

- `--agent-output`：原始 Agent 最终输出，推荐使用。
- `--diagnosis-file`：已经转换完成的 `fault_candidates`、`entity_candidates`、`agent_chain` 和 `conclusion`。

两者不能同时提供。

### 可选上下文

- `topology.json`：用于实体 canonical ID、服务关系、Pod、宿主节点和拓扑距离解析。
- `trajectory.json`：用于向 Process Judge 提供 Agent 分析过程和工具调用摘要。
- `brise-rca-score.json`：用于 Decision Replay 或 Process Review 复用。

## 输出合同

单 Case 输出示例：

```json
{
  "case_id": "example-single",
  "scoring_formula_version": "rca-local-v3.1",
  "input_sha256": "...",
  "judge_model": "qwen3.7-plus",
  "scores": {
    "fault_type": 1.0,
    "entity": 1.0,
    "process": 0.7,
    "overall": 91.0
  },
  "gate": {
    "passed": true,
    "threshold": 60.0
  },
  "details": {},
  "source": {
    "gt": "...",
    "scored_output": "...",
    "scoring": "dashscope"
  }
}
```

批量评分输出：

```text
output-dir/
├── c001/rca-score.json
├── c002/rca-score.json
└── summary.json
```

`summary.json` 记录总 Case 数、失败数以及每个 Case 的执行状态。技术失败会被显式标记，不会自动转换为 Agent 0 分。

## 评分规则

### 定因 F

定因评分根据故障类型的精确匹配、故障簇、GT 因果链距离、LLM causal rescue 和 mechanism gate 计算，分数范围为 0–1。

### 定界 E

定界评分根据实体精确匹配、服务级匹配、宿主节点匹配和拓扑邻接距离计算，分数范围为 0–1。

### 过程 P

过程评分检查三个维度：

- `causal_contradiction`：是否存在因果矛盾。
- `logic_jump`：是否存在关键逻辑跳跃。
- `evidence_gap`：是否缺少必要证据。

当冻结判定明确将 Process 标记为 NA 时，P 权重从综合分中退出并重新归一 F/E 权重。Live Process Judge 调用或协议失败属于技术失败，会显式返回错误，不会转换为 Agent 过程分 0。

### 综合分 O

默认权重：

```text
F = 0.3
E = 0.4
P = 0.3
```

基础公式：

```text
O = 100 × (F × wF + E × wE + P × wP)
```

其他规则：

- Process 为 NA 时，将 P 权重移除并重新归一 F/E 权重。
- 定界完全未命中且不是抽取失败时，定因项应用 entity penalty。
- `QUESTIONABLE` 可信度乘数为 0.5。
- `UNRELIABLE` 可信度乘数为 0.2。
- 多根因漏报时应用 `min(1, C/R)` 根因数量惩罚。
- Agent 根因数超过 GT 时，对定因和必要的定界分应用 precision penalty。
- 综合分范围为 0–100，Gate 默认阈值为 60。

## 模型与运行参数

Live 模式通过 DashScope OpenAI-compatible endpoint 调用模型。主要环境变量：

| 环境变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | API Key，Live 模式必需 |
| `DASHSCOPE_ENDPOINT` | OpenAI-compatible endpoint，Live 模式必需 |
| `DASHSCOPE_TIMEOUT_S` | HTTP 请求超时，默认 180 秒 |
| `DASHSCOPE_MAX_TOKENS` | completion token 上限覆盖值 |
| `DASHSCOPE_REASONING_EFFORT` | 模型支持的 reasoning effort |
| `DASHSCOPE_DISABLE_THINKING` | 是否关闭支持模型的 hidden reasoning；默认关闭 thinking |

DashScope 的 429、5xx、timeout 和网络错误会执行有限次数重试。结构化输出无法通过 schema 校验时，会执行 JSON-only 校正重试；最终失败会显式返回错误。

## 项目目录

```text
.
├── src/brise_rca_score/       # 评分器源码
├── tests/                     # 单元与回归测试
├── tests/fixtures/            # 脱敏评分夹具
├── examples/                  # 单根因、多根因和 Replay 示例
├── tools/                     # 校验工具
├── docs/                      # 架构、验证和 Prompt 对齐文档
├── pyproject.toml             # 包元数据与依赖
├── CHANGELOG.md               # 变更记录
└── README.md                  # 使用说明
```

## 开发与验证

```bash
pip install '.[dev]'
ruff check src tests tools
ruff format --check src tests tools
mypy src
pytest -q
python -m build
```

当前测试覆盖：

- CLI 参数与批量执行
- 输入合同与 Agent 输出转换
- 单根因和多根因评分
- F/E/P/O 评分 kernel
- Fault causal rescue 与 mechanism gate
- Entity topology 和结构距离
- Process Judge schema
- DashScope 模型协议
- Decision Replay
- 公共 API 兼容性

## 安全边界

- 不读取 `.env` 文件。
- API Key 只从构造参数或 `DASHSCOPE_API_KEY` 环境变量读取。
- 不连接 SLS、PostgreSQL、Brise API 或远程拓扑服务。
- Ground Truth、Agent 输出和评分结果不会上传到日志或数据库。
- Topology、Trajectory 和 Replay 文件只从显式本地路径读取。
- Provider 错误、网络错误和结构化响应错误不会静默转换为 Agent 低分。

## 相关文档

- [架构与模块维护边界](docs/ARCHITECTURE.md)
- [验证记录](docs/VALIDATION.md)
- [Prompt 对齐记录](docs/PROMPT_PARITY.md)
- [评分差异分析](docs/FAULT_SCORE_GAP_ANALYSIS.md)
