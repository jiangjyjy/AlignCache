# AlignCache for Qwen-Image-2.1

ComfyUI 的 `MODEL → MODEL` 加速节点。通过预测部分模型输出减少 Transformer 前向，
保留原生采样更新、VAE 和 prefix KV。提供论文适配的 `paper` 路径及此前冻结的 `legacy` 路径。

## 安装与使用

将整个目录复制或链接到 `ComfyUI/custom_nodes/AlignCache4QwenImage21`，重启 ComfyUI。
使用现有 Torch，无额外推理依赖。节点为 `AlignCacheQwenImage21`。

- 新实现：[文生图工作流](workflows/qwen21_t2i_paper.json)。CFG=1；测试双分支时将 KSampler 的 CFG 改为 2。
- 原固定预设：[legacy 工作流](workflows/qwen21_t2i_aligncache.json)。
- 原生对照：在相同工作流将 AlignCache 的 `mode` 改为 `off`，保持 prompt、seed、CFG 和采样设置相同。
- [编辑工作流](workflows/qwen21_edit_aligncache.json)用于 trace／精确回退，目前不启用编辑加速。
- 同名 `_api.json` 用于自动评测。单独新建节点默认 `trace`、零阈值。

加速验证范围为 **静态 VRAM、1024×1024、25 步 Euler/simple、batch=1、int8 convrot**。
ComfyUI 启动加 `--disable-dynamic-vram`；默认 prefix KV 不变。
重复同 seed 计时时修改 `run_nonce`，避免 ComfyUI 复用采样结果。首轮加载时间与热采样不可直接比较。

## 固定配置

| 设置 | paper | legacy |
| --- | --- | --- |
| advanced_config | `preset:paper_qwen` | `preset:fixed_t2i` |
| base_threshold | 0.12 | 0.12 |
| max_consecutive_skips | 4 | 4 |
| warmup / tail | 5 / 2 | 5 / 2 |
| predictor | scaled_output_momentum | time_normalized_output |
| beta | 阶段值 0.2 / 0.4 / 0.2 | 0.4 |
| CFG | 单条件或正／负双分支；已测 1、2 | 1 |

预设解析到本目录 `presets/*_advanced.json`，不依赖机器绝对路径；也可填写自定义 JSON 路径。
`mode=trace` 返回原生输出并观察影子决策；`mode=on` 启用预测；阈值或全局 skip 上限为 0 时全程精确。

## 实现

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 轻量节点注册，发现阶段不导入 Torch／Comfy |
| `comfyui_nodes.py` | MODEL 克隆、采样生命周期、单条件／成对 wrappers、节点参数 |
| `aligncache_accelerate.py` | 预测器、风险预算、精确锚点、阶段反馈、双分支锚点修正 |
| `runtime.py` | 兼容检查、条件内容签名、JSONL／summary 日志 |
| `benchmark.py` | 可选独立测量节点，统计真实前向与采样耗时 |

paper 补齐论文的阶段缓存年龄上限、CFG 合成 relative MSE、分阶段有界阈值调整、
双分支独立动量和无条件分支锚点修正。仅预算触发的精确计算参与反馈，预测不覆盖真实锚点。

Qwen 的原生输出是速度 v，去噪结果为 `x0 = x - sigma*v`。
本适配预测归一化去噪残差 `(x0-x)/sigma = -v`，等价地在速度空间工作，避免直接计算 `v-x`。
在 Comfy 的条件批处理边界按逻辑条件对决策，因此不依赖 CFG 分支的合批、拆批或执行先后。
CFG=1 自然退化为单分支，不额外调用无条件模型。paper 使用 MSE，legacy 保留原来的 relative L2。

模型／条件／时间表变化或不支持的组合会转为精确计算，并记录原因。
DynamicVRAM、编辑 on、条件分区、mask、未知 sampler/hooks/wrappers 的加速仍未开放。
本实现锁定验证的 ComfyUI commit 为 `b0f4b7b294ce482a2e071d9d762c133d38c7aa07`。

## 验证与开发

离线评测工具与历史原始记录保留在开发环境中，不属于本次插件交付。
RTX 3090 上，paper 的 CFG=1 三例采样加速 2.11–2.26×，PSNR 29.49–42.84 dB；
CFG=2 一例 2.25×、39.53 dB。均为热运行的小规模回归，不保证任意提示词的速度或质量。
legacy 的 10 例历史结果为采样中位数 2.28×、PSNR 中位数 33.47 dB。

在仓库根目录运行：

```bash
COMFYUI_ROOT=/path/to/locked/ComfyUI python -m pytest tests -q
```

核心测试覆盖预测／预算／反馈、真实 ComfyUI CPU 链路、CFG 合批／拆批、条件失效、
原生 KV 清理和 benchmark 数据配对。原始设计背景保留于 [DESIGN_COMFYUI.zh-CN.md](DESIGN_COMFYUI.zh-CN.md)。
