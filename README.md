# video-cover-select

## 题面

自动选封面后，总是挑到糊帧、同一部片子每次结果还不一样：清晰度评分把帧均值当成了清晰度，整张画面同分；亮度评分用绝对值差但没有归一，分数会跑出 0..1；对比度只用最大灰度减最小灰度、却没有按 8 bit 动态范围归一，暗图反而拿高分；中心加权固定乘 1.5 却不做归一，总分溢出；三分法构图只看水平线、忽略竖线；总分权重不归一，直接超过 1；候选同分时取最后一个，结果不稳定；候选为空不报错，返回 -1 后越界；批量处理共用评分数组，视频之间相互串分；`weights` 全 0 不校验，直接除零。cover/scoring.py 是核心，tests/test_scoring.py 15个红。要修的是：把上述 10 个缺陷逐项修好，并让 15 条测试全部转绿，且不得改动 tests/。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态没有收集错误、导入错误或语法错误，必须恰好是 15 条失败：

```text
15 failed
```

修复 `cover/scoring.py` 后应变为：

```text
15 passed
```

## 仓库结构

```text
video-cover-select/
├── README.md
├── conftest.py
├── .gitignore
├── cover/
│   ├── __init__.py
│   └── scoring.py
└── tests/
    └── test_scoring.py
```

- `cover/scoring.py`：八个待实现的评分/选取函数与 `CoverError`。
- `tests/test_scoring.py`：15 条只读契约测试，初始全部调用桩并失败。
- `conftest.py`：把仓库根目录加入 `sys.path`，保证从任意目录运行 pytest 都能导入 `cover`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`）和 pytest；离线运行，不使用随机数，不访问网络。
- 帧是非空、矩形的二维灰度数组，元素为 `[0, 1]` 内的有限浮点数；所有评分均为 `[0, 1]` 内的浮点数。
- 评分比较和同分判定使用 `1e-6` 容差；非法输入统一抛出 `cover.scoring.CoverError`。
- **`tests/` 不可修改**；只能修改实现文件来让测试转绿。
