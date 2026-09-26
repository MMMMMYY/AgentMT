# 实验用 polite 前五种

选择规则：以 review_stanford_assisted.md 的最新逐项分类为准，在 17 种 polite 形式中，按 tone_form_inventory.csv 保存的 wildchat_count 降序选前五种。不再按每个小类各选一种。

| 排名 | 表达形式 | 已记录 WildChat 频次 |
|---|---|---:|
| 1 | Could you {TASK}? | 2716 |
| 2 | {TASK}, please. | 2462 |
| 3 | Can you please {TASK}? | 827 |
| 4 | {TASK}, thank you. | 773 |
| 5 | {TASK}, thanks. | 590 |

17 种完整表达库保留，实验采用以上 5 种。Neutral 和 impolite 当前各 5 种，所以实验候选共 15 种形式。旧 pilot 的 prompts 和结果保留，不按本次选择追溯修改。

频次是已保存语料提取中的匹配次数，可能重叠，并非 WildChat Full 全量统计或互斥的用户数。Can you 和 Can you help me 已归入 neutral，故不参加本次 polite 排序。
