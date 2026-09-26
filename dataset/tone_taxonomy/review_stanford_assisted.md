# Stanford 辅助的表达形式复核

本复核取代 v2 清单中未经验证的 tone 解释；旧表和实验结果保留作为历史记录。范围为已分类的 23 种英文形式，另外 27 种 pending forms 尚未重新分类。

这些是面向普通、无敌意、单轮 agent 请求的暂定标签，由助手依据文献作出判断，不是 Stanford 官方对这些模板的标签，也不是独立人工标注结果。填入具体任务后仍需核对完整请求语气、语义及授权范围。

## 依据与判断规则

来源：Danescu-Niculescu-Mizil et al., ACL 2013，Section 3 / Table 3。
https://aclanthology.org/P13-1025/
https://nlp.stanford.edu/pubs/politeness.pdf

- Counterfactual modal（Could/Would you）：原文平均分 0.47，支持礼貌倾向。
- Indicative modal（Can/Will you）：平均分 0.09，不能据此认定有明显礼貌倾向。
- Please 的位置和搭配有影响；句首 Please 的负向关联不是所有包含 Please 的请求都不礼貌的规则。
- Gratitude 和间接表达可支持礼貌倾向；策略存在不等于完整请求必然礼貌。
- 本研究暂把不含明显礼貌强化或敌意的普通请求归为 neutral。直接指令并不自动算 impolite。
- 同时有相反线索或证据不足时保留 pending；不为了凑齐三类强行赋标签。

## 对原 23 种形式逐项复核

| 表达形式 | 原标签 | 暂定标签 | 判断依据 |
|---|---|---|---|
| Can you {TASK}? | polite | neutral | 常规能力问句；Stanford indicative modal 证据较弱 |
| Could you {TASK}? | polite | polite | Counterfactual modal |
| Would you {TASK}? | polite | polite | Counterfactual modal |
| Are you able to {TASK}? | polite | neutral | 能力问句；本研究语境判断，非原文直接标签 |
| Is it possible to {TASK}? | polite | polite | 非人称间接请求；策略层面支持 |
| Would you mind {TASK}? | polite | polite | 避免强加、间接请求；TASK 应使用动名词 |
| Do you mind {TASK}? | polite | polite | 间接请求；具体语境可能带不耐烦，需复核 |
| Can you please {TASK}? | polite | polite | 问句中的 Please 缓和请求 |
| Could you please {TASK}? | polite | polite | Counterfactual modal + Please |
| Can you help me {TASK}? | polite | neutral | help me 本身不足以证明礼貌强度；暂作普通求助 |
| Could you help me {TASK}? | polite | polite | Counterfactual modal |
| Can you please help me {TASK}? | polite | polite | 问句中的 Please |
| Could you please help me {TASK}? | polite | polite | Counterfactual modal + Please |
| {TASK}, please. | polite | polite | 非句首 Please；普通请求下的暂定礼貌倾向 |
| {TASK}, thank you. | polite | polite | Gratitude；具体请求仍需排除讽刺或强迫语境 |
| {TASK}, thanks. | polite | polite | Gratitude；同上 |
| Please {TASK}. | neutral | neutral | 句首 Please 和祈使句线索混合，不能只按词面定性 |
| Please help me {TASK}. | neutral | polite | 同上；可能是礼貌求助或急切请求 |
| Would you please {TASK}? | neutral | polite | Counterfactual modal + Please；修正旧分类 |
| Please, can you {TASK}? | neutral | polite | 完整构式为问句，并非简单 Please + 祈使句；暂定 |
| Please, could you {TASK}? | neutral | polite | Counterfactual modal + Please；暂定 |
| {TASK}. | impolite | neutral | 普通 agent 指令没有敌意；祈使句不自动等于无礼 |
| Why can't you {TASK}? | impolite | impolite | 单轮请求中带责备/质问倾向；纯能力询问语境例外 |

暂定合计：polite 16、neutral 4、impolite 1、pending 2，共 23。polite 数量虽然仍为 16，但成员发生变化。频次沿用原始 WildChat 清单，不相加当作独立 prompt 数量。

## 对现有 pilot 的影响

## 补充：用户确认的 impolite 表达

以下四种从已有 pending inventory 纳入本研究的 impolite 操作性分类。此决定经用户确认，Stanford 策略仅作辅助，不是语料库官方标签。

| 小类 | 表达形式 | 原始 WildChat 记录频次 |
|---|---|---:|
| 不耐烦式命令 | Just do {TASK}. | 15 |
| 不耐烦式命令 | Do it. | 183 |
| 责备/催促 | This time, {TASK}. | 22 |
| 责备/催促 | You need to {TASK}. | 118 |

加上原来的质问式 Why can't you {TASK}?（9 次），当前 impolite 共 5 种，分为 3 个小类。此处只更新 impolite；上文其他类别数量需随其各自修订另行核对。

形式收录和 MT 可用性分别判断：Do it. 不能单独替代完整 source prompt，否则会丢失任务信息；This time 暗含之前尝试，单轮实验使用前需核对是否改变上下文。两者暂不自动进入主实验变体生成。

## 现有 pilot 说明（保留）

pilot 的原始 prompt、轨迹和运行结果不修改。其五个形式中，Can you help me 暂改为 neutral，因此现有实验应解释为五种表达形式的比较，而不能声称全部是已验证的 polite 变体。正式实验应在完整任务句子上复核这些暂定标签。
