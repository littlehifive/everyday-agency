Measure agency as direct evidence in the user text only.

Core scoring requirements:
- Score each attribute independently.
- Do not infer one attribute from another.
- Do not infer agency from topic alone.
- Use only direct language evidence from the text itself.
- Keep negative attributes independent from positive attributes.

Calibration anchors:
- 0 to 10: effectively absent.
- 20 to 40: weak / occasional signal.
- 45 to 60: moderate / mixed signal.
- 65 to 85: strong consistent signal.
- 90 to 100: extreme and unambiguous signal (rare).

Additional rules:
- Penalize vagueness for specificity-focused attributes.
- Reward concrete next actions and constraints awareness for execution-focused attributes.
- For `agency_abdication`, score high only when responsibility is mostly transferred away from the user.
- For `fatalism_helplessness`, score high only when text expresses low perceived control over outcomes.
- Avoid defaulting to midpoint values; use the full range where justified.

Short rubric examples:
1. Text: "Can you just decide everything for me and tell me exactly what to do. I cannot handle this."
   Expected pattern: high `agency_abdication`, high `fatalism_helplessness`, low `goal_commitment`, low `execution_readiness`.
2. Text: "My goal is to ship v1 this week. Today I will finish API schema, then write two integration tests, and by Friday review regressions."
   Expected pattern: high `goal_specificity`, high `planning_specificity`, high `sequencing_prioritization`, high `execution_readiness`.
3. Text: "I tried approach A and it failed because of auth timeouts. Next I will isolate token refresh logic and test with a smaller case."
   Expected pattern: high `obstacle_diagnosis_quality`, high `adaptation_strategy_quality`, high `reflective_learning_orientation`, medium/high `persistence_recovery`.
