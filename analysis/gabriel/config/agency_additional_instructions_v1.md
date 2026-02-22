Measure agency as direct evidence in the user text only.

Core scoring requirements:
- Score each attribute independently.
- Do not infer one attribute from another.
- Do not infer agency from topic alone.
- Use only direct language evidence from the text itself.
- Use these three agency modes only: `personal_agency`, `proxy_agency`, `collective_agency`.

Calibration anchors:
- 0 to 10: effectively absent.
- 20 to 40: weak / occasional signal.
- 45 to 60: moderate / mixed signal.
- 65 to 85: strong consistent signal.
- 90 to 100: extreme and unambiguous signal (rare).

Additional rules:
- For `personal_agency`, focus on self-initiated control and capability (ownership, intention, "I will do X").
- For `proxy_agency`, score high only when the user explicitly relies on another agent's competence/control to progress.
- For `collective_agency`, score high only when there is explicit multi-actor coordination and shared team efficacy.
- Do not score `collective_agency` high just because another actor is mentioned; require evidence of joint coordination.
- Avoid defaulting to midpoint values; use the full range where justified.

Short rubric examples:
1. Text: "I will draft the timeline today, run tests tomorrow, and submit by Friday."
   Expected pattern: high `personal_agency`, low `proxy_agency`, low `collective_agency`.
2. Text: "Use your analysis to pick the best portfolio and tell me exactly what to buy."
   Expected pattern: low/moderate `personal_agency`, high `proxy_agency`, low `collective_agency`.
3. Text: "I will coordinate with support and engineering while you triage logs so we can resolve this together."
   Expected pattern: moderate/high `personal_agency`, moderate `proxy_agency`, high `collective_agency`.
