# Agency Signal Extraction Task

Extract short spans that indicate everyday agency in user-authored text.

## Valid extraction classes
- `goal_setting`: Explicit statement of a goal, objective, or desired outcome.
- `goal_navigation`: Concrete movement toward a goal through sequencing, prioritizing, or adjusting steps.
- `problem_decomposition`: Breaking a larger challenge into smaller tasks, causes, or sub-problems.
- `strategic_planning`: Choosing a method, plan, or strategy before acting.
- `progress_monitoring`: Checking progress, milestones, status, or performance against expectations.
- `obstacle_management`: Identifying and working through blockers, errors, or constraints.
- `help_seeking_resourcefulness`: Requesting tools, advice, references, or support to move forward.
- `self_efficacy_or_confidence`: Language signaling confidence in ability to execute or learn.
- `resilience_or_persistence`: Continuing effort despite setbacks; retrying, adapting, or persevering.

## Rules
- Extract only text that appears verbatim in the input.
- Keep spans minimal while preserving meaning.
- Use only the listed classes.
- Include multiple classes if the text clearly supports them.
- If no agency signal is present, return no extractions.
