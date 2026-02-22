# Gabriel Agency Pipeline

This pipeline is fully isolated from `analysis/langextract`.

Rubric scope:
- `personal_agency`
- `proxy_agency`
- `collective_agency`

## Outputs
All generated outputs are written under:

`/Users/michaelfive/Code/mystuff/everyday-agency/data/derived/gabriel`

No script in this folder should write to `data/derived/langextract`.

## Scripts
1. `01_prepare_conversation_corpus_gabriel.py`
2. `02_select_agency_candidates_gabriel.py`
3. `03_rate_agency_gabriel.py`
4. `04_build_agency_features_gabriel.py`
5. `05_validate_agency_rubric_gabriel.py`

## Typical Run
```bash
python analysis/gabriel/01_prepare_conversation_corpus_gabriel.py
python analysis/gabriel/02_select_agency_candidates_gabriel.py
python analysis/gabriel/03_rate_agency_gabriel.py --n-runs 2
python analysis/gabriel/04_build_agency_features_gabriel.py
python analysis/gabriel/05_validate_agency_rubric_gabriel.py
```

## Pilot Run
```bash
python analysis/gabriel/03_rate_agency_gabriel.py --limit 120 --n-runs 3 --reset-files
```

## Notes
- `03_rate_agency_gabriel.py` requires an environment where `openai-gabriel` is installed and importable as `gabriel`.
- API key is loaded from `.env` if present.
