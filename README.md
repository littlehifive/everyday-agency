# Measuring Agency in My ChatGPT Conversations

**Zezhen Wu | The Agency Fund**

This project analyzes everyday signals of human agency in human-AI interactions using snippets from my exported ChatGPT conversations.

To try it on your own, make a privacy request to download your ChatGPT history via [OpenAI's privacy portal](https://privacy.openai.com/policies/en/). You will receive an email from OpenAI with a link to download your full chat history. In the download zip file, you will see a large HTML file named `chat.html`, which contains all text data.

I mainly used Google's [langextract](https://github.com/google/langextract) for labelling signals of agency (e.g., goal-setting) in my queries, and OpenAI's [Gabriel](https://openai.com/index/scaling-social-science-research/) for constructing rubrics for scoring how much personal, proxy, or collective agency is manifested in my queries.

## What Has Been Built So Far

### 1) Data extraction and preprocessing

- `parse_chat.py` parses the exported ChatGPT HTML file (`data/chat.html`) into message-level parquet (`data/chat_messages.parquet`).
- Messages are grouped into conversation-level user corpora for downstream coding.

### 2) LangExtract analysis pipeline (`analysis/langextract`)

Implemented end-to-end pipeline:

1. `01_prepare_conversation_corpus.py`
2. `02_select_agency_candidates.py`
3. `03_build_langextract_examples.py`
4. `04_extract_agency_langextract.py`
5. `05_build_agency_features.py`

This pipeline generates:

- candidate diagnostics
- redacted snippet examples
- construct-level extraction outputs
- conversation-level agency flags/counts

Primary outputs are written to `data/derived/langextract`.

### 3) Gabriel analysis pipeline (`analysis/gabriel`)

Implemented separate end-to-end pipeline:

1. `01_prepare_conversation_corpus_gabriel.py`
2. `02_select_agency_candidates_gabriel.py`
3. `03_rate_agency_gabriel.py`
4. `04_build_agency_features_gabriel.py`
5. `05_validate_agency_rubric_gabriel.py`

Current Gabriel rubric uses 3 conversation-level agency dimensions:

- `personal_agency`
- `proxy_agency`
- `collective_agency`

Gabriel outputs are isolated under `data/derived/gabriel`.

## Reports Created

- LangExtract report: `docs/langextract/agency_signals_report.qmd` and rendered HTML.
- Gabriel report: `docs/gabriel/gabriel_agency_report.qmd` and rendered HTML.

The Gabriel report currently includes:

- definitions and score construction for personal/proxy/collective agency
- score distributions and summary stats
- pairwise scatterplots with fitted lines and correlations
- a section that reuses LangExtract redacted evidence examples and appends Gabriel scores (filtered to rows with non-null Gabriel average agency score)

## Typical Run Commands

### LangExtract

```bash
python analysis/langextract/01_prepare_conversation_corpus.py
python analysis/langextract/02_select_agency_candidates.py
python analysis/langextract/03_build_langextract_examples.py
python analysis/langextract/04_extract_agency_langextract.py
python analysis/langextract/05_build_agency_features.py
```

### Gabriel

```bash
python analysis/gabriel/01_prepare_conversation_corpus_gabriel.py
python analysis/gabriel/02_select_agency_candidates_gabriel.py
python analysis/gabriel/03_rate_agency_gabriel.py --n-runs 2
python analysis/gabriel/04_build_agency_features_gabriel.py
python analysis/gabriel/05_validate_agency_rubric_gabriel.py
```

## Notes

- `03_rate_agency_gabriel.py` requires an environment where `openai-gabriel` is installed.
- API credentials are read from `.env` when present. Please set up your OPENAI_API_KEY there.
- A `data/` folder that contains `chat.html` is gitignored.
