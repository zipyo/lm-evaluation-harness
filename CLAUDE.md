# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Language Model Evaluation Harness (lm-eval) is a unified framework for evaluating generative language models on academic benchmarks. It powers Hugging Face's Open LLM Leaderboard and supports 60+ standard benchmarks with hundreds of subtasks.

## Common Commands

### Installation
```bash
pip install -e "."                    # Base install
pip install -e ".[dev,hf]"            # Development with HuggingFace backend
pip install -e ".[vllm]"              # vLLM backend
pip install -e ".[api]"               # API models (OpenAI, Anthropic, etc.)
```

### Running Tests
```bash
pytest                                # Run all tests
pytest tests/test_evaluator.py       # Run single test file
pytest tests/ -k "test_name"         # Run specific test by name
pytest -x                            # Stop on first failure
pytest --cov=lm_eval                 # With coverage
pytest -n=auto                       # Run in parallel
```

### Linting
```bash
pre-commit run --all-files           # Run all pre-commit hooks
ruff check .                         # Lint only
ruff check --fix .                   # Lint with auto-fix
ruff format .                        # Format code
```

### Running Evaluations
```bash
lm-eval ls tasks                     # List available tasks
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --limit 10
lm-eval validate --tasks hellaswag   # Validate task config
```

## Architecture

### Core Components

**Entry Points:**
- `lm_eval/__main__.py` → CLI entry point
- `lm_eval/_cli/` → CLI subcommands (run, ls, validate)
- `lm_eval.simple_evaluate()` and `lm_eval.evaluate()` → Python API

**Model Layer (`lm_eval/api/model.py`, `lm_eval/models/`):**
- `LM` base class defines the interface all models must implement
- Three request types: `loglikelihood`, `loglikelihood_rolling`, `generate_until`
- Models register via `@register_model` decorator or `MODEL_MAPPING` in `models/__init__.py`
- Lazy loading via registry for fast CLI startup

**Task Layer (`lm_eval/api/task.py`, `lm_eval/tasks/`):**
- `Task` base class and `ConfigurableTask` for YAML-defined tasks
- Tasks are YAML configs in `lm_eval/tasks/<task_name>/` subdirectories
- `TaskConfig` dataclass in `lm_eval/config/task.py` defines all task parameters
- `TaskManager` in `lm_eval/tasks/__init__.py` handles task discovery and loading

**Registry System (`lm_eval/api/registry.py`):**
- Central registration for models, metrics, filters, aggregations
- Supports lazy loading via string targets (e.g., `"module.path:ClassName"`)
- Key registries: `model_registry`, `metric_registry`, `filter_registry`

**Evaluation Flow (`lm_eval/evaluator.py`):**
- `simple_evaluate()` → instantiates model + tasks, calls `evaluate()`
- `evaluate()` → runs evaluation loop, collects results
- `evaluator_utils.py` → result consolidation and formatting

### Task Configuration

Tasks are primarily YAML-based. Key fields:
```yaml
task: task_name
dataset_path: huggingface/dataset    # HF dataset or path
dataset_name: subset_name            # Dataset config/subset
test_split: test                     # Split to evaluate on
doc_to_text: "{{question}}"          # Jinja2 template for input
doc_to_target: "{{answer}}"          # Jinja2 template for target
metric_list:                         # Metrics to compute
  - metric: acc
```

Custom processing via `!function utils.process_docs` pointing to Python functions.

### Adding New Components

**New Task:** Create `lm_eval/tasks/<name>/<name>.yaml` with task config. See `templates/new_yaml_task/` for template.

**New Model:**
1. Create `lm_eval/models/<name>.py` implementing `LM` interface
2. Add `@register_model("name")` decorator or entry in `MODEL_MAPPING`
3. Implement `loglikelihood()`, `loglikelihood_rolling()`, `generate_until()`

**New Metric:** Use `@register_metric` decorator in `lm_eval/api/metrics.py`

## Key Files

- `lm_eval/evaluator.py` - Main evaluation logic
- `lm_eval/api/task.py` - Task base classes and ConfigurableTask (~2000 lines)
- `lm_eval/api/model.py` - LM base class interface
- `lm_eval/models/huggingface.py` - Reference model implementation
- `lm_eval/tasks/__init__.py` - TaskManager and task loading
- `lm_eval/config/task.py` - TaskConfig dataclass

## Environment Variables

- `LMEVAL_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `LM_HARNESS_CACHE_PATH` - Cache directory for requests
- `HF_TOKEN` - HuggingFace token for private models/datasets
- `HF_DATASETS_OFFLINE=1` - Enable offline mode for datasets
- `HF_HUB_OFFLINE=1` - Enable offline mode for HuggingFace Hub
- `TRANSFORMERS_OFFLINE=1` - Enable offline mode for transformers

## Important Notes

- The library supports both local models (via HuggingFace, vLLM, etc.) and API-based models (OpenAI, Anthropic, etc.)
- Multi-GPU evaluation is supported through data parallelism (accelerate) or model parallelism (device_map)
- Task configurations use Jinja2 templating for flexible prompt construction
- Results are deterministic given the same seed values (random_seed, numpy_random_seed, torch_random_seed, fewshot_random_seed)

## Language and Communication Guidelines

- 코드 및 전문용어, 대명사 등을 제외한 언어는 한국어를 사용
- 커밋 메시지는 한국어로 작성하고, CLAUDE에 관한 정보는 제외
