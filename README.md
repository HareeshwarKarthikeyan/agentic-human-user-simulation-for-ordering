# Agentic Persona Control and Task State Tracking for Realistic User Simulation in Interactive Scenarios

## Overview

This repository contains the supplementary materials for the paper "Agentic Persona Control and Task State Tracking for Realistic User Simulation in Interactive Scenarios" submitted to the Workshop on Scaling Environments for Agents. The codebase implements a three-agent architecture for simulating realistic human users in conversational ordering scenarios, achieving superior task completion accuracy and persona adherence compared to single-agent baselines.

## Architecture 

Our multi-agent framework employs structured collaboration between specialized agents to achieve realistic user simulation:

![Architecture Figure](./Architecture%20Figure.jpg)

**Figure 1: Multi-agent architecture for human user simulation** showing the three-agent framework:
1. **User Agent** (`agents/guest_agent.py`) - Primary orchestrator generating simulated user responses
2. **State Tracking Agent** (`agents/order_tracking_agent.py`) - Maintains structured task state representation  
3. **Message Attributes Generation Agent** (`agents/message_generation_agent.py`) - Determines behavioral characteristics based on persona

## Reproducibility Instructions

This repository provides all code and data necessary to reproduce the experimental results presented in the paper. The codebase is structured to facilitate easy replication of all experiments and ablation studies.

### 1. Prerequisites

```bash
# Install uv package manager (https://github.com/astral-sh/uv)
pip install uv

# Clone the repository
git clone [repository-url]
cd agentic-human-guest-simulation-for-ordering
```

### 2. Environment Setup

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your-key-here
# OPENAI_MODEL=gpt-4o
```

## Reproducing Experimental Results

### Test Case Generation

The test cases used in our experiments are pre-generated and available in `test_case_generation/data/`:
- **personas.json** - 20 diverse customer personas with distinct behavioral traits
- **order_test_cases.json** - 60 ordering scenarios (3 per persona)
- **menu.json** - Restaurant menu used for all experiments
- **message_generation_attributes_for_personas.json** - Persona-specific behavioral attributes

To regenerate test cases, use the prompts in `test_case_generation/prompts/` with Claude or GPT-4.

### Running Experiments

We provide 5 experiments corresponding to the configurations in the paper:

#### Experiment 1: Baseline (Single LLM)
```bash
cd experiments/exp1
uv run run_exp_1.py
```

#### Experiment 2: Full Multi-Agent System
```bash
cd experiments/exp2
uv run run_exp_2.py
```

#### Experiment 3: Ablation - Without State Tracking
```bash
cd experiments/exp3
uv run run_exp_3.py
```

#### Experiment 4: Ablation - Without Message Attributes
```bash
cd experiments/exp4
uv run run_exp_4.py
```

#### Experiment 5: Ablation - Without Both Components
```bash
cd experiments/exp5
uv run run_exp_5.py
```

Each experiment runs 60 test cases (20 personas × 3 scenarios) and generates conversation logs in the respective `logs/` directory.

### Evaluating Results

After running experiments, evaluate the results using our analysis pipeline:

```bash
cd evaluations

# Extract states and calculate metrics for all experiments
uv run batched_state_extractor.py

# Run statistical analysis and generate figures
uv run run_ablation_analysis.py

# Analyze costs and token usage
uv run cost_analyzer.py
```

### Output Files

Results are saved in `evaluations/results/`:

#### Metrics & Statistics
- **summary_statistics.csv** - Performance metrics for all experiments
- **statistical_report.json** - Detailed statistical analysis including t-tests
- **ablation_results.json** - Component-wise performance impact

#### Visualizations 
- **figures/metric_comparison.png** - Bar charts comparing metrics across experiments
- **figures/improvement_heatmap.png** - Heatmap showing improvements over baseline
- **figures/significance_plot.png** - Statistical significance visualization
- **figures/cost_analysis_comparison.png** - Cost and latency comparisons

#### Extracted States
- **extracted_states/exp[1-5]/all_conversations_states.csv** - Final states for each conversation
- Individual conversation state files for detailed analysis

## Key Results

Our multi-agent architecture demonstrates:
- **24.5% improvement** in task completion accuracy over single-agent baseline
- **14.7% better** persona adherence scores
- Statistically significant improvements (p < 0.01) across all metrics
- State tracking agent contributes most to performance gains

## Repository Structure

```
├── agents/                 # Core agent implementations
├── experiments/           # Experiment configurations (exp1-5)
├── evaluations/          # Evaluation scripts and results
├── test_case_generation/ # Test data and generation prompts
├── paper/               # Research paper draft
└── scripts/            # Utility scripts for running simulations
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{agentic-persona-2025,
  title={Agentic Persona Control and Task State Tracking for Realistic User Simulation in Interactive Scenarios},
  author={[Authors]},
  booktitle={Workshop on Scaling Environments for Agents},
  year={2025}
}
```

## Contact

For questions about reproducing results or implementation details, please open an issue on GitHub.

