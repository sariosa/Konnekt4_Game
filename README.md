# Connect-4 Q-Learning Project

This project implements a Connect-4 environment, baseline opponent policies, a feature-based Q-learning agent, and a test script for checking robustness / overfitting behavior.

## Project files

- `connect4_env.py`  
  Gymnasium-style Connect-4 environment.

- `policies.py`  
  Baseline policies:
  - random policy
  - heuristic policy
  - base tabular Q-learning policy class

- `qlearning.py`  
  Feature-based Q-learning training and evaluation script.

- `test_overfitting.py`  
  Extra evaluation script for robustness / generalization testing.

- `play.py`  
  Optional console play script.

---

## Requirements

Install Python 3.10+ and the required packages.

### Recommended
Create and activate a virtual environment first.

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activat
```
