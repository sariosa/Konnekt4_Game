# Konnekt4_Game
Can reinforcement learning learn to play Connect 4 better than simple rule-based reasoning?
Random Vs Q-Learning Vs Heuristic

Connect 4 project with:
- a Gymnasium environment (`connect4_env.py`)
- Q-learning training/evaluation (`qlearning.py`)
- rule-based and random policies (`policies.py`)
- a console game menu (`play_console.py`)

## 1) Get the project files

### Option A (easiest, no Git needed)
1. Open the GitHub repository page.
2. Click **Code**.
3. Click **Download ZIP**.
4. Unzip it.
5. Move the folder somewhere easy to find (for example Desktop).

### Option B (with Git)
```bash
git clone https://github.com/felipe-a7/Konnekt4_Game.git
cd Konnekt4_Game
```

## 2) Open Terminal in the project folder

If you downloaded ZIP, go into that folder first:
```bash
cd /path/to/Konnekt4_Game
```

## 3) Create and activate a virtual environment

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5) Run the project

### Train a Q-learning agent
```bash
python qlearning.py --train --episodes 50000
```

This creates:
- `q_table.pkl`
- `learning_curve.png`

### Evaluate the trained agent
```bash
python qlearning.py --eval --games 500 --qfile q_table.pkl
```

This creates:
- `evaluation.png`

### Train and evaluate in one command
```bash
python qlearning.py --train --eval --episodes 50000 --games 500
```

### Play console modes
```bash
python play_console.py
```

Menu options in `play_console.py`:
1. Human vs Human
2. Human vs Random
3. Human vs Heuristic
4. Train Q-learning then Human vs Q-learning
5. Exit

Important behavior:
- There is no `--play` mode in `qlearning.py`.
- Menu option 4 trains a fresh Q-learning agent for 50,000 episodes before playing.

## 6) Common issues

### `ModuleNotFoundError`
You are usually running the wrong Python interpreter.

Use:
```bash
which python
```

It should point to `.venv/...`. If not, reactivate:
```bash
source .venv/bin/activate
```

Then run scripts with:
```bash
python qlearning.py --train
```
