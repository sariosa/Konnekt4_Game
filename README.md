# Konnekt4_Game

Can reinforcement learning learn to play Connect 4 better than simple rule-based reasoning?

This project compares three approaches for playing Connect 4:

Random vs Q-Learning vs Heuristic

The project includes:

- A Gymnasium environment (`connect4_env.py`)
- Tabular Q-learning training and evaluation (`qlearning.py`)
- A console game interface (`play_console.py`)

---

# 1) Get the project files

### Option A (no Git required)

1. Open the GitHub repository page  
2. Click **Code**  
3. Click **Download ZIP**  
4. Unzip the folder  
5. Move the folder somewhere easy to access (for example Desktop)

### Option B (with Git)

```
git clone https://github.com/felipe-a7/Konnekt4_Game.git
cd Konnekt4_Game
```

---

# 2) Open Terminal in the project folder

If you downloaded the ZIP file:

```
cd /path/to/Konnekt4_Game
```

---

# 3) Create and activate a virtual environment

### macOS / Linux

```
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

# 4) Install dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# 5) Run the project

## Train a Q-learning agent

```
python qlearning.py --train --episodes 50000
```

This creates:

- `q_table.pkl`
- `learning_curve.png`

---

## Evaluate the trained agent

```
python qlearning.py --eval --games 500 --qfile q_table.pkl
```

This creates:

- `evaluation.png`

---

## Play console modes

```
python play_console.py
```

Available modes:

- Human vs Human  
- Human vs Random  
- Human vs Heuristic  
- Train Q-learning then Human vs Q-learning  

---

# 6) Common issues

### ModuleNotFoundError

This usually means the wrong Python interpreter is being used.

Check:

```
which python
```

It should point to `.venv/...`.

If not, reactivate the virtual environment:

```
source .venv/bin/activate
```

Then run the scripts again:

```
python qlearning.py --train
```
