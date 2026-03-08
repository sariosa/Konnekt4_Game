Step 1: train
     python qlearning.py --train --episodes 300000


Step 2: evaluate
     python qlearning.py --eval --qfile q_table.pkl


Step 3: test robustness
     python test_overfitting.py --qfile q_table.pkl --games 300 --seeds 5


Step 4:  run the play_console.py to see the game menu

* game menu option 5 (Human vs Q-learning) is using the Q-learning table you trained and stored in step 1 and 2

---------------------------------------------------------------------------------------------------------------------------

Main command-line arguments

qlearning.py
--train : run training
--eval : run evaluation
--episodes : number of training episodes
--games : number of evaluation games
--seed : random seed
--print_every : progress print interval during training
--qfile : path to saved Q-table
--curve : output file for learning curve image


test_overfitting.py
--qfile : path to saved Q-table
--games : games per seed per opponent
--seeds : number of random seeds
--gap_threshold : threshold for generalization-gap warning


Example full run:
python qlearning.py --train --episodes 300000
python qlearning.py --eval --qfile q_table.pkl
python test_overfitting.py --qfile q_table.pkl --games 300 --seeds 5


Clean restart:
       rm q_table.pkl
