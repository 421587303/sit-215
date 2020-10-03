# Requirements

  - gym

  - numpy

  - pytorch

# How to Setup

```
pip install -r requirements.txt
```

# How to Use

```
# Q-learning for Taxi

python src/taxi.py --train

# Evaluate Q-learning Model for Taxi

python src/taxi.py --test

# Radom Baseline

python src/taxi.py --random

# Q-learning for CartPole

python src/cartpole.py --train

# Evaluate Q-learning Model for CartPole

python src/cartpole.py --test

# Random Baseline

python src/cartpole.py --random
```

After convergence, you can interrupt the learning by `Ctrl + C`, and then the script will save the model into the current directory.

# Note

  - Evaulate Q-learning in Linux is better, because Taxi need to be printed in console using the Linux color formatter.

  - It is better to monitor the training manually in that you can early stop the training once seeing the convergences.

  - You can directly evaluate the training result, because the pre-trained weights have already been provided.