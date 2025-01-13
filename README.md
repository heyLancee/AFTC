# AFTC

The project is a active fault-tolerant control method for satellite attitude control system using reinforcement learning.

The fault scenario considered in this project is actuator fault with the fault model and fault params set in the `satellite.py`.

The reinforcement learning algorithm used in this project is TD3, and the training models and results are in the `models` and `results` folders, respectively.

To run the project convieniently, you can use the `run_experiments.sh` file in `scripts` folder.

By the way, the project uses a dynamic network to fit the dynamics of the satellite, which can be found in `DanamicNet.py`.

