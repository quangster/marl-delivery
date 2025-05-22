colab evaluation: https://colab.research.google.com/drive/1cWQhg1L42NTQ7nPvGq0keMopMpPD8MLo?usp=sharing

# marl-delivery
MARL samples code for Package Delivery.
You has to run and test against BFS agents for the following 5 configs.
The seeds are given at later time.

- Version v1.1: Fix a small logic of `start_time` should less than `dead_line`


# Testing scripts
```python main.py --seed 10 --max_time_steps 1000 --map map1.txt --num_agents 5 --n_packages 100```

```python main.py --seed 10 --max_time_steps 1000 --map map2.txt --num_agents 5 --n_packages 100```

```python main.py --seed 10 --max_time_steps 1000 --map map3.txt --num_agents 5 --n_packages 500```

```python main.py --seed 10 --max_time_steps 1000 --map map4.txt --num_agents 10 --n_packages 500```

```python main.py --seed 10 --max_time_steps 1000 --map map5.txt --num_agents 10 --n_packages 1000```

# For RL testing
- You can use `simple_PPO.ipynb` as the starting point.
- Avoid modify the class `Env`, you can try to modify the `convert_state` function or `reward_shaping`
- You can choose to use or change the standard `PPO`. Note that: It is not easy to match the greedy agent, using RL.


# TODO:
- [x]: Add BFS agents
- [x]: Add test scripts
- [x]: Add RL agents
