# Impact of Normalization in Deep RL

This repository accompanies the course project "Impact of Normalization in Deep RL", submitted as part of the Reinforcement Learning course at the University of Zurich (Fall 2025).

The project provides a systematic empirical study of how different normalization, scaling, and clipping techniques influence learning stability, convergence, and performance in Deep Reinforcement Learning (DRL). The work is inspired by and builds upon:

> [Schaul et al., Return-based Scaling: Yet Another Normalisation Trick for Deep RL (2021)](https://arxiv.org/abs/2105.05347)

Our study evaluates these techniques across two policy-gradient algorithms (PPO, A2C) and two Atari environments (Pacman, Gravitar) using a controlled experimental setup.

# Overview

**Research Question**

How does parameter normalization influence the learning performance and stability of a Deep Reinforcement Learning agent?

**Objectives**

Analyze the effect of normalization on training stability and variance

- Study convergence speed and asymptotic performance under different techniques.
- Compare algorithm sensitivity (PPO vs A2C) to normalization
- Evaluate generalizability across environments with different reward structures

# Methods: Algorithms

The following on-policy policy-gradient methods are implemented:

- Proximal Policy Optimization (PPO): Uses a clipped surrogate objective to constrain policy updates and improve robustness.

- Advantage Actor-Critic (A2C): A synchronous actor–critic method without an explicit trust region, making it more sensitive to optimization instability.

PPO is the primary algorithm of interest, while A2C serves as a comparative baseline.

# Experimental Domains

Experiments are conducted on two environments from the Arcade Learning Environment (ALE) via Gymnasium:

- Pacman: Dense rewards with relatively structured dynamics and smaller action space.
- Gravitar: Sparse rewards with complex physics (larger action space) and high variance.

# Methods: Normalization and Stabilization Techniques

The following techniques are evaluated:

**Normalization**

- Observation Normalization
- Advantage Normalization
- Return Normalization
- Reward Normalization

**Scaling**

- Return-based Scaling (Schaul et al., 2021)

**Clipping**

- Gradient Clipping (max-norm)
- Reward Clipping (dynamic, statistics-based)

In addition to individual techniques, selected cross-combinations are evaluated:

- Advantage Normalization + Gradient Clipping
- Observation Normalization + Advantage Normalization + Gradient Clipping
- Reward Clipping + Advantage Normalization + Gradient Clipping

# Experimental Setup

- Framework: PyTorch + Gymnasium
- Architecture: Atari-style CNN (as in Mnih et al., 2013)
- Algorithms: PPO and A2C
- Seeds: 3 (10, 20, 30)

| Command | PPO | A2C |
| :--- | --- | --- |
| Learning Rate | 2.5e-2 | 2.5e-4 |
| Gamma ($\gamma$) | 0.997 | 0.997 |
| Entropy Coefficient | 0.001 | 0.001 |
| Reward Clip Range | $\pm2$ | $\pm2$ |
| Gradient Clip Range | 5 | 5 |
| Batch Size | 10 | 10 |

All experiments use matched architectures and settings to isolate the effect of normalization.

# Results

**PPO**

- Demonstrates strong inherent robustness due to its clipped objective
- External normalization often provides marginal or no benefit
- Observation normalization improves early learning but not final performance
- [Loss and Reward Plots](ppo)

**A2C**

- Highly sensitive to optimization instability
- Fails to learn in complex environments without stabilization
- Gradient Clipping + Advantage Normalization yields the best performance, especially in Gravitar
- [Loss and Reward Plots](a2c)

**Reward Clipping**

- Dynamic reward clipping introduces instability and high variance
- Non-stationary clipping thresholds disrupt advantage estimation
- These findings align with recent literature showing that normalization effectiveness is algorithm-dependent
- [Loss and Reward Plots](reward_clip_plots)

# Reproducibility & Evaluation Protocol

- Deterministic execution given a fixed random seed
- All results are averaged over 3 seeds
- Vanilla (unnormalized) PPO/A2C serves as the primary baseline
- Performance is reported relative to this baseline

# Citation

If you use this codebase in your work, please cite:
```
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@article{schaul2021return,
  title={Return-based scaling: Yet another normalisation trick for deep rl},
  author={Schaul, Tom and Ostrovski, Georg and Kemaev, Iurii and Borsa, Diana},
  journal={arXiv preprint arXiv:2105.05347},
  year={2021}
}
}
```

# References

- Schaul et al., Return-based Scaling: Yet Another Normalisation Trick for Deep RL
- Schulman et al., Proximal Policy Optimization Algorithms
- Mnih et al., Asynchronous Methods for Deep Reinforcement Learning

# Contributors
- Anoozh Akileswaran
- Fransiskus Adrian Gunawan
- Manan Tejas Sodha
- Shuvam Ganguli
- Harsh Laxmikant Mewada

# Acknowledgements
- Course: Reinforcement Learning
- Institution: University of Zurich
- Semester: Fall 2025
- Professor: Dr. Giorgia Ramponi
- Project Supervisor: Ziang Liu
