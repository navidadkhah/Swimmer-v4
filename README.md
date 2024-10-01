# Gymnasium Swimmer Environment

This project showcases the implementation of two versions of Proximal Policy Optimization (PPO) algorithms — **PPO-AdaptiveKL** and **PPO-Clip** — applied to the Swimmer environment in the Gymnasium framework. These implementations help to explore the balance between policy update stability and efficiency in continuous control tasks.

## Overview

- **PPO-Clip**: This is the more commonly used version of PPO. It uses a clipping mechanism to restrict the policy update ratio, preventing large changes that might destabilize learning. This approach is simple yet effective, yielding stable training results.
  
- **PPO-AdaptiveKL**: This variant introduces a penalty on the KL divergence between the old and new policies. The KL threshold is adapted dynamically during training, allowing more flexibility when updates are safe and restricting them when needed.

## PPO-Clip Algorithm

The **PPO-Clip** algorithm limits how much the policy can change by clipping the probability ratio between the new and old policies. This ensures smoother updates and stabilizes learning.
For determine the policy parameter, we use this formula:
<p align="center">
  <img alt="Policy Formula" src="https://github.com/user-attachments/assets/3ab52b7a-583f-49e4-987d-408e41f14212" width="400">
</p>

In this formula ```L``` is:
<p align="center">
  <img alt="About the L" src="https://github.com/user-attachments/assets/b6cd1977-c78c-465c-9012-70acaee50071" width="400">
</p>

### Reward Progression

<p align="center">
  <img alt="PPO-Clip Reward" src="https://github.com/user-attachments/assets/30b83b8e-c98a-4353-9325-59cdccf2ca45" width="400">
</p>
<p align="center">
  
  *Figure: PPO-Clip reward evolution during training episodes. The steady rise shows the model gradually improving policy performance while maintaining stability.*
</p>


https://github.com/user-attachments/assets/aa7d09d6-9946-423c-a94f-c025161b66d5


## PPO-AdaptiveKL Algorithm

In contrast, **PPO-AdaptiveKL** adjusts the KL divergence penalty dynamically. This allows larger updates when the divergence between the policies is small, and restricts updates when the divergence grows too large, making learning more flexible.

### Reward Progression

<p align="center">
  <img alt="PPO-Adaptive Reward" src="https://github.com/user-attachments/assets/7e3ea5ce-d6b7-41da-96d7-ac463bd0cb95" width="400">
</p>
<p align="center">
  
  *Figure: PPO-AdaptiveKL reward progression. The adaptive nature of the KL divergence threshold results in more fluctuation but eventually converges to a stable reward trajectory.*
</p>


https://github.com/user-attachments/assets/8b9ab3b9-d2e7-4adc-ad03-981c596ab796

## Results Summary

Both algorithms are effective in controlling the swimmer, but they exhibit different characteristics:

- **PPO-Clip** maintains more stable and predictable updates, with smoother convergence.
- **PPO-AdaptiveKL** introduces more flexibility by adapting to the policy's behavior, allowing for potentially faster learning, but at the cost of occasional instability.

For a more detailed discussion of the algorithms, mathematical details, and hyperparameters, please refer to the full report.

## License

This project is under the MIT License, and I’d be thrilled if you use and improve my work!
