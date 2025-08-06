import torch

config = {
    # Environment and Seed
    "env_id": "HalfCheetah-v5",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Training Parameters
    "num_train_steps": 500_000,      # Increased to 500k steps
    "start_timesteps": 10_000,       # Longer warmup for more stability
    "replay_buffer_size": 1_000_000, # Increased buffer size for 500k steps
    "batch_size": 256,
    "trans_optimization_epochs": 50, # Keep at 50, as in original config

    # Network Architecture
    "hidden_dim": 1024,              # Increased hidden dim, closer to paper for better learning
    "skill_dim": 16,
    "discrete_skills": True,

    # SAC Hyperparameters
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.01,

    # METRA-specific Hyperparameters
    "dual_reg": True,
    "dual_lam_init": 30.0,
    "dual_slack": 1e-3,
    "unit_length_skill": True,

    # HRL Parameters
    "skill_length": 50,

    # Logging and Evaluation
    "log_interval": 25_000,          # Log training curves every 25k steps
    "eval_interval": 100_000,        # Evaluate (and plot skills/zero-shot) every 100k steps
    "num_eval_skills": 8,

    # Goal reaching
    "enable_goal_reaching": True,
}