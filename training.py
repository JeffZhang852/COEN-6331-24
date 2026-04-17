# Distributed Deep Reinforcement Learning Training Script
#
# Trains a shared policy for all 10 houses using PPO.
# Events like fire, BBQ, smoking, and FDI are randomly injected during training
# To add or change, environment/iot_env.py needs to be changed for the training
# to teach the agent the appropriate defensive responses.
import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.iot_env import IoTDefenderEnv, load_baseline_data, EventManager, \
    EVENT_FIRE, EVENT_BBQ, EVENT_SMOKING, EVENT_FDI_TEMP, EVENT_FDI_PM, NUM_HOUSES

# 1) CONFIGURATION
DATA_FILE = "environment/environment_data_30days.txt"
MODEL_SAVE_PATH = "models/iot_ddrl_model"          # final trained model
CHECKPOINT_PREFIX = "iot_cp"                       # checkpoint files will be named iot_cp_XXXXXX_steps

# Training hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Episode settings
STEPS_PER_EPISODE = 96 * 2      # 2 days per episode, can be changed
EVENT_PROBABILITY = 0.15         # chance of injecting an event per house per check


# 2) EVENT INJECTION FOR TRAINING
def create_training_event_manager(total_steps, num_houses=NUM_HOUSES, event_prob=0.15):
    # Build an EventManager with randomly placed events.
    # This makes sure the agent sees many different situations while learning.
    event_manager = EventManager()
    rng = np.random.RandomState(42)

    event_types = [EVENT_FIRE, EVENT_BBQ, EVENT_SMOKING, EVENT_FDI_TEMP, EVENT_FDI_PM]
    event_weights = [0.05, 0.2, 0.2, 0.25, 0.3]  # fewer fires, more FDI attacks

    for step in range(0, total_steps, 10):
        for house in range(num_houses):
            if rng.rand() < event_prob:
                etype = rng.choice(event_types, p=event_weights)
                duration = 1
                params = {}
                if etype == EVENT_FIRE:
                    duration = rng.randint(6, 12)      # 1.5 to 3 hours
                elif etype == EVENT_BBQ:
                    duration = rng.randint(2, 6)       # 0.5 to 1.5 hours
                elif etype == EVENT_SMOKING:
                    duration = rng.randint(1, 3)       # 15 to 45 minutes
                elif etype == EVENT_FDI_TEMP:
                    params['fake_val'] = rng.uniform(80, 120)
                elif etype == EVENT_FDI_PM:
                    params['fake_val'] = rng.uniform(150, 400)
                event_manager.add_event(step, house, etype, duration, params)
    return event_manager


# 3) ENVIRONMENT WRAPPERS
def make_env(house_id, baseline_data, event_manager, max_steps):
    # Factory that creates a monitored environment for a specific house.
    def _init():
        env = IoTDefenderEnv(
            baseline_data=baseline_data,
            house_id=house_id,
            event_manager=event_manager,
            max_steps=max_steps
        )
        env = Monitor(env)  # logs episode rewards
        return env
    return _init


def create_vectorized_env(baseline_data, event_manager, max_steps, num_envs=NUM_HOUSES):
    # Create one environment per house, running in parallel.
    env_fns = [make_env(i, baseline_data, event_manager, max_steps) for i in range(num_envs)]
    if num_envs > 1:
        try:
            return SubprocVecEnv(env_fns)
        except:
            return DummyVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


# 4) TRAINING CALLBACKS
def get_callbacks(save_path):
    # Set up checkpoint saving.
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix=CHECKPOINT_PREFIX
    )
    # Optional evaluation environment can be added here
    eval_env = None
    if eval_env:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        return [checkpoint_callback, eval_callback]
    return [checkpoint_callback]


# 5) MAIN TRAINING FUNCTION
def train(total_timesteps=500000, model_save_path=MODEL_SAVE_PATH):
    print("Loading baseline data...")
    baseline = load_baseline_data(DATA_FILE)
    total_baseline_steps = baseline.shape[0]
    print(f"Baseline data shape: {baseline.shape}")

    print("Generating training event schedule...")
    event_manager = create_training_event_manager(
        total_steps=total_baseline_steps,
        num_houses=NUM_HOUSES,
        event_prob=EVENT_PROBABILITY
    )

    max_episode_steps = min(STEPS_PER_EPISODE, total_baseline_steps)

    print(f"Creating vectorized environment with {NUM_HOUSES} parallel houses...")
    env = create_vectorized_env(baseline, event_manager, max_episode_steps, NUM_HOUSES)

    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        seed=42
    )

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    callbacks = get_callbacks(model_save_path)

    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    env.close()
    print("Training complete.")


# 6) EVALUATION (OPTIONAL)
def evaluate(model_path, num_episodes=10):
    # Load a trained model and test it on clean baseline data.
    from stable_baselines3 import PPO

    baseline = load_baseline_data(DATA_FILE)
    event_manager = EventManager()  # no events during evaluation

    model = PPO.load(model_path)

    total_rewards = []
    for episode in range(num_episodes):
        env = IoTDefenderEnv(baseline, house_id=0, event_manager=event_manager, max_steps=96 * 2)
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} +/ {np.std(total_rewards):.2f}")


# 7) COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--save_model", type=str, default=MODEL_SAVE_PATH, help="Path to save final model")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate a saved model (provide path)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.episodes)
    else:
        train(total_timesteps=args.timesteps, model_save_path=args.save_model)