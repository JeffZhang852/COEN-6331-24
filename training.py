import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.iot_env import (
    IoTDefenderEnv, load_baseline_data, load_scenario,
    EventManager,
    EVENT_FIRE, EVENT_BBQ, EVENT_SMOKING, EVENT_FDI_TEMP, EVENT_FDI_PM,
    NUM_HOUSES,
)

DATA_FILE         = "environment/environment_data_30days.txt"
MODEL_SAVE_PATH   = "models/iot_ddrl_model"
CHECKPOINT_PREFIX = "iot_cp"

#PPO hyperparameters
LEARNING_RATE  = 3e-4
N_STEPS        = 2048
BATCH_SIZE     = 64
N_EPOCHS       = 10
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.01
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5

STEPS_PER_EPISODE = 96 * 2    # 2 simulated days per episode
EVENT_PROBABILITY = 0.15       # used only in random-injection mode

def create_random_event_manager(total_steps, num_houses=NUM_HOUSES,
                                 event_prob=EVENT_PROBABILITY):
    """build EventManager with randomly placed events (seed=42)"""
    event_manager = EventManager()
    rng = np.random.RandomState(42)

    event_types   = [EVENT_FIRE, EVENT_BBQ, EVENT_SMOKING, EVENT_FDI_TEMP, EVENT_FDI_PM]
    event_weights = [0.05, 0.20, 0.20, 0.25, 0.30]

    for step in range(0, total_steps, 10):
        for house in range(num_houses):
            if rng.rand() < event_prob:
                etype    = rng.choice(event_types, p=event_weights)
                params   = {}
                if etype == EVENT_FIRE:
                    duration = rng.randint(6, 12)
                elif etype == EVENT_BBQ:
                    duration = rng.randint(2, 6)
                elif etype == EVENT_SMOKING:
                    duration = rng.randint(1, 3)
                elif etype == EVENT_FDI_TEMP:
                    duration = rng.randint(2, 8)
                    params['fake_val'] = rng.uniform(80, 120)
                elif etype == EVENT_FDI_PM:
                    duration = rng.randint(2, 8)
                    params['fake_val'] = rng.uniform(150, 400)
                else:
                    duration = 1
                event_manager.add_event(step, house, etype, duration, params)
    return event_manager

def load_scenario_event_manager(scenario_file):
    """build EventManager from scenario JSON file"""
    print(f"Loading scenario from {scenario_file} ...")
    scenario      = load_scenario(scenario_file)
    event_manager = EventManager.from_scenario(scenario)
    n = len(event_manager.events)
    name = (scenario.get('scenario_name', scenario_file)
            if isinstance(scenario, dict) else scenario_file)
    print(f"  Loaded scenario '{name}' with {n} event(s).")
    return event_manager

def make_env(house_id, baseline_data, event_manager, max_steps):
    def _init():
        env = IoTDefenderEnv(
            baseline_data=baseline_data,
            house_id=house_id,
            event_manager=event_manager,
            max_steps=max_steps,
        )
        return Monitor(env)
    return _init
    
def create_vectorized_env(baseline_data, event_manager, max_steps, num_envs=NUM_HOUSES):
    env_fns = [make_env(i, baseline_data, event_manager, max_steps)
               for i in range(num_envs)]
    if num_envs > 1:
        try:
            return SubprocVecEnv(env_fns)
        except Exception:
            return DummyVecEnv(env_fns)
    return DummyVecEnv(env_fns)

def get_callbacks():
    return [CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix=CHECKPOINT_PREFIX,
    )]

def train(total_timesteps=500000, model_save_path=MODEL_SAVE_PATH,
          scenario_file=None):
    print("Loading baseline data ...")
    baseline            = load_baseline_data(DATA_FILE)
    total_baseline_steps = baseline.shape[0]
    print(f"Baseline shape: {baseline.shape}")

    if scenario_file:
        event_manager = load_scenario_event_manager(scenario_file)
    else:
        print("Generating random event schedule (seed=42) ...")
        event_manager = create_random_event_manager(total_baseline_steps)

    max_episode_steps = min(STEPS_PER_EPISODE, total_baseline_steps)

    print(f"Creating vectorised environment ({NUM_HOUSES} parallel houses) ...")
    env = create_vectorized_env(baseline, event_manager, max_episode_steps)

    print("Initialising PPO agent ...")
    model = PPO(
        "MlpPolicy", env,
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
        seed=42,
    )
    os.makedirs("./checkpoints",              exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    print(f"Training for {total_timesteps:,} timesteps ...")
    model.learn(total_timesteps=total_timesteps,
                callback=get_callbacks(), progress_bar=True)
    model.save(model_save_path)
    print(f"Model saved → {model_save_path}.zip")
    env.close()
    print("Training complete.")

def evaluate(model_path, num_episodes=10, scenario_file=None):
    """test trained model and print per-episode and summary statistics."""
    baseline = load_baseline_data(DATA_FILE)

    if scenario_file:
        event_manager = load_scenario_event_manager(scenario_file)
        print("Evaluating with scenario events.")
    else:
        event_manager = EventManager()   # no events → clean baseline
        print("Evaluating on clean baseline (no events).")

    model = PPO.load(model_path)
    total_rewards = []

    for episode in range(num_episodes):
        env = IoTDefenderEnv(baseline, house_id=0,
                              event_manager=event_manager, max_steps=96 * 2)
        obs, _ = env.reset()
        ep_reward = 0.0
        done      = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        print(f"Episode {episode + 1:2d}: Reward = {ep_reward:.2f}")

    print(f"\nAverage Reward: {np.mean(total_rewards):.2f} "
          f"+/- {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate the IoT DRL defence agent.")
    parser.add_argument("--timesteps",     type=int,   default=500000, help="Total training timesteps (default: 500000)")
    parser.add_argument("--save_model",    type=str,   default=MODEL_SAVE_PATH, help="Path to save final model")
    parser.add_argument("--evaluate",      type=str,   default=None, help="Evaluate a saved model (provide .zip path)")
    parser.add_argument("--episodes",      type=int,   default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--scenario_file", type=str,   default=None, help="Path to scenario JSON file for structured event injection. " "If omitted, random injection is used.")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.episodes, scenario_file=args.scenario_file)
    else:
        train(total_timesteps=args.timesteps,
              model_save_path=args.save_model,
              scenario_file=args.scenario_file)
