# IoT Sensor Network Defense Environment
#
# Gymnasium environment for training DRL agents to defend against
# False Data Injection attacks and physical events.
#
# Each house is treated as an independent agent in a Multi Agent setup.
# This environment supports one house at a time. Training will instantiate
# one environment per house or use a vectorized wrapper.
#
# Observation is local sensor history plus neighbor summaries.
# Action can be 0 Do Nothing, 1 Isolate Temp Sensor, 2 Isolate PM Sensor,
# 3 Activate Ventilation, or 4 Trigger Fire Alert.
# Reward is based on correct identification of events and minimal false alarms.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

# Constants for the simulation
NUM_HOUSES = 10
NUM_SENSORS = 7
SENSOR_INDICES = {
    'out_temp': 0,
    'out_hum': 1,
    'out_pm': 2,
    'in_temp': 3,
    'in_hum': 4,
    'in_pm': 5,
    'power': 6
}

# Actions the agent can choose
ACTION_DO_NOTHING = 0
ACTION_ISOLATE_TEMP = 1
ACTION_ISOLATE_PM = 2
ACTION_VENTILATE = 3
ACTION_ALERT_FIRE = 4
NUM_ACTIONS = 5

# Event type codes used by the EventManager
EVENT_NORMAL = 0
EVENT_BBQ = 1
EVENT_SMOKING = 2
EVENT_FIRE = 3
EVENT_FDI_TEMP = 4
EVENT_FDI_PM = 5


# Event Manager handles injecting accidents and attacks into the data stream.
class EventManager:
    def __init__(self, event_schedule=None):
        # event_schedule maps step to a list of house, event_type, params
        self.schedule = event_schedule if event_schedule else {}
        self.active_overrides = {}

    def add_event(self, step, house, event_type, duration=1, params=None):
        # Add an event to the schedule
        if step not in self.schedule:
            self.schedule[step] = []
        self.schedule[step].append((house, event_type, duration, params or {}))

    def get_event_type(self, step, house):
        # Return the event type active at this step for the given house
        if step in self.schedule:
            for h, etype, dur, params in self.schedule[step]:
                if h == house:
                    return etype
        return EVENT_NORMAL

    def apply_overrides(self, baseline_data, step, house_id=None):
        # Return observed data with any active event overrides applied
        observed = baseline_data[step].copy()

        if step in self.schedule:
            for h, etype, dur, params in self.schedule[step]:
                if house_id is not None and h != house_id:
                    continue
                if etype == EVENT_FIRE:
                    # Fire causes temp to rise, humidity to drop, and heavy smoke
                    observed[h, SENSOR_INDICES['out_temp']] += 25
                    observed[h, SENSOR_INDICES['in_temp']] += 20
                    observed[h, SENSOR_INDICES['out_hum']] = max(20, observed[h, SENSOR_INDICES['out_hum']] - 20)
                    observed[h, SENSOR_INDICES['in_hum']] = max(20, observed[h, SENSOR_INDICES['in_hum']] - 20)
                    observed[h, SENSOR_INDICES['out_pm']] = 500
                    observed[h, SENSOR_INDICES['in_pm']] = 500
                elif etype == EVENT_BBQ:
                    # BBQ adds heat, drops humidity a bit, and produces outdoor smoke
                    observed[h, SENSOR_INDICES['out_temp']] += 3
                    observed[h, SENSOR_INDICES['out_hum']] = max(20, observed[h, SENSOR_INDICES['out_hum']] - 5)
                    observed[h, SENSOR_INDICES['out_pm']] += 150
                elif etype == EVENT_SMOKING:
                    # Smoking only raises indoor particle levels
                    observed[h, SENSOR_INDICES['in_pm']] += 80
                elif etype == EVENT_FDI_TEMP:
                    # FDI attack on outdoor temperature sensor
                    observed[h, SENSOR_INDICES['out_temp']] = params.get('fake_val', 95.0)
                elif etype == EVENT_FDI_PM:
                    # FDI attack on outdoor particle sensor
                    observed[h, SENSOR_INDICES['out_pm']] = params.get('fake_val', 200.0)
        return observed


# IoT Defense Environment class following the Gymnasium interface.
class IoTDefenderEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    def __init__(self, baseline_data, house_id, event_manager=None,
                 history_len=8, max_steps=None):
        super().__init__()

        self.baseline = baseline_data  # shape is steps, houses, sensors
        self.house_id = house_id
        self.event_manager = event_manager if event_manager else EventManager()
        self.history_len = history_len
        self.total_steps = baseline_data.shape[0]
        self.max_steps = max_steps if max_steps else self.total_steps

        # Observation includes flattened history and neighbor averages
        obs_dim = self.history_len * NUM_SENSORS + NUM_SENSORS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Five possible actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Internal state
        self.current_step = 0
        self.sensor_isolated = {'temp': False, 'pm': False}

        # Counters for reward tracking
        self.cumulative_reward = 0.0
        self.false_alarms = 0
        self.missed_fires = 0
        self.correct_isolations = 0

    def _get_observed_data(self):
        # Get the current observed data for all houses after applying event overrides
        return self.event_manager.apply_overrides(self.baseline, self.current_step)

    def _get_neighbor_average(self, observed):
        # Compute the average sensor readings of all other houses
        mask = np.ones(NUM_HOUSES, dtype=bool)
        mask[self.house_id] = False
        if np.any(mask):
            return observed[mask].mean(axis=0)
        else:
            return observed[self.house_id]

    def _build_observation(self):
        # Build the observation vector for the agent
        observed = self._get_observed_data()

        # If a sensor is isolated, replace its reading with the neighbor average
        if self.sensor_isolated['temp']:
            neighbor_avg = self._get_neighbor_average(observed)
            observed[self.house_id, SENSOR_INDICES['out_temp']] = neighbor_avg[SENSOR_INDICES['out_temp']]
            observed[self.house_id, SENSOR_INDICES['in_temp']] = neighbor_avg[SENSOR_INDICES['in_temp']]
        if self.sensor_isolated['pm']:
            neighbor_avg = self._get_neighbor_average(observed)
            observed[self.house_id, SENSOR_INDICES['out_pm']] = neighbor_avg[SENSOR_INDICES['out_pm']]
            observed[self.house_id, SENSOR_INDICES['in_pm']] = neighbor_avg[SENSOR_INDICES['in_pm']]

        # Collect local sensor history
        start = max(0, self.current_step - self.history_len)
        history = self.baseline[start:self.current_step, self.house_id, :]
        if history.shape[0] < self.history_len:
            pad = np.zeros((self.history_len - history.shape[0], NUM_SENSORS))
            history = np.vstack([pad, history])
        history_flat = history.flatten()

        # Neighbor averages
        neighbor_avg = self._get_neighbor_average(observed)

        return np.concatenate([history_flat, neighbor_avg]).astype(np.float32)

    def _get_current_event(self):
        # Return the event type active right now for this house
        return self.event_manager.get_event_type(self.current_step, self.house_id)

    def _calculate_reward(self, action):
        # Reward function to guide the agent's learning
        event = self._get_current_event()
        reward = 0.0

        # Small penalty for taking any action to encourage efficiency
        if action != ACTION_DO_NOTHING:
            reward -= 0.05

        if event == EVENT_FIRE:
            if action == ACTION_ALERT_FIRE:
                reward += 20.0
            else:
                reward -= 30.0
                self.missed_fires += 1

        elif event == EVENT_FDI_TEMP:
            if action == ACTION_ISOLATE_TEMP:
                reward += 10.0
                self.correct_isolations += 1
            elif action == ACTION_ALERT_FIRE:
                reward -= 15.0
                self.false_alarms += 1
            elif action == ACTION_DO_NOTHING:
                reward -= 5.0

        elif event == EVENT_FDI_PM:
            if action == ACTION_ISOLATE_PM:
                reward += 10.0
                self.correct_isolations += 1
            elif action == ACTION_ALERT_FIRE:
                reward -= 15.0
                self.false_alarms += 1
            elif action == ACTION_DO_NOTHING:
                reward -= 5.0

        elif event == EVENT_BBQ or event == EVENT_SMOKING:
            if action == ACTION_ALERT_FIRE:
                reward -= 12.0
                self.false_alarms += 1
            elif action == ACTION_VENTILATE:
                reward += 5.0
            elif action == ACTION_DO_NOTHING:
                reward += 1.0

        else:  # Normal situation
            if action == ACTION_DO_NOTHING:
                reward += 2.0
            elif action in [ACTION_ALERT_FIRE, ACTION_ISOLATE_TEMP, ACTION_ISOLATE_PM]:
                reward -= 8.0
                if action == ACTION_ALERT_FIRE:
                    self.false_alarms += 1
            elif action == ACTION_VENTILATE:
                reward -= 1.0

        # Tiny penalty for power usage to promote energy savings
        observed = self._get_observed_data()
        power_usage = observed[self.house_id, SENSOR_INDICES['power']]
        reward -= power_usage * 0.005

        return reward

    def _apply_action(self, action):
        # Update isolation state based on the chosen action
        if action == ACTION_ISOLATE_TEMP:
            self.sensor_isolated['temp'] = True
        elif action == ACTION_ISOLATE_PM:
            self.sensor_isolated['pm'] = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.sensor_isolated = {'temp': False, 'pm': False}
        self.cumulative_reward = 0.0
        self.false_alarms = 0
        self.missed_fires = 0
        self.correct_isolations = 0
        return self._build_observation(), {}

    def step(self, action):
        # Apply the chosen action
        self._apply_action(action)

        # Compute reward
        reward = self._calculate_reward(action)
        self.cumulative_reward += reward

        # Move to the next time step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 1
        truncated = False

        # Build the next observation
        obs = self._build_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'event': self._get_current_event() if not terminated else EVENT_NORMAL,
            'cumulative_reward': self.cumulative_reward,
            'false_alarms': self.false_alarms,
            'missed_fires': self.missed_fires,
            'correct_isolations': self.correct_isolations
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        # Visualization is handled by the separate dashboard
        pass


# Helper function to load the 30 day baseline data from the text file
def load_baseline_data(filepath):
    # Make sure the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Count total steps by reading the file once
    with open(filepath, 'r') as f:
        lines = f.readlines()

    steps = 0
    for line in lines:
        if not line.startswith('#') and line.strip():
            steps += 1
    total_steps = steps // NUM_HOUSES

    data = np.zeros((total_steps, NUM_HOUSES, NUM_SENSORS))

    current_day = -1
    for line in lines:
        line = line.strip()
        if line.startswith('#Day'):
            current_day = int(line.split()[1]) - 1
            continue
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 9:
            continue

        house_id = int(parts[0]) - 1
        time_str = parts[1]
        h, m = map(int, time_str.split(':'))
        step_of_day = (h * 60 + m) // 15
        global_step = current_day * 96 + step_of_day

        out_temp = float(parts[2])
        out_hum = float(parts[3])
        out_pm = float(parts[4])
        in_temp = float(parts[5])
        in_hum = float(parts[6])
        in_pm = float(parts[7])
        power = float(parts[8])

        data[global_step, house_id, :] = [out_temp, out_hum, out_pm,
                                          in_temp, in_hum, in_pm, power]
    return data

# Quick test to verify the environment works
if __name__ == "__main__":
    data_file = "environment_data_30days.txt"
    if os.path.exists(data_file):
        baseline = load_baseline_data(data_file)
        print(f"Loaded data shape: {baseline.shape}")

        env = IoTDefenderEnv(baseline, house_id=0)
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Event: {info['event']}")
        print("Test complete.")
    else:
        print("Data file not found. Run data_generator.py first.")