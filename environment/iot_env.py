import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json

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

ACTION_DO_NOTHING = 0
ACTION_ISOLATE_TEMP = 1
ACTION_ISOLATE_PM = 2
ACTION_VENTILATE = 3
ACTION_ALERT_FIRE = 4
NUM_ACTIONS = 5

EVENT_NORMAL = 0
EVENT_BBQ = 1
EVENT_SMOKING = 2
EVENT_FIRE = 3
EVENT_FDI_TEMP = 4
EVENT_FDI_PM = 5

EVENT_TYPE_TO_CODE = {
    "Normal":   EVENT_NORMAL,
    "BBQ":      EVENT_BBQ,
    "Smoking":  EVENT_SMOKING,
    "Fire":     EVENT_FIRE,
    "FDI_temp": EVENT_FDI_TEMP,
    "FDI_pm":   EVENT_FDI_PM,
}
EVENT_CODE_TO_TYPE = {v: k for k, v in EVENT_TYPE_TO_CODE.items()}

def load_scenario(filepath):
    """load scenario dict or bare events list from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

class EventManager:
    def __init__(self, events=None):
        #internal list of event dicts in JSON schema format
        self.events = list(events) if events else []

    @classmethod
    def load_from_file(cls, filepath):
        scenario = load_scenario(filepath)
        return cls.from_scenario(scenario)

    @classmethod
    def from_scenario(cls, scenario):
        if isinstance(scenario, dict) and 'events' in scenario:
            return cls(events=scenario['events'])
        elif isinstance(scenario, list):
            return cls(events=scenario)
        return cls()

    def set_scenario(self, scenario):
        if isinstance(scenario, dict) and 'events' in scenario:
            self.events = list(scenario['events'])
        elif isinstance(scenario, list):
            self.events = list(scenario)
        else:
            self.events = []

    def add_event_dict(self, event_dict):
        self.events.append(event_dict)

    def add_event(self, step, house, event_type, duration=1, params=None):
        if isinstance(event_type, int):
            type_str = EVENT_CODE_TO_TYPE.get(event_type, "Normal")
        else:
            type_str = str(event_type)

        self.events.append({
            "event_id":     f"auto_{len(self.events)}",
            "event_type":   type_str,
            "target_house": house,
            "start_time":   step,
            "duration":     duration,
            "severity":     1.0,
            "parameters":   params or {}
        })

    def get_active_events(self, step, house=None):
        active = []
        for ev in self.events:
            start = ev['start_time']
            if start <= step < start + ev['duration']:
                if house is None or ev['target_house'] == house:
                    active.append(ev)
        return active

    def get_event_type(self, step, house):
        active = self.get_active_events(step, house)
        if active:
            return EVENT_TYPE_TO_CODE.get(active[0]['event_type'], EVENT_NORMAL)
        return EVENT_NORMAL

    def apply_overrides(self, baseline_data, step, house_id=None):
        observed = baseline_data[step].copy()
        for ev in self.get_active_events(step):
            h = ev['target_house']
            if house_id is not None and h != house_id:
                continue
            self._apply_event_effect(observed, ev, step)
        return observed

    def _apply_event_effect(self, observed, ev, step):
        """apply 1 event sensor effect to observed array"""
        h         = ev['target_house']
        etype     = ev['event_type']
        severity  = float(ev.get('severity', 1.0))
        params    = ev.get('parameters', {})
        step_in   = step - ev['start_time']   # how far into the event we are
        si        = SENSOR_INDICES

        if etype == 'Fire':
            #temperature rises gradually then plateaus PM spikes immediately
            temp_rise = min(severity * 25.0, step_in * 8.0 + severity * 3.0)
            observed[h, si['out_temp']] += temp_rise
            observed[h, si['in_temp']] += temp_rise * 0.8
            observed[h, si['out_hum']] = max(20.0, observed[h, si['out_hum']] - severity * 20.0)
            observed[h, si['in_hum']] = max(20.0, observed[h, si['in_hum']]  - severity * 20.0)
            pm = severity * 500.0
            observed[h, si['out_pm']] = pm
            observed[h, si['in_pm']] = pm

        elif etype == 'BBQ':
            observed[h, si['out_temp']] += severity * 3.0
            observed[h, si['out_hum']] = max(20.0, observed[h, si['out_hum']] - severity * 5.0)
            observed[h, si['out_pm']] += severity * 150.0

        elif etype == 'Smoking':
            observed[h, si['in_pm']] += severity * 80.0

        elif etype == 'FDI_temp':
            #use explicit fake_val if provided otherwise derive from severity
            fake_val = params.get('fake_val', 25.0 + severity * 70.0)
            observed[h, si['out_temp']] = float(fake_val)

        elif etype == 'FDI_pm':
            fake_val = params.get('fake_val', 50.0 + severity * 350.0)
            observed[h, si['out_pm']] = float(fake_val)

class IoTDefenderEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, baseline_data, house_id, event_manager=None, history_len=8, max_steps=None):
        super().__init__()

        self.baseline     = baseline_data   # shape: steps, houses, sensors
        self.house_id     = house_id
        self.event_manager = event_manager if event_manager else EventManager()
        self.history_len  = history_len
        self.total_steps  = baseline_data.shape[0]
        self.max_steps    = max_steps if max_steps else self.total_steps

        #flattened local history + neighbour averages
        obs_dim = self.history_len * NUM_SENSORS + NUM_SENSORS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        #internal state
        self.current_step    = 0
        self.sensor_isolated = {'temp': False, 'pm': False}

        #counters for reward tracking
        self.cumulative_reward  = 0.0
        self.false_alarms       = 0
        self.missed_fires       = 0
        self.correct_isolations = 0

    def set_scenario(self, scenario):
        """replace event schedule from scenario dict or list"""
        self.event_manager.set_scenario(scenario)

    def _get_observed_data(self):
        return self.event_manager.apply_overrides(self.baseline, self.current_step)

    def _get_neighbor_average(self, observed):
        mask = np.ones(NUM_HOUSES, dtype=bool)
        mask[self.house_id] = False
        if np.any(mask):
            return observed[mask].mean(axis=0)
        return observed[self.house_id]

    def _build_observation(self):
        observed = self._get_observed_data()

        #replace isolated sensor readings with neighbour consensus
        if self.sensor_isolated['temp']:
            nb = self._get_neighbor_average(observed)
            observed[self.house_id, SENSOR_INDICES['out_temp']] = nb[SENSOR_INDICES['out_temp']]
            observed[self.house_id, SENSOR_INDICES['in_temp']]  = nb[SENSOR_INDICES['in_temp']]
        if self.sensor_isolated['pm']:
            nb = self._get_neighbor_average(observed)
            observed[self.house_id, SENSOR_INDICES['out_pm']] = nb[SENSOR_INDICES['out_pm']]
            observed[self.house_id, SENSOR_INDICES['in_pm']]  = nb[SENSOR_INDICES['in_pm']]

        #local sensor history (padded at episode start)
        start   = max(0, self.current_step - self.history_len)
        history = self.baseline[start:self.current_step, self.house_id, :]
        if history.shape[0] < self.history_len:
            pad     = np.zeros((self.history_len - history.shape[0], NUM_SENSORS))
            history = np.vstack([pad, history])
        history_flat = history.flatten()

        neighbor_avg = self._get_neighbor_average(observed)
        return np.concatenate([history_flat, neighbor_avg]).astype(np.float32)

    def _get_current_event(self):
        return self.event_manager.get_event_type(self.current_step, self.house_id)

    def _calculate_reward(self, action):
        event  = self._get_current_event()
        reward = 0.0

        #penalty for non-trivial action encourage efficiency
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

        elif event in (EVENT_BBQ, EVENT_SMOKING):
            if action == ACTION_ALERT_FIRE:
                reward -= 12.0
                self.false_alarms += 1
            elif action == ACTION_VENTILATE:
                reward += 5.0
            elif action == ACTION_DO_NOTHING:
                reward += 1.0

        else:  # EVENT_NORMAL
            if action == ACTION_DO_NOTHING:
                reward += 2.0
            elif action in (ACTION_ALERT_FIRE, ACTION_ISOLATE_TEMP, ACTION_ISOLATE_PM):
                reward -= 8.0
                if action == ACTION_ALERT_FIRE:
                    self.false_alarms += 1
            elif action == ACTION_VENTILATE:
                reward -= 1.0

        #promote energy efficiency
        observed     = self._get_observed_data()
        power_usage  = observed[self.house_id, SENSOR_INDICES['power']]
        reward      -= power_usage * 0.005
        return reward

    def _apply_action(self, action):
        if action == ACTION_ISOLATE_TEMP:
            self.sensor_isolated['temp'] = True
        elif action == ACTION_ISOLATE_PM:
            self.sensor_isolated['pm'] = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step       = 0
        self.sensor_isolated    = {'temp': False, 'pm': False}
        self.cumulative_reward  = 0.0
        self.false_alarms       = 0
        self.missed_fires       = 0
        self.correct_isolations = 0
        return self._build_observation(), {}

    def step(self, action):
        self._apply_action(action)

        reward                 = self._calculate_reward(action)
        self.cumulative_reward += reward
        self.current_step      += 1
        terminated             = self.current_step >= self.max_steps - 1
        truncated              = False

        obs = (self._build_observation() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            'event':            self._get_current_event() if not terminated else EVENT_NORMAL,
            'cumulative_reward': self.cumulative_reward,
            'false_alarms':     self.false_alarms,
            'missed_fires':     self.missed_fires,
            'correct_isolations': self.correct_isolations
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

#load the 30-day baseline data from the text file
def load_baseline_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    #count data lines to determine total timesteps
    data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
    total_steps = len(data_lines) // NUM_HOUSES

    data        = np.zeros((total_steps, NUM_HOUSES, NUM_SENSORS))
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

        house_id     = int(parts[0]) - 1
        time_str     = parts[1]
        h, m         = map(int, time_str.split(':'))
        step_of_day  = (h * 60 + m) // 15
        global_step  = current_day * 96 + step_of_day

        out_temp, out_hum, out_pm = float(parts[2]), float(parts[3]), float(parts[4])
        in_temp,  in_hum,  in_pm  = float(parts[5]), float(parts[6]), float(parts[7])
        power                      = float(parts[8])

        data[global_step, house_id, :] = [
            out_temp, out_hum, out_pm,
            in_temp,  in_hum,  in_pm, power
        ]
    return data
    
#quick smoke-test
if __name__ == "__main__":
    data_file = "environment_data_30days.txt"
    if os.path.exists(data_file):
        baseline = load_baseline_data(data_file)
        print(f"Loaded data shape: {baseline.shape}")

        env  = IoTDefenderEnv(baseline, house_id=0)
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Event: {info['event']}")
        print("Test complete.")
    else:
        print("Data file not found. Run data_generator.py first.")
