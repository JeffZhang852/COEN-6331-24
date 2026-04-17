# IoT Sensor Network Simulation Dashboard
#
# Place this file in the project root as simulation.py
# Loads data from environment/environment_data_30days.txt
# Loads trained model from models/iot_ddrl_model.zip
#
# Controls:
#   Right Arrow / n : Next timestep
#   Left Arrow / p  : Previous timestep
#   Space           : Play / Pause
#   F1              : Inject FIRE on selected house
#   F2              : Inject BBQ
#   F3              : Inject Smoking
#   F4              : Inject FDI attack (temp = 95°C)
#   1 to 9, 0       : Select house (0 = house 10)
#   r               : Reset to start
#   q               : Quit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

# Disable default matplotlib hotkeys to avoid conflicts
mpl.rcParams['keymap.save'] = ''
mpl.rcParams['keymap.fullscreen'] = ''
mpl.rcParams['keymap.quit'] = ''
mpl.rcParams['keymap.grid'] = ''
mpl.rcParams['keymap.home'] = ''
mpl.rcParams['keymap.back'] = ''
mpl.rcParams['keymap.forward'] = ''
mpl.rcParams['keymap.pan'] = ''
mpl.rcParams['keymap.zoom'] = ''
mpl.rcParams['keymap.xscale'] = ''
mpl.rcParams['keymap.yscale'] = ''

# 1) CONFIGURATION
DATA_FILE = "environment/environment_data_30days.txt"
MODEL_PATH = "models/iot_ddrl_model.zip"

NUM_HOUSES = 10
SENSOR_NAMES = ['Out Temp\n(°C)', 'Out Hum\n(%)', 'Out PM\n(µg/m³)',
                'In Temp\n(°C)', 'In Hum\n(%)', 'In PM\n(µg/m³)', 'Power\n(kW)']
NUM_SENSORS = len(SENSOR_NAMES)
TIMESTEPS_PER_DAY = 96
TOTAL_DAYS = 30
TOTAL_STEPS = TOTAL_DAYS * TIMESTEPS_PER_DAY

# DRL observation settings, must match training
HISTORY_LEN = 8

# Normalization ranges for the heatmap colours
SENSOR_RANGES = {
    0: (10, 35),
    1: (30, 100),
    2: (0, 200),
    3: (18, 28),
    4: (30, 80),
    5: (0, 200),
    6: (0, 3.0)
}

# What each action means in the AI decision panel
ACTION_MAP = {
    0: ("NORMAL", 0.99),
    1: ("ISOLATE TEMP", 0.9),
    2: ("ISOLATE PM", 0.9),
    3: ("VENTILATE", 0.8),
    4: ("FIRE ALERT", 0.95)
}


# 2) LOAD TRAINED DRL MODEL
def load_drl_model(model_path):
    if os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Loaded trained model from {model_path}")
            return model
        except ImportError:
            print("Stable Baselines3 not installed. Using dummy AI.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}. Using dummy AI.")
            return None
    else:
        print(f"Model not found at {model_path}. Using dummy AI.")
        return None


# 3) LOAD BASELINE DATA FROM FILE
def load_baseline_data(filepath):
    print(f"Loading data from {filepath}...")
    data = np.zeros((TOTAL_STEPS, NUM_HOUSES, NUM_SENSORS))

    with open(filepath, 'r') as f:
        lines = f.readlines()

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
        global_step = current_day * TIMESTEPS_PER_DAY + step_of_day

        out_temp = float(parts[2])
        out_hum = float(parts[3])
        out_pm = float(parts[4])
        in_temp = float(parts[5])
        in_hum = float(parts[6])
        in_pm = float(parts[7])
        power = float(parts[8])

        data[global_step, house_id, :] = [out_temp, out_hum, out_pm,
                                          in_temp, in_hum, in_pm, power]
    print(f"Loaded {TOTAL_STEPS} timesteps.")
    return data


# 4) EVENT MANAGER FOR INJECTING ACCIDENTS AND ATTACKS
class EventManager:
    def __init__(self):
        self.overrides = {}
        self.active_events = []

    def add_override(self, step, house, sensor, value):
        if step not in self.overrides:
            self.overrides[step] = {}
        if house not in self.overrides[step]:
            self.overrides[step][house] = {}
        self.overrides[step][house][sensor] = value

    def add_event(self, start_step, duration_steps, house, event_type, params):
        end_step = min(start_step + duration_steps, TOTAL_STEPS)
        self.active_events.append((start_step, end_step, house, event_type, params))

    def get_observed(self, baseline, step):
        observed = baseline[step].copy()

        if step in self.overrides:
            for house, sensors in self.overrides[step].items():
                for sensor, val in sensors.items():
                    observed[house, sensor] = val

        for start, end, house, etype, params in self.active_events:
            if start <= step < end:
                if etype == 'fire':
                    rel_step = step - start
                    temp_rise = min(25, rel_step * 8)
                    observed[house, 0] += temp_rise
                    observed[house, 3] += temp_rise * 0.8
                    observed[house, 1] = max(20, observed[house, 1] - 15)
                    observed[house, 4] = max(20, observed[house, 4] - 15)
                    pm_spike = 500 if rel_step < 4 else 0
                    observed[house, 2] = pm_spike
                    observed[house, 5] = pm_spike
                elif etype == 'bbq':
                    observed[house, 0] += 3
                    observed[house, 1] = max(20, observed[house, 1] - 5)
                    observed[house, 2] += 150
                elif etype == 'smoking':
                    observed[house, 5] += 80
                elif etype == 'fdi_temp':
                    observed[house, 0] = params.get('fake_temp', 95.0)
        return observed


# 5) BUILD OBSERVATION FOR THE AI
def build_observation(baseline, current_step, house_id, observed):
    start = max(0, current_step - HISTORY_LEN)
    history = baseline[start:current_step, house_id, :]
    if history.shape[0] < HISTORY_LEN:
        pad = np.zeros((HISTORY_LEN - history.shape[0], NUM_SENSORS))
        history = np.vstack([pad, history])
    history_flat = history.flatten()

    mask = np.ones(NUM_HOUSES, dtype=bool)
    mask[house_id] = False
    if np.any(mask):
        neighbor_avg = observed[mask].mean(axis=0)
    else:
        neighbor_avg = observed[house_id]

    return np.concatenate([history_flat, neighbor_avg]).astype(np.float32)


def drl_ai_inference(model, baseline, current_step, house_id, observed):
    obs = build_observation(baseline, current_step, house_id, observed)
    action, _ = model.predict(obs, deterministic=True)
    return ACTION_MAP.get(action, ("UNKNOWN", 0.0))


def dummy_ai_inference(house_id, observed_row, neighbors_avg):
    out_temp = observed_row[0]
    out_pm = observed_row[2]
    in_pm = observed_row[5]

    if out_temp > 50 and neighbors_avg[0] < 35:
        return "⚠️ FDI DETECTED", 0.95
    elif out_temp > 40 and (out_pm > 300 or in_pm > 300):
        return "🔥 FIRE ALERT", 0.90
    elif (out_pm > 100 or in_pm > 100) and out_temp < 35:
        return "🍖 SMOKE EVENT", 0.70
    else:
        return "✅ NORMAL", 0.99


# 6) MAIN DASHBOARD CLASS
class SensorDashboard:
    def __init__(self, baseline_data, drl_model=None):
        self.baseline = baseline_data
        self.drl_model = drl_model
        self.event_manager = EventManager()
        self.current_step = 0
        self.paused = True
        self.selected_house = 0

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('IoT Sensor Network Simulation')

        gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 1, 0.5], width_ratios=[3, 1])

        self.ax_heatmap = self.fig.add_subplot(gs[0, :])
        self.ax_ai = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_controls = self.fig.add_subplot(gs[2, :])

        self.heatmap_img = None
        self.value_texts = []
        self._init_heatmap()
        self._init_info_panel()
        self._init_controls()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.tight_layout()

    def _init_heatmap(self):
        observed = self.event_manager.get_observed(self.baseline, self.current_step)
        norm_data = np.zeros((NUM_HOUSES, NUM_SENSORS))
        for s in range(NUM_SENSORS):
            vmin, vmax = SENSOR_RANGES[s]
            norm_data[:, s] = np.clip((observed[:, s] - vmin) / (vmax - vmin), 0, 1)

        self.heatmap_img = self.ax_heatmap.imshow(norm_data.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        self.ax_heatmap.set_xticks(range(NUM_HOUSES))
        self.ax_heatmap.set_xticklabels([f'House {i+1}' for i in range(NUM_HOUSES)])
        self.ax_heatmap.set_yticks(range(NUM_SENSORS))
        self.ax_heatmap.set_yticklabels(SENSOR_NAMES)
        self.ax_heatmap.set_title(f"Step {self.current_step} / {TOTAL_STEPS}  "
                                  f"Day {self.current_step//96 + 1}  {self._step_to_time_str(self.current_step)}")

        for h in range(NUM_HOUSES):
            for s in range(NUM_SENSORS):
                val = observed[h, s]
                color = 'black' if 0.2 < norm_data[h, s] < 0.8 else 'white'
                text = self.ax_heatmap.text(h, s, f'{val:.1f}', ha='center', va='center',
                                            color=color, fontsize=8)
                self.value_texts.append(text)

    def _init_info_panel(self):
        self.ax_info.axis('off')
        info_text = (
            "Controls:\n"
            "→ / n : Next step\n"
            "← / p : Prev step\n"
            "Space : Play / Pause\n"
            "F1 : Inject FIRE on selected house\n"
            "F2 : Inject BBQ\n"
            "F3 : Inject Smoking\n"
            "F4 : Inject FDI (temp=95°C)\n"
            "1 to 9, 0 : Select house (0=10)\n"
            "r : Reset\n"
            "q : Quit\n\n"
            f"Selected House: {self.selected_house+1}"
        )
        self.info_text_obj = self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                                               fontsize=10, verticalalignment='top', family='monospace')

    def _init_controls(self):
        self.ax_controls.axis('off')

    def _step_to_time_str(self, step):
        day = step // 96 + 1
        step_of_day = step % 96
        hours = step_of_day * 15 // 60
        minutes = (step_of_day * 15) % 60
        return f"{hours:02d}:{minutes:02d}"

    def update_display(self):
        observed = self.event_manager.get_observed(self.baseline, self.current_step)

        norm_data = np.zeros((NUM_HOUSES, NUM_SENSORS))
        for s in range(NUM_SENSORS):
            vmin, vmax = SENSOR_RANGES[s]
            norm_data[:, s] = np.clip((observed[:, s] - vmin) / (vmax - vmin), 0, 1)
        self.heatmap_img.set_data(norm_data.T)

        idx = 0
        for h in range(NUM_HOUSES):
            for s in range(NUM_SENSORS):
                val = observed[h, s]
                self.value_texts[idx].set_text(f'{val:.1f}')
                if 0.2 < norm_data[h, s] < 0.8:
                    self.value_texts[idx].set_color('black')
                else:
                    self.value_texts[idx].set_color('white')
                idx += 1

        self.ax_heatmap.set_title(f"Step {self.current_step} / {TOTAL_STEPS}  "
                                  f"Day {self.current_step//96 + 1}  {self._step_to_time_str(self.current_step)}")

        self._update_ai_panel(observed)

        self.info_text_obj.set_text(
            "Controls:\n"
            "→ / n : Next step\n"
            "← / p : Prev step\n"
            "Space : Play / Pause\n"
            "F1 : Inject FIRE on selected house\n"
            "F2 : Inject BBQ\n"
            "F3 : Inject Smoking\n"
            "F4 : Inject FDI (temp=95°C)\n"
            "1 to 9, 0 : Select house (0=10)\n"
            "r : Reset\n"
            "q : Quit\n\n"
            f"Selected House: {self.selected_house+1}"
        )

        self.fig.canvas.draw_idle()

    def _update_ai_panel(self, observed):
        self.ax_ai.clear()
        self.ax_ai.axis('off')

        decisions = []
        for h in range(NUM_HOUSES):
            mask = np.ones(NUM_HOUSES, dtype=bool)
            mask[h] = False
            neighbors_avg = observed[mask].mean(axis=0) if np.any(mask) else observed[h]

            if self.drl_model is not None:
                try:
                    decision, conf = drl_ai_inference(self.drl_model, self.baseline, self.current_step, h, observed)
                except:
                    decision, conf = dummy_ai_inference(h, observed[h], neighbors_avg)
            else:
                decision, conf = dummy_ai_inference(h, observed[h], neighbors_avg)
            decisions.append((decision, conf))

        y_pos = 0.95
        self.ax_ai.text(0.05, y_pos, "House  AI Decision", transform=self.ax_ai.transAxes,
                        fontweight='bold', fontsize=10)
        y_pos -= 0.06
        for h, (dec, conf) in enumerate(decisions):
            if "NORMAL" in dec:
                color = 'green'
            elif "FIRE" in dec:
                color = 'red'
            elif "SMOKE" in dec or "VENTILATE" in dec:
                color = 'orange'
            elif "ISOLATE" in dec or "FDI" in dec:
                color = 'purple'
            else:
                color = 'black'
            text = f"House {h+1:2d}: {dec} ({conf:.0%})"
            self.ax_ai.text(0.05, y_pos, text, transform=self.ax_ai.transAxes,
                            fontsize=9, color=color, family='monospace')
            y_pos -= 0.055

    def on_key_press(self, event):
        if event.key in ['right', 'n']:
            self.current_step = min(self.current_step + 1, TOTAL_STEPS - 1)
            self.update_display()
        elif event.key in ['left', 'p']:
            self.current_step = max(self.current_step - 1, 0)
            self.update_display()
        elif event.key == ' ':
            self.paused = not self.paused
            if not self.paused:
                self.auto_play()
        elif event.key == 'r':
            self.current_step = 0
            self.event_manager = EventManager()
            self.update_display()
        elif event.key == 'f1':
            self.inject_fire()
        elif event.key == 'f2':
            self.inject_bbq()
        elif event.key == 'f3':
            self.inject_smoking()
        elif event.key == 'f4':
            self.inject_fdi()
        elif event.key in [str(i) for i in range(10)]:
            if event.key == '0':
                self.selected_house = 9
            else:
                self.selected_house = int(event.key) - 1
            self.update_display()
        elif event.key == 'q':
            plt.close()
            sys.exit(0)

    def inject_fire(self):
        duration = 12
        self.event_manager.add_event(self.current_step, duration, self.selected_house, 'fire', {})
        print(f"Injected FIRE at House {self.selected_house+1} starting now, duration {duration} steps.")
        self.update_display()

    def inject_bbq(self):
        duration = 4
        self.event_manager.add_event(self.current_step, duration, self.selected_house, 'bbq', {})
        print(f"Injected BBQ at House {self.selected_house+1} for {duration} steps.")
        self.update_display()

    def inject_smoking(self):
        duration = 2
        self.event_manager.add_event(self.current_step, duration, self.selected_house, 'smoking', {})
        print(f"Injected Smoking at House {self.selected_house+1} for {duration} steps.")
        self.update_display()

    def inject_fdi(self):
        self.event_manager.add_override(self.current_step, self.selected_house, 0, 95.0)
        self.event_manager.add_event(self.current_step, 1, self.selected_house, 'fdi_temp', {'fake_temp': 95.0})
        print(f"Injected FDI (temp=95°C) at House {self.selected_house+1} for current step.")
        self.update_display()

    def auto_play(self):
        if self.paused or self.current_step >= TOTAL_STEPS - 1:
            return
        self.current_step += 1
        self.update_display()
        self.fig.canvas.draw()
        self.fig.canvas.manager.window.after(300, self.auto_play)

    def run(self):
        plt.show()


# 7) RUN THE DASHBOARD
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Please run environment/data_generator.py first.")
        sys.exit(1)

    baseline = load_baseline_data(DATA_FILE)
    drl_model = load_drl_model(MODEL_PATH)
    dashboard = SensorDashboard(baseline, drl_model)
    dashboard.run()