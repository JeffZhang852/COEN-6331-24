import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import uuid
import argparse

from environment.iot_env import (
    EventManager,
    load_baseline_data,
    load_scenario,
    EVENT_FIRE, EVENT_BBQ, EVENT_SMOKING, EVENT_FDI_TEMP, EVENT_FDI_PM,
)

#disable matplotlib default hotkeys that clash with simulation controls
mpl.rcParams['keymap.save']     = ''
mpl.rcParams['keymap.fullscreen'] = ''
mpl.rcParams['keymap.quit']     = ''
mpl.rcParams['keymap.grid']     = ''
mpl.rcParams['keymap.home']     = ''
mpl.rcParams['keymap.back']     = ''
mpl.rcParams['keymap.forward']  = ''
mpl.rcParams['keymap.pan']      = ''
mpl.rcParams['keymap.zoom']     = ''
mpl.rcParams['keymap.xscale']   = ''
mpl.rcParams['keymap.yscale']   = ''

DATA_FILE    = "environment/environment_data_30days.txt"
MODEL_PATH   = "models/iot_ddrl_model.zip"
SCENARIO_FILE = "scenario.json"

NUM_HOUSES          = 10
SENSOR_NAMES        = ['Out Temp\n(°C)', 'Out Hum\n(%)', 'Out PM\n(µg/m³)',
                       'In Temp\n(°C)',  'In Hum\n(%)',  'In PM\n(µg/m³)', 'Power\n(kW)']
NUM_SENSORS         = len(SENSOR_NAMES)
TIMESTEPS_PER_DAY   = 96
TOTAL_DAYS          = 30
TOTAL_STEPS         = TOTAL_DAYS * TIMESTEPS_PER_DAY
HISTORY_LEN         = 8    # must match training

SENSOR_RANGES = {
    0: (10, 35),
    1: (30, 100),
    2: (0, 200),
    3: (18, 28),
    4: (30, 80),
    5: (0, 200),
    6: (0, 3.0),
}

ACTION_MAP = {
    0: ("NORMAL",       0.99),
    1: ("ISOLATE TEMP", 0.90),
    2: ("ISOLATE PM",   0.90),
    3: ("VENTILATE",    0.80),
    4: ("FIRE ALERT",   0.95),
}

def load_drl_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using rule-based fallback.")
        return None
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"Loaded trained model from {model_path}")
        return model
    except ImportError:
        print("stable-baselines3 not installed. Using rule-based fallback.")
        return None
    except Exception as exc:
        print(f"Error loading model: {exc}. Using rule-based fallback.")
        return None

def build_observation(baseline, current_step, house_id, observed):
    start   = max(0, current_step - HISTORY_LEN)
    history = baseline[start:current_step, house_id, :]
    if history.shape[0] < HISTORY_LEN:
        pad     = np.zeros((HISTORY_LEN - history.shape[0], NUM_SENSORS))
        history = np.vstack([pad, history])
    history_flat = history.flatten()

    mask = np.ones(NUM_HOUSES, dtype=bool)
    mask[house_id] = False
    neighbor_avg = observed[mask].mean(axis=0) if np.any(mask) else observed[house_id]
    return np.concatenate([history_flat, neighbor_avg]).astype(np.float32)

def drl_ai_inference(model, baseline, current_step, house_id, observed):
    obs    = build_observation(baseline, current_step, house_id, observed)
    action, _ = model.predict(obs, deterministic=True)
    return ACTION_MAP.get(int(action), ("UNKNOWN", 0.0))


def rule_based_inference(house_id, observed_row, neighbors_avg):
    """fallback heuristic when no trained model is available"""
    out_temp = observed_row[0]
    out_pm   = observed_row[2]
    in_pm    = observed_row[5]

    if out_temp > 50 and neighbors_avg[0] < 35:
        return "FDI DETECTED", 0.95
    elif out_temp > 40 and (out_pm > 300 or in_pm > 300):
        return "FIRE ALERT", 0.90
    elif out_pm > 100 or in_pm > 100:
        return "SMOKE EVENT", 0.70
    return "NORMAL", 0.99

class SensorDashboard:
    def __init__(self, baseline_data, drl_model=None,
                 scenario=None, debug_mode=False):
        self.baseline      = baseline_data
        self.drl_model     = drl_model
        self.debug_mode    = debug_mode
        self.current_step  = 0
        self.paused        = True
        self.selected_house = 0

        #build EventManager from scenario (or empty if none provided)
        self.scenario       = scenario
        self.event_manager  = (EventManager.from_scenario(scenario)
                                if scenario else EventManager())

        #build figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('IoT Sensor Network Simulation')

        gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 1, 0.5],
                                    width_ratios=[3, 1])
        self.ax_heatmap  = self.fig.add_subplot(gs[0, :])
        self.ax_ai       = self.fig.add_subplot(gs[1, 0])
        self.ax_info     = self.fig.add_subplot(gs[1, 1])
        self.ax_controls = self.fig.add_subplot(gs[2, :])

        self.heatmap_img  = None
        self.value_texts  = []
        self._init_heatmap()
        self._init_info_panel()
        self._init_controls()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.tight_layout()
                   
    def _init_heatmap(self):
        observed  = self.event_manager.apply_overrides(self.baseline, self.current_step)
        norm_data = self._normalise(observed)

        self.heatmap_img = self.ax_heatmap.imshow(
            norm_data.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        self.ax_heatmap.set_xticks(range(NUM_HOUSES))
        self.ax_heatmap.set_xticklabels([f'House {i+1}' for i in range(NUM_HOUSES)])
        self.ax_heatmap.set_yticks(range(NUM_SENSORS))
        self.ax_heatmap.set_yticklabels(SENSOR_NAMES)
        self._update_heatmap_title()

        for h in range(NUM_HOUSES):
            for s in range(NUM_SENSORS):
                val   = observed[h, s]
                color = 'black' if 0.2 < norm_data[h, s] < 0.8 else 'white'
                text  = self.ax_heatmap.text(h, s, f'{val:.1f}',
                                              ha='center', va='center',
                                              color=color, fontsize=8)
                self.value_texts.append(text)

    def _init_info_panel(self):
        self.ax_info.axis('off')
        self.info_text_obj = self.ax_info.text(
            0.05, 0.95, self._controls_text(),
            transform=self.ax_info.transAxes,
            fontsize=9, verticalalignment='top', family='monospace')

    def _init_controls(self):
        self.ax_controls.axis('off')

    def _controls_text(self):
        lines = [
            "Controls:",
            "→ / n : Next step",
            "← / p : Prev step",
            "Space : Play / Pause",
            "1–9, 0 : Select house (0=10)",
            "r : Reset",
            "q : Quit",
        ]
        if self.debug_mode:
            lines += [
                "",
                "Debug (--debug):",
                "F1 : Inject FIRE",
                "F2 : Inject BBQ",
                "F3 : Inject Smoking",
                "F4 : Inject FDI (temp=95°C)",
            ]
        lines.append(f"\nSelected House: {self.selected_house + 1}")

        active = self.event_manager.get_active_events(self.current_step)
        if active:
            ev    = active[0]
            lines.append(f"\nActive: {ev['event_type']}")
            lines.append(f"  House {ev['target_house']}  sev={ev.get('severity', 1.0):.2f}")
        return "\n".join(lines)
      
    def _normalise(self, observed):
        norm = np.zeros((NUM_HOUSES, NUM_SENSORS))
        for s in range(NUM_SENSORS):
            vmin, vmax = SENSOR_RANGES[s]
            norm[:, s]  = np.clip((observed[:, s] - vmin) / (vmax - vmin), 0, 1)
        return norm

    def _update_heatmap_title(self):
        step = self.current_step
        self.ax_heatmap.set_title(
            f"Step {step} / {TOTAL_STEPS}   "
            f"Day {step // 96 + 1}   {self._step_to_time(step)}"
        )

    def _step_to_time(self, step):
        sod     = step % 96
        hours   = sod * 15 // 60
        minutes = (sod * 15) % 60
        return f"{hours:02d}:{minutes:02d}"

    def update_display(self):
        observed  = self.event_manager.apply_overrides(self.baseline, self.current_step)
        norm_data = self._normalise(observed)

        self.heatmap_img.set_data(norm_data.T)
        self._update_heatmap_title()

        idx = 0
        for h in range(NUM_HOUSES):
            for s in range(NUM_SENSORS):
                val = observed[h, s]
                self.value_texts[idx].set_text(f'{val:.1f}')
                self.value_texts[idx].set_color(
                    'black' if 0.2 < norm_data[h, s] < 0.8 else 'white')
                idx += 1

        self._update_ai_panel(observed)
        self.info_text_obj.set_text(self._controls_text())
        self.fig.canvas.draw_idle()

    def _update_ai_panel(self, observed):
        self.ax_ai.clear()
        self.ax_ai.axis('off')

        y_pos = 0.95
        self.ax_ai.text(0.05, y_pos, "House  AI Decision",
                         transform=self.ax_ai.transAxes,
                         fontweight='bold', fontsize=10)
        y_pos -= 0.06

        for h in range(NUM_HOUSES):
            mask  = np.ones(NUM_HOUSES, dtype=bool)
            mask[h] = False
            nb_avg = observed[mask].mean(axis=0) if np.any(mask) else observed[h]

            if self.drl_model is not None:
                try:
                    decision, conf = drl_ai_inference(
                        self.drl_model, self.baseline, self.current_step, h, observed)
                except Exception:
                    decision, conf = rule_based_inference(h, observed[h], nb_avg)
            else:
                decision, conf = rule_based_inference(h, observed[h], nb_avg)

            color = ("green"  if "NORMAL"   in decision else
                     "red"    if "FIRE"     in decision else
                     "orange" if "SMOKE" in decision or "VENTILATE" in decision else
                     "purple" if "ISOLATE"  in decision or "FDI" in decision else
                     "black")

            self.ax_ai.text(0.05, y_pos, f"House {h+1:2d}: {decision} ({conf:.0%})",
                             transform=self.ax_ai.transAxes,
                             fontsize=9, color=color, family='monospace')
            y_pos -= 0.055

    def on_key_press(self, event):
        k = event.key
        if k in ('right', 'n'):
            self.current_step = min(self.current_step + 1, TOTAL_STEPS - 1)
            self.update_display()
        elif k in ('left', 'p'):
            self.current_step = max(self.current_step - 1, 0)
            self.update_display()
        elif k == ' ':
            self.paused = not self.paused
            if not self.paused:
                self.auto_play()
        elif k == 'r':
            self.current_step   = 0
            self.event_manager  = (EventManager.from_scenario(self.scenario)
                                    if self.scenario else EventManager())
            self.update_display()
        elif k in [str(i) for i in range(10)]:
            self.selected_house = 9 if k == '0' else int(k) - 1
            self.update_display()
        elif k == 'q':
            plt.close()
            sys.exit(0)
        #debug-only event injection
        elif self.debug_mode:
            if k == 'f1':
                self._debug_inject('Fire',    12)
            elif k == 'f2':
                self._debug_inject('BBQ',      4)
            elif k == 'f3':
                self._debug_inject('Smoking',  2)
            elif k == 'f4':
                self._debug_inject_fdi()

    def _debug_inject(self, event_type, duration):
        ev = {
            "event_id":     f"dbg_{uuid.uuid4().hex[:6]}",
            "event_type":   event_type,
            "target_house": self.selected_house,
            "start_time":   self.current_step,
            "duration":     duration,
            "severity":     1.0,
            "parameters":   {},
        }
        self.event_manager.add_event_dict(ev)
        print(f"[DEBUG] Injected {event_type} at House {self.selected_house + 1} "
              f"for {duration} steps.")
        self.update_display()

    def _debug_inject_fdi(self):
        ev = {
            "event_id":     f"dbg_{uuid.uuid4().hex[:6]}",
            "event_type":   "FDI_temp",
            "target_house": self.selected_house,
            "start_time":   self.current_step,
            "duration":     1,
            "severity":     1.0,
            "parameters":   {"sensor": "outdoor_temp", "fake_val": 95.0},
        }
        self.event_manager.add_event_dict(ev)
        print(f"[DEBUG] Injected FDI_temp (95°C) at House {self.selected_house + 1}.")
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

def main():
    parser = argparse.ArgumentParser(description="IoT Sensor Simulation Dashboard")
    parser.add_argument("--scenario", default=SCENARIO_FILE, help=f"Path to scenario JSON file (default: {SCENARIO_FILE})")
    parser.add_argument("--model",    default=MODEL_PATH, help=f"Path to trained model (default: {MODEL_PATH})")
    parser.add_argument("--data",     default=DATA_FILE, help=f"Path to baseline data (default: {DATA_FILE})")
    parser.add_argument("--debug",    action="store_true", help="Enable keyboard event injection (F1–F4)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}")
        sys.exit(1)

    print(f"Loading baseline data from {args.data} ...")
    baseline = load_baseline_data(args.data)

    scenario = None
    if os.path.exists(args.scenario):
        print(f"Loading scenario from {args.scenario} ...")
        scenario = load_scenario(args.scenario)
        n = len(scenario.get('events', scenario) if isinstance(scenario, dict) else scenario)
        print(f"  → {n} events loaded.")
    else:
        if args.scenario != SCENARIO_FILE:
            print(f"Warning: scenario file not found: {args.scenario}")
        print("Running with no scenario events (flat baseline).")

    drl_model = load_drl_model(args.model)

    if args.debug:
        print("Debug mode enabled: F1–F4 inject events at the selected house.")

    dashboard = SensorDashboard(
        baseline_data=baseline,
        drl_model=drl_model,
        scenario=scenario,
        debug_mode=args.debug,
    )
    dashboard.run()


if __name__ == "__main__":
    main()
