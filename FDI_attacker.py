import uuid
import numpy as np
import json
from environment.iot_env import SENSOR_INDICES

EVENT_TYPE_FDI_TEMP = "FDI_temp"
EVENT_TYPE_FDI_PM   = "FDI_pm"

SENSOR_OUTDOOR_TEMP = "outdoor_temp"
SENSOR_INDOOR_TEMP  = "indoor_temp"
SENSOR_OUTDOOR_PM   = "outdoor_pm"
SENSOR_INDOOR_PM    = "indoor_pm"

def compute_fdi_temp_value(severity, base_normal=20.0):
    """map severity (0–1) injected temperature reading
    severity=0  →  25 °C  (small deviation, easy to miss)
    severity=1  →  95 °C  (extreme, clearly anomalous)
    """
    return base_normal + 5.0 + severity * 70.0


def compute_fdi_pm_value(severity, base_normal=10.0):
    """map severity (0–1) injected PM2.5/PM10 reading (µg/m³)
    severity=0  →   60 µg/m³  (mild spike)
    severity=1  →  400 µg/m³  (extreme spike)
    """
    return base_normal + 50.0 + severity * 340.0

def make_fdi_temp_event(target_house, start_time, duration, severity=0.5, fake_val=None, event_id=None):
    if event_id is None:
        event_id = f"fdi_temp_{uuid.uuid4().hex[:8]}"
    params = {"sensor": SENSOR_OUTDOOR_TEMP}
    if fake_val is not None:
        params["fake_val"] = float(fake_val)
    return {
        "event_id":     event_id,
        "event_type":   EVENT_TYPE_FDI_TEMP,
        "target_house": int(target_house),
        "start_time":   int(start_time),
        "duration":     int(duration),
        "severity":     round(float(severity), 4),
        "parameters":   params,
    }

def make_fdi_pm_event(target_house, start_time, duration, severity=0.5, fake_val=None, event_id=None):
    if event_id is None:
        event_id = f"fdi_pm_{uuid.uuid4().hex[:8]}"
    params = {"sensor": SENSOR_OUTDOOR_PM}
    if fake_val is not None:
        params["fake_val"] = float(fake_val)
    return {
        "event_id":     event_id,
        "event_type":   EVENT_TYPE_FDI_PM,
        "target_house": int(target_house),
        "start_time":   int(start_time),
        "duration":     int(duration),
        "severity":     round(float(severity), 4),
        "parameters":   params,
    }

def apply_fdi_temp(observed, house_idx, severity, params=None):
    params   = params or {}
    fake_val = params.get('fake_val', compute_fdi_temp_value(severity))
    observed[house_idx, SENSOR_INDICES['out_temp']] = float(fake_val)
    return observed

def apply_fdi_pm(observed, house_idx, severity, params=None):
    params   = params or {}
    fake_val = params.get('fake_val', compute_fdi_pm_value(severity))
    observed[house_idx, SENSOR_INDICES['out_pm']] = float(fake_val)
    return observed

def generate_fdi_scenario(num_events=5, total_steps=2880, num_houses=10, seed=42):
    rng    = np.random.RandomState(seed)
    events = []
    fdi_types = [EVENT_TYPE_FDI_TEMP, EVENT_TYPE_FDI_PM]

    for _ in range(num_events):
        fdi_type = rng.choice(fdi_types)
        house    = int(rng.randint(0, num_houses))
        start    = int(rng.randint(0, max(1, total_steps - 20)))
        duration = int(rng.randint(2, 12))
        severity = round(float(rng.uniform(0.3, 1.0)), 2)

        if fdi_type == EVENT_TYPE_FDI_TEMP:
            events.append(make_fdi_temp_event(house, start, duration, severity))
        else:
            events.append(make_fdi_pm_event(house, start, duration, severity))
    return events

if __name__ == "__main__":
    print("=== FDI_attacker self-test ===")

    #factory functions
    ev1 = make_fdi_temp_event(target_house=3, start_time=120, duration=10, severity=0.7, fake_val=95.0)
    ev2 = make_fdi_pm_event(target_house=5, start_time=250, duration=6, severity=0.5)
    print("FDI-temp event:", json.dumps(ev1, indent=2))
    print("FDI-pm event:  ", json.dumps(ev2, indent=2))

    #value derivation
    for sev in [0.0, 0.5, 1.0]:
        print(f"  severity={sev:.1f} → temp={compute_fdi_temp_value(sev):.1f}°C,"
              f"  pm={compute_fdi_pm_value(sev):.1f} µg/m³")

    #batch generator
    batch = generate_fdi_scenario(num_events=3, seed=7)
    print(f"\nGenerated {len(batch)} FDI events:")
    for ev in batch:
        print(f"  [{ev['event_id']}] {ev['event_type']} house={ev['target_house']}"
              f" t={ev['start_time']}+{ev['duration']} sev={ev['severity']}")
