import uuid
import numpy as np
import json
from environment.iot_env import SENSOR_INDICES

EVENT_TYPE_FIRE    = "Fire"
EVENT_TYPE_BBQ     = "BBQ"
EVENT_TYPE_SMOKING = "Smoking"

def make_fire_event(target_house, start_time, duration, severity=1.0, event_id=None):
    if event_id is None:
        event_id = f"fire_{uuid.uuid4().hex[:8]}"
    return {
        "event_id":     event_id,
        "event_type":   EVENT_TYPE_FIRE,
        "target_house": int(target_house),
        "start_time":   int(start_time),
        "duration":     int(duration),
        "severity":     round(float(severity), 4),
        "parameters":   {},
    }

def make_bbq_event(target_house, start_time, duration, severity=0.7, event_id=None):
    if event_id is None:
        event_id = f"bbq_{uuid.uuid4().hex[:8]}"
    return {
        "event_id":     event_id,
        "event_type":   EVENT_TYPE_BBQ,
        "target_house": int(target_house),
        "start_time":   int(start_time),
        "duration":     int(duration),
        "severity":     round(float(severity), 4),
        "parameters":   {},
    }

def make_smoking_event(target_house, start_time, duration, severity=0.5, event_id=None):
    if event_id is None:
        event_id = f"smoking_{uuid.uuid4().hex[:8]}"
    return {
        "event_id":     event_id,
        "event_type":   EVENT_TYPE_SMOKING,
        "target_house": int(target_house),
        "start_time":   int(start_time),
        "duration":     int(duration),
        "severity":     round(float(severity), 4),
        "parameters":   {},
    }

def apply_fire_effect(observed, house_idx, severity, step_in_event=0):
    si = SENSOR_INDICES
    temp_rise = min(severity * 25.0, step_in_event * 8.0 + severity * 3.0)
    observed[house_idx, si['out_temp']] += temp_rise
    observed[house_idx, si['in_temp']]  += temp_rise * 0.8
    observed[house_idx, si['out_hum']]   = max(20.0, observed[house_idx, si['out_hum']] - severity * 20.0)
    observed[house_idx, si['in_hum']]    = max(20.0, observed[house_idx, si['in_hum']]  - severity * 20.0)
    pm = severity * 500.0
    observed[house_idx, si['out_pm']] = pm
    observed[house_idx, si['in_pm']]  = pm
    return observed

def apply_bbq_effect(observed, house_idx, severity):
    si = SENSOR_INDICES
    observed[house_idx, si['out_temp']] += severity * 3.0
    observed[house_idx, si['out_hum']]   = max(20.0, observed[house_idx, si['out_hum']] - severity * 5.0)
    observed[house_idx, si['out_pm']]   += severity * 150.0
    return observed

def apply_smoking_effect(observed, house_idx, severity):
    observed[house_idx, SENSOR_INDICES['in_pm']] += severity * 80.0
    return observed

def generate_accident_scenario(num_events=5, total_steps=2880, num_houses=10, seed=42):
    rng    = np.random.RandomState(seed)
    events = []
    event_types   = [EVENT_TYPE_FIRE, EVENT_TYPE_BBQ, EVENT_TYPE_SMOKING]
    event_weights = [0.20, 0.40, 0.40]   # fewer fires more everyday events

    for _ in range(num_events):
        etype    = rng.choice(event_types, p=event_weights)
        house    = int(rng.randint(0, num_houses))
        start    = int(rng.randint(0, max(1, total_steps - 20)))
        severity = round(float(rng.uniform(0.4, 1.0)), 2)

        if etype == EVENT_TYPE_FIRE:
            duration = int(rng.randint(6, 12))
            events.append(make_fire_event(house, start, duration, severity))
        elif etype == EVENT_TYPE_BBQ:
            duration = int(rng.randint(2, 6))
            events.append(make_bbq_event(house, start, duration, severity))
        else:
            duration = int(rng.randint(1, 3))
            events.append(make_smoking_event(house, start, duration, severity))
    return events

if __name__ == "__main__":
    print("=== accident_trigger self-test ===")
    ev1 = make_fire_event(target_house=1, start_time=96, duration=8, severity=0.9)
    ev2 = make_bbq_event(target_house=4, start_time=300, duration=4, severity=0.65)
    ev3 = make_smoking_event(target_house=7, start_time=500, duration=2, severity=0.5)
    
    print("Fire event:", json.dumps(ev1, indent=2))
    print("BBQ event: ", json.dumps(ev2, indent=2))
    print("Smoking event: ", json.dumps(ev3, indent=2))

    #batch generator
    batch = generate_accident_scenario(num_events=4, seed=13)
    print(f"\nGenerated {len(batch)} accident events:")
    for ev in batch:
        print(f"  [{ev['event_id']}] {ev['event_type']} house={ev['target_house']}"
              f" t={ev['start_time']}+{ev['duration']} sev={ev['severity']}")
