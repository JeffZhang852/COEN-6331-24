# Environment Data Generator for IoT DRL Defense Project.
#
# Generates 30 days of synthetic sensor data for 10 houses.
# Each house has 7 sensors:
# - Outdoor Temp (°C)
# - Outdoor Humidity (%)
# - Outdoor Particles (µg/m³)
# - Indoor Temp (°C)
# - Indoor Humidity (%)
# - Indoor Particles (µg/m³)
# - Electricity Usage (kW)
#
# Output file: environment/environment_data_30days.txt
# Format per line: house_id,time,out_temp,out_hum,out_pm,in_temp,in_hum,in_pm,power
#
# This is for "real" data that creates variation for each house.

import math
import random

# 1) CONFIGURATION
NUM_DAYS = 30
NUM_HOUSES = 10
INTERVAL_MIN = 15
TIMES_PER_DAY = 96          # 24 hours * 4 slots per hour (15‑min intervals)

# Weather settings for a typical Summer climate
TEMP_HIGH = 28.0            # °C hottest part of the afternoon
TEMP_LOW = 14.0             # °C coldest part of the early morning
HUMIDITY_HIGH = 85.0        # % humid mornings
HUMIDITY_LOW = 45.0         # % drier afternoons
RAIN_PROB = 0.28            # chance of rain on any given day

# Baseline outdoor particulate matter for clean air
OUTDOOR_PM_BASE = 5.0       # µg/m³

# 2) HOUSEHOLD PROFILES. To add randomness to mimic "real" life
# name: description of the occupant
# temp_setpoint: preferred indoor temperature °C
# insulation: 0.7 = well insulated, 1.2 = drafty affects how fast indoor temp reacts
# elec_mult: electricity usage multiplier
# indoor_pm_base: typical particle level indoors µg/m³
# indoor_hum_offset: extra humidity on top of outdoor value %
# cooking_hours: time indices when cooking causes particle spikes
# outdoor_pm_offset: micro‑location effect. Like trees, gardens, etc

PROFILES = [
    # House 1: Elderly Couple – warm home, low power, a bit dusty
    {"name": "Elderly Couple", "temp_setpoint": 23.0, "insulation": 0.8, "elec_mult": 0.8,
     "indoor_pm_base": 8.0, "indoor_hum_offset": 2.0, "cooking_hours": [28, 48, 76],
     "outdoor_pm_offset": 1.0},
    # House 2: Family with Kids – cooler, high power usage, messy
    {"name": "Family with Kids", "temp_setpoint": 21.0, "insulation": 0.9, "elec_mult": 1.4,
     "indoor_pm_base": 12.0, "indoor_hum_offset": 5.0, "cooking_hours": [26, 50, 78],
     "outdoor_pm_offset": 2.0},
    # House 3: Work‑from‑Home Professional – clean, moderate power
    {"name": "WFH Professional", "temp_setpoint": 22.5, "insulation": 0.7, "elec_mult": 1.2,
     "indoor_pm_base": 5.0, "indoor_hum_offset": -2.0, "cooking_hours": [30, 52, 80],
     "outdoor_pm_offset": 0.5},
    # House 4: Night Owl Gamer – cool, high power at night
    {"name": "Night Owl Gamer", "temp_setpoint": 20.0, "insulation": 1.0, "elec_mult": 1.6,
     "indoor_pm_base": 6.0, "indoor_hum_offset": -1.0, "cooking_hours": [32, 84],
     "outdoor_pm_offset": 1.5},
    # House 5: Retired Gardener – warm, windows open, pollen
    {"name": "Retired Gardener", "temp_setpoint": 24.0, "insulation": 1.1, "elec_mult": 0.9,
     "indoor_pm_base": 15.0, "indoor_hum_offset": 8.0, "cooking_hours": [28, 48, 76],
     "outdoor_pm_offset": 3.0},
    # House 6: Minimalist Single – efficient, very clean
    {"name": "Minimalist Single", "temp_setpoint": 21.5, "insulation": 0.8, "elec_mult": 0.6,
     "indoor_pm_base": 4.0, "indoor_hum_offset": -3.0, "cooking_hours": [30, 78],
     "outdoor_pm_offset": 0.0},
    # House 7: Pet Owner – normal power, pet dander
    {"name": "Pet Owner", "temp_setpoint": 22.0, "insulation": 0.9, "elec_mult": 1.0,
     "indoor_pm_base": 20.0, "indoor_hum_offset": 4.0, "cooking_hours": [28, 50, 76],
     "outdoor_pm_offset": 2.0},
    # House 8: Home Baker – oven use, flour dust, extra humidity
    {"name": "Home Baker", "temp_setpoint": 21.0, "insulation": 0.85, "elec_mult": 1.3,
     "indoor_pm_base": 10.0, "indoor_hum_offset": 10.0, "cooking_hours": [24, 44, 64],
     "outdoor_pm_offset": 1.0},
    # House 9: Tech Enthusiast – servers, HEPA filter, very clean
    {"name": "Tech Enthusiast", "temp_setpoint": 20.5, "insulation": 0.7, "elec_mult": 1.5,
     "indoor_pm_base": 3.0, "indoor_hum_offset": -4.0, "cooking_hours": [30, 80],
     "outdoor_pm_offset": 0.5},
    # House 10: Frequent Traveler – often away, low power
    {"name": "Frequent Traveler", "temp_setpoint": 22.0, "insulation": 1.0, "elec_mult": 0.5,
     "indoor_pm_base": 7.0, "indoor_hum_offset": 0.0, "cooking_hours": [28, 76],
     "outdoor_pm_offset": 1.0}
]

# Electricity load profile over a day
# Values are multipliers applied to a base of 0.5 kW
# The pattern: low overnight, morning bump, midday dip, evening peak
BASE_LOAD_PROFILE = [
    # 00:00 – 05:45 – people sleep
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    # 06:00 – 08:45 – morning ramp
    0.6, 0.7, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3,
    # 09:00 – 14:45 – midday low
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    # 15:00 – 16:45 – late afternoon
    0.4, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9,
    # 17:00 – 19:45 – evening peak: cooking, lights, entertainment
    1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
    # 20:00 – 23:45 – wind down
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3
]

# 3) FUNCTIONS
def time_index_to_str(idx):
    # Turn a slot index (0–95) into a readable HH:MM string.
    total_minutes = idx * INTERVAL_MIN
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"

def generate_outdoor_profile(day_num):
    # Build one full day of outdoor temperature and humidity data.
    # Returns two lists: temperatures and humidities, each with 96 values.
    temps = []
    hums = []

    # Day‑to‑day variation, no two days are exactly alike
    day_offset_temp = random.uniform(-1.5, 1.5)
    day_offset_hum = random.uniform(-5.0, 5.0)

    # Decide randomly whether it rains today
    is_rainy = random.random() < RAIN_PROB
    rain_start = random.randint(30, 50)      # start between 7:30 and 12:30
    rain_duration = random.randint(8, 24)    # last 2 to 6 hours

    for i in range(TIMES_PER_DAY):
        # Create a smooth sine wave for temperature.
        # Peak at index 60 (15:00), trough at index 12 (03:00).
        hour_angle = (i - 60) * (2 * math.pi / TIMES_PER_DAY)
        temp_range = (TEMP_HIGH - TEMP_LOW) / 2
        temp_mid = (TEMP_HIGH + TEMP_LOW) / 2
        base_temp = temp_mid + temp_range * math.cos(hour_angle)

        # Humidity is roughly the inverse of temperature.
        hum_range = (HUMIDITY_HIGH - HUMIDITY_LOW) / 2
        hum_mid = (HUMIDITY_HIGH + HUMIDITY_LOW) / 2
        base_hum = hum_mid - hum_range * math.cos(hour_angle)

        # If it's raining, cool it down and add moisture.
        if is_rainy and rain_start <= i < rain_start + rain_duration:
            base_temp -= random.uniform(1.0, 2.5)
            base_hum += random.uniform(5.0, 12.0)

        # Sprinkle in a tiny bit of randomness
        temp = base_temp + day_offset_temp + random.uniform(-0.3, 0.3)
        hum = base_hum + day_offset_hum + random.uniform(-1.5, 1.5)

        # Keep humidity in the physically possible range
        hum = max(20.0, min(100.0, hum))

        temps.append(round(temp, 1))
        hums.append(round(hum, 1))

    return temps, hums


def generate_house_data(house_id, outdoor_temps, outdoor_hums):
    # Build a full day of data for one specific house using its unique profile.
    # Returns a list of formatted strings, one per 15‑minute slot.
    profile = PROFILES[house_id - 1]
    rows = []

    # Starting indoor conditions
    indoor_temp = profile["temp_setpoint"]
    indoor_hum = outdoor_hums[0] * 0.7 + profile["indoor_hum_offset"] + random.uniform(-2, 2)

    for i in range(TIMES_PER_DAY):
        out_temp = outdoor_temps[i]
        out_hum = outdoor_hums[i]

        # Outdoor particles – baseline plus house‑specific micro‑location offset
        out_pm = OUTDOOR_PM_BASE + profile["outdoor_pm_offset"] + random.uniform(-2.0, 2.0)
        out_pm = max(0.0, round(out_pm, 1))

        # Indoor temperature – drifts toward a mix of setpoint and outdoor temp
        eq_weight_out = 0.3 * profile["insulation"]
        eq_weight_set = 1.0 - eq_weight_out
        equilibrium = eq_weight_set * profile["temp_setpoint"] + eq_weight_out * out_temp
        indoor_temp = indoor_temp * 0.9 + equilibrium * 0.1 + random.uniform(-0.1, 0.1)

        # Indoor humidity – dampened outdoor value plus house‑specific offset
        indoor_hum = out_hum * 0.6 + 15.0 + profile["indoor_hum_offset"] + random.uniform(-1.5, 1.5)
        indoor_hum = max(20.0, min(80.0, indoor_hum))

        # Indoor particles – base level plus extra if cooking right now
        indoor_pm = profile["indoor_pm_base"]
        if i in profile["cooking_hours"]:
            indoor_pm += random.uniform(15.0, 40.0)
        indoor_pm += random.uniform(-3.0, 3.0)
        indoor_pm = max(0.0, round(indoor_pm, 1))

        # Electricity usage – base profile multiplied by house factor
        base_power = 0.5 * BASE_LOAD_PROFILE[i]
        power = base_power * profile["elec_mult"]

        # Extra power for special habits
        if profile["name"] == "Night Owl Gamer" and (i < 12 or i > 80):
            power *= 1.3
        if profile["name"] == "Home Baker" and i in profile["cooking_hours"]:
            power += 0.8

        power += random.uniform(-0.03, 0.03)
        power = max(0.0, round(power, 2))

        # Time display
        time_str = time_index_to_str(i)

        # Build the final line for this slot
        line = f"{house_id},{time_str},{out_temp:.1f},{out_hum:.1f},{out_pm:.1f},{indoor_temp:.1f},{indoor_hum:.1f},{indoor_pm:.1f},{power:.2f}"
        rows.append(line)

    return rows


# 4) DATA GENERATION

def main():
    output_file = "environment/environment_data_30days.txt"
    print(f"Generating 30 days of data for {NUM_HOUSES} houses with realistic profiles")

    with open(output_file, 'w') as f:
        for day in range(1, NUM_DAYS + 1):
            # Mark the start of a new day in the output file
            f.write(f"#Day {day}\n")

            # Get the outdoor conditions for this whole day
            outdoor_temps, outdoor_hums = generate_outdoor_profile(day)

            # Now write the data for every house, one by one
            for house_id in range(1, NUM_HOUSES + 1):
                house_rows = generate_house_data(house_id, outdoor_temps, outdoor_hums)
                for row in house_rows:
                    f.write(row + "\n")

            # Show a little progress every 5 days
            if day % 5 == 0:
                print(f"  Completed Day {day}")

    print(f"\nData generation complete. Output saved to: {output_file}")
    print("Profiles used:")
    for i, p in enumerate(PROFILES, 1):
        print(f"  House {i}: {p['name']}  PM base={p['indoor_pm_base']}  Elec mult={p['elec_mult']}")


if __name__ == "__main__":
    main()