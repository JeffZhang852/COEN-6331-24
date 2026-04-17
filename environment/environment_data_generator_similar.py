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
# This is for similar data that does not create randomness for individual houses.

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

# Indoor temperature preference for each of the 10 houses
INDOOR_TEMP_SETPOINTS = [21.0, 22.0, 20.5, 22.5, 21.5, 23.0, 20.0, 22.0, 21.0, 21.8]

# Electricity load profile over a day
# Values are multipliers applied to a base of 0.5 kW
# The pattern: low overnight, morning bump, midday dip, evening peak
BASE_LOAD_PROFILE = [
    # 00:00 – 05:45 - people sleep
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    # 06:00 – 11:45 – people wake up, use appliances
    0.6, 0.7, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    # 12:00 – 17:45 – lower usage while many are away
    0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
    0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8,
    # 18:00 – 23:45 – evening peak: cooking, lights, entertainment
    0.9, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
    0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2
]

# Small per‑house adjustments to electricity usage,some homes use more, some less
HOUSE_POWER_MULTIPLIERS = [0.8, 1.1, 0.9, 1.3, 0.7, 1.0, 1.2, 0.85, 0.95, 1.05]

# Baseline particulate matter reading for clean air, plus allowed noise
PM_BASELINE = 5.0      # µg/m³ for a clean outdoor environment
PM_NOISE = 2.0         # random dirtiness


# 2) FUNCTIONS

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
    # Build a full day of data for one specific house.
    # Returns a list of formatted strings, one per 15‑minute slot.
    rows = []

    # Grab the house's preferred indoor temperature and power habit
    setpoint = INDOOR_TEMP_SETPOINTS[house_id - 1]
    power_mult = HOUSE_POWER_MULTIPLIERS[house_id - 1]

    # Each house has a slightly different indoor humidity offset
    indoor_hum_offset = random.uniform(-5.0, 5.0)

    # Start the indoor temperature at the preferred setpoint
    indoor_temp = setpoint

    for i in range(TIMES_PER_DAY):
        # Current outdoor conditions
        out_temp = outdoor_temps[i]
        out_hum = outdoor_hums[i]

        # Outdoor particles clean baseline with a little noise
        out_pm = PM_BASELINE + random.uniform(-PM_NOISE, PM_NOISE)
        out_pm = max(0.0, round(out_pm, 1))

        # Indoor temperature drifts slowly toward a balance between
        # the thermostat setting and the outdoor temperature.
        equilibrium = 0.7 * setpoint + 0.3 * out_temp
        indoor_temp = indoor_temp * 0.9 + equilibrium * 0.1
        indoor_temp += random.uniform(-0.1, 0.1)

        # Indoor humidity follows outdoor, but dampened and with a house‑specific offset.
        indoor_hum = out_hum * 0.8 + 10.0 + indoor_hum_offset
        indoor_hum += random.uniform(-1.0, 1.0)
        indoor_hum = max(20.0, min(80.0, indoor_hum))

        # Indoor particles clean air unless something unusual happens.
        indoor_pm = PM_BASELINE + random.uniform(-PM_NOISE / 2, PM_NOISE / 2)
        indoor_pm = max(0.0, round(indoor_pm, 1))

        # Electricity consumption for this 15‑minute slot.
        base_power = 0.5 * BASE_LOAD_PROFILE[i]
        power = base_power * power_mult + random.uniform(-0.02, 0.02)
        power = max(0.0, round(power, 2))

        # Time display
        time_str = time_index_to_str(i)

        # Build the final line for this slot
        line = f"{house_id},{time_str},{out_temp:.1f},{out_hum:.1f},{out_pm:.1f},{indoor_temp:.1f},{indoor_hum:.1f},{indoor_pm:.1f},{power:.2f}"
        rows.append(line)

    return rows

# 3) DATA GENERATION
def main():
    output_file = "environment/environment_data_30days.txt"
    print(f"Generating 30 days of data for {NUM_HOUSES} houses")

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
    print(f"Total lines: {NUM_DAYS * NUM_HOUSES * TIMES_PER_DAY} data rows (plus day headers)")

if __name__ == "__main__":
    main()