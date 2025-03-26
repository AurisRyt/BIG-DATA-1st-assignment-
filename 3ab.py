import pandas as pd
import numpy as np
import multiprocessing as mp
from geopy.distance import geodesic
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import psutil
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# Kodo eiliškumas:
# 1. Parametrų ir aplinkos nustatymai: failo kelias, testavimo režimas, riba laiko tarp AIS įrašų, fallback šuolio riba.
# 2. Konfigūracijos sąrašas su skirtingais chunk/core deriniais, skirtais testuoti našumą.
# 3. Pagalbinės funkcijos: chunkifikavimas, dinaminio slenksčio ir greičio santykio ribų skaičiavimas.
# 4. Pagrindinis darbininkas (`detect_anomalies`) apdoroja AIS duomenis kiekvienam laivui, skaičiuoja greitį, atstumą, ratio ir žymi anomalijas.
# 5. `run_sequential` – paleidžia viengijų analizę visam laivų sąrašui.
# 6. `run_parallel` – paleidžia analizę paraleliai per multiprocessing Pool.
# 7. `main` – duomenų įkėlimas, filtravimas, vykdymas pagal kiekvieną konfigūraciją, matavimai, vizualizacijos.


# ------- SETTINGS -------
FILE_PATH = 'cleaned_ais_data.csv'
TEST_MODE = False
NROWS = 2500000
TIME_THRESHOLD_SEC = 10
JUMP_FALLBACK_KM = 50

CONFIGS = [
    ('Sequential', None, None),
    ('2 chunks / 2 cores', 2, 2),
    ('2 chunks / 4 cores', 2, 4),
    ('4 chunks / 2 cores', 4, 2),
    ('4 chunks / 4 cores', 4, 4),
    ('8 chunks / 4 cores', 8, 4),
    ('3 chunks / 6 cores', 3, 6),
    ('6 chunks / 6 cores', 6, 6),
    ('12 chunks / 6 cores', 12, 6),
    ('4 chunks / 8 cores', 4, 8),
    ('8 chunks / 8 cores', 8, 8),

]

# ------- UTILS -------
def chunkify(data, num_chunks):
    avg = len(data) // num_chunks
    return [data[i * avg:(i + 1) * avg] if i < num_chunks - 1 else data[i * avg:] for i in range(num_chunks)]

def compute_dynamic_threshold(speeds_nm_per_hr, percentile=95, fallback_km=JUMP_FALLBACK_KM):
    try:
        if len(speeds_nm_per_hr) == 0:
            return fallback_km
        speed_thresh = np.percentile(speeds_nm_per_hr, percentile)
        max_dist_km = (speed_thresh / 3600) * TIME_THRESHOLD_SEC * 1.852
        return max(max_dist_km, 1.0)
    except Exception:
        return fallback_km

def compute_ratio_bounds(ratios, lower_pct=5, upper_pct=95):
    try:
        lower = np.percentile(ratios, lower_pct)
        upper = np.percentile(ratios, upper_pct)
        return max(lower, 0.01), min(upper, 100.0)
    except Exception:
        return 0.1, 10.0

# ------- MERGED WORKER WITH DYNAMIC THRESHOLDS -------
def detect_anomalies(mmsi_group):
    mmsi, group = mmsi_group
    group = group.sort_values('timestamp')
    group['timestamp'] = pd.to_datetime(group['timestamp'])
    jumps, speed_mismatches = [], []

    coords = list(zip(group['latitude'], group['longitude']))
    sogs = group['sog'].tolist()
    times = group['timestamp'].tolist()

    speeds_nm_per_hr = []
    ratios_all = []

    for i in range(1, len(coords)):
        t1, t2 = times[i - 1], times[i]
        time_diff_hr = (t2 - t1).total_seconds() / 3600
        if time_diff_hr * 3600 < TIME_THRESHOLD_SEC:
            continue
        dist_km = geodesic(coords[i - 1], coords[i]).km
        dist_nm = dist_km * 0.539957
        if time_diff_hr > 0:
            calc_speed = dist_nm / time_diff_hr
            sog = sogs[i]
            speeds_nm_per_hr.append(calc_speed)
            if calc_speed > 0:
                ratios_all.append(sog / calc_speed)

    threshold_km = compute_dynamic_threshold(speeds_nm_per_hr)
    ratio_lower, ratio_upper = compute_ratio_bounds(ratios_all)

    for i in range(1, len(coords)):
        t1, t2 = times[i - 1], times[i]
        time_diff_hr = (t2 - t1).total_seconds() / 3600
        if time_diff_hr * 3600 < TIME_THRESHOLD_SEC:
            continue

        dist_km = geodesic(coords[i - 1], coords[i]).km
        dist_nm = dist_km * 0.539957
        calc_speed = dist_nm / time_diff_hr if time_diff_hr > 0 else 0
        sog = sogs[i]

        if dist_km > threshold_km:
            jumps.append({
                'mmsi': mmsi,
                'jump_km': dist_km,
                'start_time': t1,
                'end_time': t2,
                'threshold_km': threshold_km
            })

        if calc_speed != 0:
            ratio = sog / calc_speed
        else:
            ratio = np.inf
        if ratio < ratio_lower or ratio > ratio_upper:
            speed_mismatches.append({
                'mmsi': mmsi,
                'sog': sog,
                'calc_speed': calc_speed,
                'start_time': t1,
                'end_time': t2,
                'ratio': ratio,
                'dynamic_lower': ratio_lower,
                'dynamic_upper': ratio_upper
            })

    return jumps, speed_mismatches

# ------- SEQUENTIAL -------
def run_sequential(mmsi_groups):
    jumps, mismatches = [], []
    for group in tqdm(mmsi_groups, desc="Sequential"):
        j, m = detect_anomalies(group)
        jumps.extend(j)
        mismatches.extend(m)
    return pd.DataFrame(jumps), pd.DataFrame(mismatches)

# ------- PARALLEL -------
def run_parallel(mmsi_groups, num_cores):
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(detect_anomalies, mmsi_groups), total=len(mmsi_groups)))
    jumps, mismatches = [], []
    for j, m in results:
        jumps.extend(j)
        mismatches.extend(m)
    return pd.DataFrame(jumps), pd.DataFrame(mismatches)

# ------- MAIN -------
def main():
    print("Loading and preparing data...")
    if TEST_MODE:
        df = pd.read_csv(FILE_PATH, nrows=NROWS)
        print(f"⚠️ Test mode: using {NROWS} rows")
    else:
        df = pd.read_csv(FILE_PATH)

    df = df[(df['is_valid_position'] == 1) & (df['is_moving'] == 1)].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Filtered data: {len(df)} rows, {df['mmsi'].nunique()} vessels")

    grouped = list(df.groupby('mmsi'))
    runtimes, cpu_usages, mem_usages, peak_cpus, peak_mems = {}, {}, {}, {}, {}

    for label, chunks, cores in CONFIGS:
        print(f"\nRunning: {label}")
        cpu_profile, mem_profile = [], []

        def sample():
            proc = psutil.Process()
            while not stop_signal:
                cpu_profile.append(psutil.cpu_percent(interval=None))
                mem_profile.append(proc.memory_info().rss / 1024**2)
                time.sleep(1)

        stop_signal = False
        sampler = threading.Thread(target=sample)
        sampler.start()

        start_time = time.time()
        process = psutil.Process()
        cpu_start = process.cpu_times()
        mem_before = process.memory_info().rss / (1024 ** 2)

        if label == 'Sequential':
            jumps_df, mismatches_df = run_sequential(grouped)
        else:
            mmsi_chunks = chunkify(grouped, chunks)
            jump_chunks, mismatch_chunks = [], []
            for chunk in mmsi_chunks:
                j_df, m_df = run_parallel(chunk, num_cores=cores)
                jump_chunks.append(j_df)
                mismatch_chunks.append(m_df)
            jumps_df = pd.concat(jump_chunks, ignore_index=True)
            mismatches_df = pd.concat(mismatch_chunks, ignore_index=True)

        duration = time.time() - start_time
        stop_signal = True
        sampler.join()

        mem_after = process.memory_info().rss / (1024 ** 2)
        cpu_end = process.cpu_times()

        runtimes[label] = duration
        mem_usages[label] = mem_after - mem_before
        cpu_usages[label] = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        peak_cpus[label] = max(cpu_profile or [0])
        peak_mems[label] = max(mem_profile or [0])

        print(f"{label} finished in {duration:.2f}s | Jumps: {len(jumps_df)}, Speed mismatches: {len(mismatches_df)}")

    labels = list(runtimes.keys())
    times = list(runtimes.values())
    baseline = times[0]
    speedups = [baseline / t for t in times]
    cpu_vals = [cpu_usages[k] for k in labels]
    mem_vals = [mem_usages[k] for k in labels]
    peak_cpu_vals = [peak_cpus[k] for k in labels]
    peak_mem_vals = [peak_mems[k] for k in labels]

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))

    axs[0, 0].bar(labels, times)
    axs[0, 0].set_title("Execution Time")
    axs[0, 0].set_ylabel("Seconds")
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].bar(labels, speedups)
    axs[0, 1].set_title("Speedup vs. Sequential")
    axs[0, 1].set_ylabel("Speedup (x)")
    axs[0, 1].tick_params(axis='x', rotation=45)

    axs[1, 0].bar(labels, cpu_vals)
    axs[1, 0].set_title("CPU Time Used")
    axs[1, 0].set_ylabel("CPU seconds")
    axs[1, 0].tick_params(axis='x', rotation=45)

    axs[1, 1].bar(labels, mem_vals)
    axs[1, 1].set_title("Memory Delta")
    axs[1, 1].set_ylabel("MB used")
    axs[1, 1].tick_params(axis='x', rotation=45)

    axs[2, 0].bar(labels, peak_cpu_vals)
    axs[2, 0].set_title("Peak CPU Usage")
    axs[2, 0].set_ylabel("% usage")
    axs[2, 0].tick_params(axis='x', rotation=45)

    axs[2, 1].bar(labels, peak_mem_vals)
    axs[2, 1].set_title("Peak Memory Usage")
    axs[2, 1].set_ylabel("MB")
    axs[2, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # ------- Anomalies Visualization -------
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 10))

    jumps_df_sorted = jumps_df.sort_values('start_time')
    axs2[0].plot(jumps_df_sorted['start_time'], jumps_df_sorted['jump_km'], marker='o', linestyle='-', color='tab:red',
                 alpha=0.7)
    axs2[0].set_title("Jump Anomalies Over Time")
    axs2[0].set_xlabel("Time")
    axs2[0].set_ylabel("Jump Distance (km)")

    mismatches_df_sorted = mismatches_df.sort_values('start_time')
    axs2[1].plot(mismatches_df_sorted['start_time'], mismatches_df_sorted['ratio'], marker='o', linestyle='-',
                 color='tab:blue', alpha=0.7)
    axs2[1].set_title("Speed Ratio Anomalies Over Time")
    axs2[1].set_xlabel("Time")
    axs2[1].set_ylabel("SOG / Calculated Speed Ratio")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mp.freeze_support()
    main()

