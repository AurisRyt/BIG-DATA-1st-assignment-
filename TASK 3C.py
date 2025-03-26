import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import psutil
import os
import sys
# ------------------------------------------------------------------------------
# GPS Spoofing Detection – Method C (Positional + COG Conflict Detection)
# ------------------------------------------------------------------------------
# Šis skriptas analizuoja AIS duomenis ir aptinka galimus GPS spoofing atvejus
# remiantis laiko, erdvės ir kurso (COG) konfliktais tarp laivų.
#
# Kodo eiliškumas:
# 1. Parametrų apibrėžimas (failų keliai, nustatymai, slenksčiai, branduolių skaičius).
# 2. `ProgressBar` klasė – naudojama gražiam progreso atvaizdavimui konsolėje.
# 3. `haversine_np` – funkcija atstumo tarp koordinačių skaičiavimui jūrmylėmis.
# 4. `load_data` – įkelia ir išfiltruoja AIS duomenis pagal kokybę ir judėjimą.
# 5. `process_bin` – pagrindinė funkcija, kuri analizuoja vieną erdvinį-laiko "bin'ą"
#    ir aptinka konfliktus tarp netoliese esančių laivų:
#    - Patikrina, ar laivai arti vienas kito (poziciškai).
#    - Įvertina COG skirtumą (didesnį nei 150°).
#    - Užfiksuoja konfliktus kaip "positional", "directional" arba "both".
# 6. `process_chunk` – apdoroja visus bin'us viename duomenų gabale, galima seka ar paraleliai.
# 7. `run_sequential` – paleidžia detektorių vienu branduoliu, be padalijimo į laiko chunk'us.
# 8. `run_parallel` – paleidžia analizę paraleliai, padalijus laiką į chunk'us ('1h', '2h').
# 9. `run_benchmark` – paleidžia visus testus (1, 2, 4 branduoliai), surenka metrikas:
#    - Trukmė, CPU laikas, RAM naudojimas, konfliktų skaičius, greitaveika (speedup).
# 10. `analyze_conflicts` – apžvelgia aptiktus konfliktus:
#     - Laiko pasiskirstymas-chunks (valandomis)
#     - Dažniausiai į konfliktus įsivėlę laivai
#     - Galimi spoofing atvejai (artumas < 0.1 NM)
#     - Rezultatai eksportuojami į CSV
# 11. `main` – valdymo funkcija: įkelia duomenis, paleidžia benchmark’ą ir analizę.
# SETTINGS
FILE_PATH = '/Users/studentas/Desktop/BIG DATA/Main data set/cleaned_ais_data.csv'
OUTPUT_FILE = 'flagged_conflicts.csv'
PERF_LOG = 'benchmark_results.csv'
PROXIMITY_RADIUS_NM = 5.0
COG_DIFF_THRESHOLD = 150
GRID_SIZE = 0.05
TIME_BIN_MIN = 10
NM_TO_DEG = 1 / 60
NROWS = None

CORE_CONFIGS = [1, 2, 4]
TIME_CHUNKS = ['1h', '2h']
NEIGHBOR_OFFSETS = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]

class ProgressBar:
    def __init__(self):
        self.start_time = time.time()

    def update(self, current, total, label=""):
        percent = 100 * (current / float(total)) if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        elapsed = time.time() - self.start_time
        eta = (elapsed / current) * (total - current) if current > 0 else 0
        time_info = f"{elapsed:.1f}s elapsed, ETA: {eta:.1f}s"
        sys.stdout.write(f"\r{label} |{bar}| {percent:.1f}% ({current}/{total}) {time_info}")
        sys.stdout.flush()
        if current == total:
            sys.stdout.write('\n')

progress = ProgressBar()

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a)) * 0.539957

def load_data():
    print("Loading data...")
    start_time = time.time()
    df = pd.read_csv(FILE_PATH, nrows=NROWS)
    original_count = len(df)
    df = df[(df['is_valid_position'] == 1) & (df['is_moving'] == 1) & (df['sog'] > 1)].copy()
    filtered_count = len(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['latitude'] = pd.to_numeric(df['latitude'], downcast='float')
    df['longitude'] = pd.to_numeric(df['longitude'], downcast='float')
    df['cog'] = pd.to_numeric(df['cog'], downcast='float')
    df['mmsi'] = pd.to_numeric(df['mmsi'], downcast='integer')
    df['grid_lat'] = (df['latitude'] / GRID_SIZE).astype(int)
    df['grid_lon'] = (df['longitude'] / GRID_SIZE).astype(int)
    df['grid_cell'] = list(zip(df['grid_lat'], df['grid_lon']))
    df['time_bin'] = df['timestamp'].dt.floor(f'{TIME_BIN_MIN}min')
    print(f"Loaded: {original_count} rows (filtered to {filtered_count}) in {time.time() - start_time:.2f}s")
    return df[['timestamp', 'latitude', 'longitude', 'cog', 'mmsi', 'grid_lat', 'grid_lon', 'grid_cell', 'time_bin']]

def process_bin(bin_tuple):
    """Process a single bin to detect vessel conflicts"""
    key, df_bin = bin_tuple
    flagged = set()

    for row in df_bin.itertuples(index=False):
        lat, lon = row.latitude, row.longitude
        lat_min = lat - (PROXIMITY_RADIUS_NM * NM_TO_DEG)
        lat_max = lat + (PROXIMITY_RADIUS_NM * NM_TO_DEG)
        lon_min = lon - (PROXIMITY_RADIUS_NM * NM_TO_DEG)
        lon_max = lon + (PROXIMITY_RADIUS_NM * NM_TO_DEG)
        grid_lat, grid_lon = row.grid_lat, row.grid_lon
        grid_candidates = [(grid_lat + i, grid_lon + j) for i, j in NEIGHBOR_OFFSETS]

        nearby = df_bin[
            df_bin['grid_cell'].isin(grid_candidates) &
            (df_bin['latitude'] >= lat_min) & (df_bin['latitude'] <= lat_max) &
            (df_bin['longitude'] >= lon_min) & (df_bin['longitude'] <= lon_max) &
            (df_bin['mmsi'] != row.mmsi)
        ]

        if nearby.empty:
            continue

        distances = haversine_np(row.latitude, row.longitude, nearby['latitude'].values, nearby['longitude'].values)
        nearby = nearby.copy()
        nearby['distance_nm'] = distances
        nearby = nearby[nearby['distance_nm'] <= PROXIMITY_RADIUS_NM]

        if nearby.empty:
            continue

        # COG difference check
        cog_diff = np.abs(row.cog - nearby['cog'])
        cog_diff = np.where(cog_diff > 180, 360 - cog_diff, cog_diff)
        nearby['cog_diff'] = cog_diff

        # Flags
        nearby['is_directional'] = nearby['cog_diff'] > COG_DIFF_THRESHOLD
        nearby['is_positional'] = (
            np.isclose(nearby['latitude'], row.latitude, atol=0.001) &
            np.isclose(nearby['longitude'], row.longitude, atol=0.001)
        )

        for conflict in nearby.itertuples(index=False):
            pair = tuple(sorted((row.mmsi, conflict.mmsi)))

            # Determine anomaly type
            if conflict.is_directional and conflict.is_positional:
                anomaly_type = 'both'
            elif conflict.is_directional:
                anomaly_type = 'directional_conflict'
            elif conflict.is_positional:
                anomaly_type = 'positional_conflict'
            else:
                continue  # No anomaly, skip

            distance = conflict.distance_nm if conflict.is_directional else 0.0
            cog = float(conflict.cog_diff) if conflict.is_directional else 0.0

            flagged.add((
                row.timestamp,
                pair[0], pair[1],
                distance,
                cog,
                row.latitude, row.longitude,
                anomaly_type
            ))

    return list(flagged)


def process_chunk(df_chunk, cores, output_file=None):
    grouped = df_chunk.groupby(['grid_cell', 'time_bin'])
    bins = [(key, group) for key, group in grouped if len(group) > 1]
    conflicts = []

    if cores == 1:
        for i, bin_data in enumerate(bins):
            if i % 100 == 0:
                progress.update(i, len(bins), "Processing bins")
            result = process_bin(bin_data)
            if result:
                conflicts.extend(result)
    else:
        with mp.Pool(cores) as pool:
            results = pool.map(process_bin, bins)
        for result in results:
            if result:
                conflicts.extend(result)

    if output_file and conflicts:
        df_conflicts = pd.DataFrame(conflicts, columns=[
            'timestamp', 'mmsi', 'neighbor_mmsi',
            'distance_nm', 'cog_diff', 'latitude', 'longitude', 'anomaly_type'
        ])
        df_conflicts.drop_duplicates(subset=['timestamp', 'mmsi', 'neighbor_mmsi'], inplace=True)
        df_conflicts.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

    return len(conflicts)

def run_sequential(df, output_file=None):
    print("Running sequential (1 core, no chunks)...")
    start_time = time.time()
    process = psutil.Process()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss

    total_conflicts = process_chunk(df, cores=1, output_file=output_file)

    duration = time.time() - start_time
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss
    cpu_time = (cpu_end.user - cpu_start.user + cpu_end.system - cpu_start.system)
    mem_used = (mem_end - mem_start) / (1024 ** 2)

    print(f"Sequential: {duration:.2f}s | CPU: {cpu_time:.2f}s | Memory: {mem_used:.2f}MB | Conflicts: {total_conflicts}")

    return {
        'cores': 1, 'chunk_size': 'none',
        'duration': duration, 'cpu_time': cpu_time,
        'memory_mb': mem_used, 'conflicts': total_conflicts
    }

def run_parallel(df, cores, chunk_size, output_file=None):
    print(f"Running {cores} cores with {chunk_size} chunks...")
    start_time = time.time()
    process = psutil.Process()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss

    time_chunks = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq=chunk_size)
    total_conflicts = 0

    for chunk_idx, chunk_start in enumerate(time_chunks):
        chunk_end = chunk_start + pd.Timedelta(chunk_size)
        chunk_df = df[(df['timestamp'] >= chunk_start) & (df['timestamp'] < chunk_end)]
        if chunk_df.empty:
            continue
        progress.update(chunk_idx + 1, len(time_chunks), f"{cores} cores, {chunk_size}")
        conflicts = process_chunk(chunk_df, cores, output_file)
        total_conflicts += conflicts

    progress.update(len(time_chunks), len(time_chunks), f"{cores} cores, {chunk_size}")
    duration = time.time() - start_time
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss
    cpu_time = (cpu_end.user - cpu_start.user + cpu_end.system - cpu_start.system)
    mem_used = (mem_end - mem_start) / (1024 ** 2)

    print(f"{cores} cores, {chunk_size}: {duration:.2f}s | CPU: {cpu_time:.2f}s | Memory: {mem_used:.2f}MB | Conflicts: {total_conflicts}")

    return {
        'cores': cores, 'chunk_size': chunk_size,
        'duration': duration, 'cpu_time': cpu_time,
        'memory_mb': mem_used, 'conflicts': total_conflicts
    }

def run_benchmark(df):
    results = []
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    if os.path.exists(PERF_LOG): os.remove(PERF_LOG)

    baseline = run_sequential(df, OUTPUT_FILE)
    baseline_duration = baseline['duration']
    results.append(baseline)

    for cores in CORE_CONFIGS[1:]:
        if cores > mp.cpu_count():
            print(f"Skipping {cores} cores (system has {mp.cpu_count()} cores)")
            continue
        for chunk_size in TIME_CHUNKS:
            result = run_parallel(df, cores, chunk_size, OUTPUT_FILE)
            result['speedup'] = baseline_duration / result['duration']
            results.append(result)
            print(f"Speedup: {result['speedup']:.2f}x")

    results_df = pd.DataFrame(results)
    results_df.to_csv(PERF_LOG, index=False)
    return results_df

def analyze_conflicts():
    if not os.path.exists(OUTPUT_FILE):
        print("No conflict data to analyze")
        return

    conflicts = pd.read_csv(OUTPUT_FILE)
    conflicts['timestamp'] = pd.to_datetime(conflicts['timestamp'])
    print(f"\nConflict analysis:\nTotal conflicts detected: {len(conflicts)}")

    hourly = conflicts.groupby(conflicts['timestamp'].dt.strftime('%Y-%m-%d %H')).size()
    print("\nHourly distribution:\n", hourly)

    vessel_conflicts = pd.concat([
        conflicts['mmsi'].value_counts().rename('conflict_count'),
        conflicts['neighbor_mmsi'].value_counts().rename('conflict_count')
    ]).groupby(level=0).sum().sort_values(ascending=False)

    print("\nTop 10 vessels by conflict count:\n", vessel_conflicts.head(10))

    hourly.to_csv('conflict_summary_by_hour.csv', header=['count'])
    vessel_conflicts.head(20).to_csv('top_conflict_vessels.csv', header=['conflict_count'])

    very_close = conflicts[conflicts['distance_nm'] < 0.1]
    if len(very_close) > 0:
        print("\nPotential GPS conflicts (vessels within 0.1 NM):\n", very_close[['mmsi', 'neighbor_mmsi', 'distance_nm', 'timestamp']].head(10))
        very_close.to_csv('potential_gps_conflicts.csv', index=False)

def main():
    mp.freeze_support()
    df = load_data()
    run_benchmark(df)
    analyze_conflicts()
    print("\nBenchmark and analysis complete")

if __name__ == '__main__':
    main()
