import pandas as pd 
import glob 
import os 

def load_period(data_path="data/", start_year=2025, start_month=1, end_year=2026, end_month=1):
    all_files = sorted(glob.glob(os.path.join(data_path, "MNQ_*.pkl")))
    files_to_load = []  
    
    print(f"Target: {start_month}/{start_year} to {end_month}/{end_year}")

    for f in all_files:
        parts = os.path.basename(f).split('_')
        f_year = int(parts[1])
        f_month = int(parts[2].split('.')[0])

        start_val = start_year * 100 + start_month
        end_val = end_year * 100 + end_month
        current_val = f_year * 100 + f_month

        if start_val <= current_val <= end_val:
            files_to_load.append(f)

    if not files_to_load:
        print("❌ Aucun fichier trouvé pour cette période.")
        return None
    
    df_list = []
    for f in files_to_load:
        print(f"📥 Loading: {os.path.basename(f)}")
        df_list.append(pd.read_pickle(f))

    full_df = pd.concat(df_list).sort_index()
    print(f"✅ Terminé : {len(full_df)} lignes chargées.")
    return full_df

