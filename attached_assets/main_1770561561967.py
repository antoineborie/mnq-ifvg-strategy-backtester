import pandas as pd
import glob
import os
import datetime

from data_loader import load_period
from data_manager import DataMaster

def run_test():
    
    start_y, start_m = 2025, 12
    end_y, end_m = 2026, 1 

    raw_data = load_period(start_year = start_y, start_month = start_m, end_year = end_y, end_month = end_m)

    if raw_data is None:
        return
    
    dm = DataMaster(raw_data)

    test_day = [datetime.date(2026, 1, i) for i in range (5, 10)]

    for day in test_day:
        history, session, symbol = dm.get_market_context(day)

        if history is not None and not history.empty:
            levels = dm.calculate_ict_levels(history, day)

            print(f"\n{'='*40}")
            print(f" DATE : {day} | CONTRAT : {symbol}")
            print(f"{'='*40}")
            print(f" MIDNIGHT OPEN : {levels['mid']:.2f}")
            print(f" PREV DAY HIGH : {levels['pdh']:.2f}")
            print(f" PREV DAY LOW  : {levels['pdl']:.2f}")
            print(f" MACRO 15D HIGH: {levels['h15']:.2f}")
            print(f" MACRO 15D LOW : {levels['l15']:.2f}")
            print(f"{'='*40}")
        else:
            print(f"No data for the {day} (week end or day off)")

def audit_my_data(path="data/"):
    files = sorted(glob.glob(os.path.join(path, "*.pkl")))
    print(f"Analyse de {len(files)} fichiers...\n")
    
    summary = []
    for f in files:
        df = pd.read_pickle(f)
        symbols = df['symbol'].unique()
        start = df.index.min()
        end = df.index.max()
        count = len(df)
        
        summary.append({
            "Fichier": os.path.basename(f),
            "Symboles": symbols,
            "Début": start,
            "Fin": end,
            "Lignes": count
        })
    
    audit_df = pd.DataFrame(summary)
    print(audit_df.to_string())
    return audit_df

if __name__ == "__main__":
    #audit_my_data()
    run_test()