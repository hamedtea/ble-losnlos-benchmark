from pathlib import Path
import pandas as pd

def read_room_office():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    df_room = pd.read_pickle(data_dir / "room" / "df_3k_7k_room.pkl")
    df_office = pd.read_pickle(data_dir / "office" / "df_3k_7k_office.pkl")
    return df_room, df_office


