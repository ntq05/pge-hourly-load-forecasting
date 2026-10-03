class Settings:
    ANCHOR_YEARS = {1: 2020, 2: 2021, 3: 2022}
    TEMP_COLS = ['Site-1 Temp', 'Site-2 Temp', 'Site-3 Temp', 'Site-4 Temp', 'Site-5 Temp']
    GHI_COLS = ['Site-1 GHI', 'Site-2 GHI', 'Site-3 GHI', 'Site-4 GHI', 'Site-5 GHI']

    WEATHER = ["lag_1_temp", "lag_1_ghi_1", "lag_1_ghi_2", "delta_1_temp", "delta_1_ghi_1", "delta_1_ghi_2", "delta_24_temp", "delta_24_ghi_1", "delta_24_ghi_2"]

    T_BASE = 20.0

settings = Settings()