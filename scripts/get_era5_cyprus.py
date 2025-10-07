import cdsapi, os
os.makedirs("data", exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["geopotential"],   # one small variable
        "year": "2020",
        "month": "07",
        "day": "01",
        "time": ["00:00","06:00","12:00","18:00"],  # 4 timesteps only
        "area": [36.5, 30.0, 33.0, 36.0],           # N, W, S, E (Cyprus region)
        "grid": [0.5, 0.5],                         # coarse grid → tiny file
        "format": "netcdf",
    },
    "data/era5_cyprus_2020-07-01.nc",
)

print("Saved: data/era5_cyprus_2020-07-01.nc")
