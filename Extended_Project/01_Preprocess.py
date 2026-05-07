import xarray as xr
import numpy as np
import pandas as pd
import calendar
import time as _time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import (CHLA_DIR, CMIP6_DIR_A, CMIP6_DIR_B, PROCESSED_DIR,REGION,
                    TIME_START, TIME_END, TRAIN_END, TEST_START, CHLA_FILE_TPL, CMIP6_FILE_TPL,
                    CMIP6_PERIOD_A, CMIP6_PERIOD_B,CMIP6_VARS, CMIP6_SURFACE_VARS, QC)


def month_dates(start, end):
    return [(d.year, d.month) for d in pd.date_range(start, end, freq="MS")]


def date_strings(year, month):
    s = f"{year}{month:02d}01"
    e = f"{year}{month:02d}{calendar.monthrange(year, month)[1]:02d}"
    return s, e


def read_cmip6_all():
    var_dict = {}
    for var in CMIP6_VARS:
        is_3d = var in CMIP6_SURFACE_VARS

        fp_a = CMIP6_DIR_A / CMIP6_FILE_TPL.format(var=var, period=CMIP6_PERIOD_A)
        fp_b = CMIP6_DIR_B / CMIP6_FILE_TPL.format(var=var, period=CMIP6_PERIOD_B)
        ds_a = xr.open_dataset(fp_a)
        ds_b = xr.open_dataset(fp_b)
        ds = xr.concat([ds_a, ds_b], dim="time")
        ds_a.close()
        ds_b.close()

        time_vals = ds.time.values
        pd_times = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 15) for t in time_vals])
        ds = ds.assign_coords(time=pd_times)
        ds = ds.sel(time=slice(TIME_START, TIME_END))

        lon_vals = ds.lon.values.copy()
        lon_vals = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
        ds = ds.assign_coords(lon=lon_vals)
        ds = ds.sortby("lon")

        if is_3d and "lev" in ds.dims:
            ds = ds.isel(lev=0).drop_vars(["lev", "lev_bnds"], errors="ignore")

        drop_vars = [v for v in ds.data_vars if "bnds" in v]
        ds = ds.drop_vars(drop_vars, errors="ignore")

        r = REGION
        ds = ds.sel(lat=slice(r["lat_min"], r["lat_max"]),
                    lon=slice(r["lon_min"], r["lon_max"]),)

        var_dict[var] = ds[var]
        ds.close()

    cmip6_ds = xr.Dataset(var_dict)
    print(f"CMIP6 dataset: {dict(cmip6_ds.sizes)}")
    return cmip6_ds


def read_modis_chla(cmip6_lat, cmip6_lon):
    dates = month_dates(TIME_START, TIME_END)
    chla_list = []

    for y, m in tqdm(dates, desc="MODIS chl-a"):
        s, e = date_strings(y, m)
        fp = CHLA_DIR / CHLA_FILE_TPL.format(s=s, e=e)

        ds = xr.open_dataset(fp)
        r = REGION
        lat_mask = (ds.lat >= r["lat_min"]) & (ds.lat <= r["lat_max"])
        lon_mask = (ds.lon >= r["lon_min"]) & (ds.lon <= r["lon_max"])
        ds_c = ds.where(lat_mask & lon_mask, drop=True)

        da = ds_c["chlor_a"].astype("float32").copy()
        da = da.expand_dims(time=[pd.Timestamp(y, m, 15)])
        chla_list.append(da)
        ds.close()

    chla_all = xr.concat(chla_list, dim="time").sortby("time")
    chla_1deg = chla_all.interp(lat=cmip6_lat, lon=cmip6_lon, method="linear")
    print(f"MODIS chl-a: {dict(chla_1deg.sizes)}")
    return chla_1deg


def main():
    t0 = _time.time()
    print("Preprocess")
    print(f"Region: {REGION['lat_min']}-{REGION['lat_max']}N, "
          f"{REGION['lon_min']}-{REGION['lon_max']}E")
    print(f"Time: {TIME_START} ~ {TIME_END}")
    print(f"Train: {TIME_START} ~ {TRAIN_END}  Test: {TEST_START} ~ {TIME_END}")

    print("\nReading CMIP6 variables")
    cmip6_ds = read_cmip6_all()

    print("\nReading MODIS chl-a")
    chla_da = read_modis_chla(cmip6_ds.lat.values, cmip6_ds.lon.values)

    print("\nMerging and quality control")
    ds = cmip6_ds.copy()
    ds["chlor_a"] = chla_da
    ds["log_chla"] = np.log10(ds["chlor_a"].where(ds["chlor_a"] > 0))
    ds["log_chla"].attrs = {"long_name": "log10(chl-a)", "units": "log10(mg/m3)"}

    v = ds["chlor_a"].values
    bad = ((v < QC["chla_min"]) | (v > QC["chla_max"])) & ~np.isnan(v)
    print(f"chlor_a: {bad.sum()} out of range")
    ds["chlor_a"] = ds["chlor_a"].where((ds["chlor_a"] >= QC["chla_min"]) &
                                        (ds["chlor_a"] <= QC["chla_max"]))

    print(f"\nSaving")
    out_nc = PROCESSED_DIR / "SCS_CMIP6_MODIS_1deg.nc"
    ds.to_netcdf(out_nc)
    print(f"NetCDF: {out_nc.name}  ({out_nc.stat().st_size / 1e6:.1f} MB)")

    cmip6_cols = CMIP6_VARS
    rows = []
    for i, t in enumerate(tqdm(ds.time.values, desc="  flattening")):
        ts = pd.Timestamp(t)
        chla_2d = ds["chlor_a"].isel(time=i).values
        logc_2d = ds["log_chla"].isel(time=i).values
        lats = np.broadcast_to(ds["lat"].values[:, None], chla_2d.shape)
        lons = np.broadcast_to(ds["lon"].values[None, :], chla_2d.shape)

        rec = {
            "time": ts.strftime("%Y-%m"),
            "lat": lats.ravel(),
            "lon": lons.ravel(),
            "chlor_a": chla_2d.ravel(),
            "log_chla": logc_2d.ravel(),
        }
        for col in cmip6_cols:
            rec[col] = ds[col].isel(time=i).values.ravel()
        rows.append(pd.DataFrame(rec))

    df = pd.concat(rows, ignore_index=True)

    df["split"] = "train"
    df.loc[pd.to_datetime(df["time"]) >= pd.Timestamp(TEST_START), "split"] = "test"

    before = len(df)
    df = df.dropna(subset=["chlor_a"] + CMIP6_VARS).reset_index(drop=True)

    out_pq = PROCESSED_DIR / "SCS_CMIP6_MODIS_flat.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"Parquet: {out_pq.name}  ({out_pq.stat().st_size / 1e6:.1f} MB)")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nVariable Statistics")
    print(df[["chlor_a", "log_chla"] + cmip6_cols].describe().round(4).to_string())
    print(f"\nTrain: {(df['split'] == 'train').sum():,} rows")
    print(f"Test:  {(df['split'] == 'test').sum():,} rows")
    ds.close()


if __name__ == "__main__":
    main()





