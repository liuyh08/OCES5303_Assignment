from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHLA_DIR = DATA_DIR / "MODIS_Chlor_8D"
CMIP6_DIR = DATA_DIR / "cmip6"
CMIP6_DIR_A = CMIP6_DIR / "1990-2009"
CMIP6_DIR_B = CMIP6_DIR / "2010-2014"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in [PROCESSED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Region: South China Sea
REGION = dict(lat_min=0.0, lat_max=25.0, lon_min=100.0, lon_max=125.0)

# Time range and train/test split
TIME_START = "2002-08"
TIME_END = "2014-12"
TRAIN_END = "2011-12"
TEST_START = "2012-01"

# File name templates
CHLA_FILE_TPL = "AQUA_MODIS.{s}_{e}.L3m.MO.CHL.chlor_a.9km.nc"
CMIP6_VARS = ["no3", "po4", "si", "dfe", "o2", "mlotst", "tos"]
CMIP6_SURFACE_VARS = ["no3", "po4", "si", "dfe", "o2"]
CMIP6_2D_VARS = ["mlotst", "tos"]

# CMIP6 file template
CMIP6_FILE_TPL = "{var}_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_{period}.nc"
CMIP6_PERIOD_A = "199001-200912"
CMIP6_PERIOD_B = "201001-201412"

# Quality control
QC = dict(chla_min=0.001, chla_max=100.0)

# Downsampling
DOWNSAMPLE_DEG = 1.0

SEED = 42
TORCH_SEED = 1234
