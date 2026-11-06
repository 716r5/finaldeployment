"""Exact input contract copied from the existing calibration preprocessing."""
USABLE_BANDS = [
    3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
    20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
    37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,63,64,65,66,67,68,69,70,71,72,
    73,74,75,76,77,78,79,85,86,87,88,89,90,91,92,93,94,
    95,96,97,98,99,100,101,102,103,121,122,123,124,125,126,
    127,128,129,130,131,132,133,134,135,136,137,138,139,140,
    141,142,143,144,145,146,181,182,183,184,185,186,187,188,
    189,190,191,192,193,194,195,196,197,198,
]
PATCH_SIZE = 11
HALF = PATCH_SIZE//2
CONTRACT = 'hsi-138-scene-percentile-1-99-v1'

def normalise_band(band, np):
    """Same float32 conversion and percentile expression as data_utils.py."""
    band = np.asarray(band, dtype=np.float32)
    if not np.isfinite(band).all():
        raise ValueError('Scene contains nonfinite values in a selected spectral band')
    low, high = np.percentile(band,(1,99))
    return np.clip((band-low)/(high-low+1e-8),0.0,1.0), [float(low),float(high)]
