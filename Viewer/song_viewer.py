import pandas as pd, numpy as np, functools, scipy, xarray as xr
import toolbox, tqdm, pathlib
from typing import List, Tuple, Dict, Any, Literal
import matplotlib.pyplot as plt
import matplotlib as mpl, seaborn as sns
from scipy.stats import gaussian_kde
import math, itertools, pickle, logging, beautifullogger
import scipy.signal, scipy.io.wavfile
from matplotlib.backends.backend_pdf import PdfPages

xr.set_options(use_flox=True, display_expand_coords=True, display_max_rows=100, display_expand_data_vars=True, display_width=150)
logger = logging.getLogger(__name__)
beautifullogger.setup(displayLevel=logging.INFO)
logging.getLogger("flox").setLevel(logging.WARNING)
tqdm.tqdm.pandas(desc="Computing")


folder = pathlib.Path("./testdata")
pattern = "*.wav"
n_cols =1

def compute_spectrogram(a, fs):
    NFFT=512

    if len(a.shape)==1:
        a = np.reshape(a, (-1, 1))

    mpl_specgram_window = plt.mlab.window_hanning(np.ones(NFFT))
    f, t, sxx = scipy.signal.spectrogram(a, fs, axis=0, nfft=NFFT, detrend="constant", scaling = "density", window=mpl_specgram_window)
    r = xr.DataArray(data=sxx, dims=["f", "channel", "t"], coords=dict(f=f, t=t))
    r = 10*np.log10(r)
    return r.sel(f=slice(0, 20000))


data = toolbox.read_folder_as_database(folder, [], pattern)
if len(data.index) ==0:
    logger.warning("No files found")
    exit()
print(data)
data["fs"] = data["path"].apply(lambda x: scipy.io.wavfile.read(x)[0])
data["arr"] = data["path"].apply(lambda x: scipy.io.wavfile.read(x)[1])
data["print_path"] = data["path"].apply(lambda x:pathlib.Path(x).relative_to(folder))
data["figure"] = (data.groupby("path").ngroup()/n_cols).astype(int)
for _, df in tqdm.tqdm(data.groupby("figure")):
    df = df.copy()
    df["spectrogram"] = df.apply(lambda row: compute_spectrogram(row["arr"], row["fs"]), axis=1, result_type="reduce")
    df = df.groupby(["path", "print_path"]).apply(lambda grp: grp["spectrogram"].iat[0].to_dataframe(name="spectrogram")).reset_index()
    toolbox.FigurePlot(
        data= df, col="print_path", row="channel", subplot_title="{print_path}", margin_titles=True, sharex=False
    ).pcolormesh(x="t", y="f", value="spectrogram", ylabels=True, cmap="viridis").maximize()
    plt.show()





