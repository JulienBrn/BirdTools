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
pattern = "074051.*.wav"

data = toolbox.read_folder_as_database(folder, [], pattern)
if len(data.index) ==0:
    logger.warning("No files found")
    exit()
print(data)
data["fs"] = data["path"].apply(lambda x: scipy.io.wavfile.read(x)[0])
data["arr"] = data["path"].apply(lambda x: scipy.io.wavfile.read(x)[1])
print(data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def compute_spectrogram(song: np.ndarray, fs, NFFT=512):

    fig, ax = plt.subplots(figsize=(10, 8))

    specgram_args = {
        "NFFT": NFFT,
        # "noverlap": NFFT * 0.90,
        "scale_by_freq": True,
        "mode": "psd",
        "pad_to": 915,
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "scale": "dB",
        "detrend": "mean",
    }
    x = song.flatten()
    print(x.shape)
    spec, freq, t, im = plt.specgram(x, Fs=fs, **specgram_args)
    
    x_t = np.linspace(0, len(song), len(t))

    ax.set_ylabel("Frequency (Hz)")
    ax.locator_params(axis="x", nbins=8)
    ticks = ax.get_xticks().tolist()
    new_ticks = [
        "%.2f" % item
        for item in np.linspace(0, len(song), len(ticks))
    ]
    ax.set_xticklabels(new_ticks)
    ax.set_xlabel("Time (s)")

    plt.show(block=True)

    # plt.close()

    return np.nan

data["spectrogram"] = data.progress_apply(lambda row: compute_spectrogram(row["arr"], row["fs"]), axis=1, result_type="reduce")

#data = data.groupby("path").apply(lambda grp: grp["spectrogram"].iat[0].to_dataframe(name="spectrogram")).reset_index()
#data["print_path"] = data["path"].apply(lambda x:pathlib.Path(x).relative_to(folder))
#data["figure"] = (data.groupby("path").ngroup()/4).astype(int)
#toolbox.FigurePlot(data= data, figures="figure", col="print_path", row="channel", subplot_title="{print_path}", margin_titles=True).pcolormesh(x="t", y="f", value="spectrogram", ylabels=True, cmap="viridis").maximize()
# subplot_title="{channel}{print_path}"
# map(sns.lineplot, )

# compute_spectrogram(song, fs, NFFT=512)
