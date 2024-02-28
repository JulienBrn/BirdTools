from pvrecorder import PvRecorder
import wave
import struct
import pyaudio
import numpy as np, pathlib, logging
from circular_buffer_numpy.circular_buffer import CircularBuffer
from datetime import date, datetime, timedelta
from scipy import signal
from beautifullogger import setup, setDisplayLevel
import json, pandas as pd
import yaml

logger = logging.getLogger(__name__)
setup()

try:
  with open("soundrecorder.yaml", "r") as config:
    conf = yaml.safe_load(config)
  logging_level = logging.getLevelNamesMapping()[conf["display"]["logging_level"]]

  rate = conf["device"]["rate"]
  nchannels = conf["device"]["nchannels"]
  device_index = conf["device"]["index"]
  selected_channels = conf["device"]["selected_channels"]

  buffer_size = int(conf["record_boundaries"]["buffer_size_seconds"]*rate)
  backtrack_size = int(conf["record_boundaries"]["left_shoulder_seconds"]*rate)
  fronttrack_size = int(conf["record_boundaries"]["right_shoulder_seconds"]*rate)
  thresholds = {c: conf["record_boundaries"]["threshold"] for c in selected_channels}

  nb_song_threshold = int(conf["classification"]["min_song_duration_seconds"]*rate)
  song_filter_sos = signal.butter(3, [conf["classification"]["song_low_filter"], conf["classification"]["song_high_filter"]], 'bandpass', fs=rate, output='sos')
  song_threshold_ratio = conf["classification"]["song_threshold_ratio"]

  for str_channel, new_val_dict in conf["channel_specific_override"].items():
    chan = int(str_channel)
    if chan in selected_channels:
      for param, new_val in new_val_dict.items():
        match param:
          case "threshold":
            thresholds[chan] = new_val
          case _:
            raise Exception(f"Unknown override parameter {param}")
      
  logger.info("Configuration initialized from soundrecorder.yaml file")
except Exception as e:
  logger.warning(f"Problem while reading configuration from file soundrecorder.yaml. Continuing with configuration set up within code. Error is\n{e}")
  logging_level=20

  rate = 44100
  nchannels = 32
  device_index=14
  selected_channels=[5, 6]

  buffer_size = int(0.1*rate)
  backtrack_size = 2*rate
  fronttrack_size = 2*rate
  thresholds={c: 500 for c in selected_channels}

  nb_song_threshold = int(0.5*rate)
  song_filter_sos = signal.butter(3, [1000, 5000], 'bandpass', fs=rate, output='sos')
  song_threshold_ratio = 0.02

setDisplayLevel(logging_level)
logger.info("Initializing pyaudio. This may print much information")
paudio = pyaudio.PyAudio()
logger.info(f"Retrieving device with index {device_index}")
r = paudio.get_default_input_device_info()
logger.debug(f"Used device: {r}")
stream = paudio.open(format=pyaudio.paInt16, output_device_index=14, 
                channels=nchannels, rate=rate, input=True, frames_per_buffer=buffer_size,
)

nb_since_last_noise = {c:np.inf for c in selected_channels}
write_data = {c:np.array([], dtype='int16') for c in selected_channels}
backtrack_data = {c:CircularBuffer(shape=backtrack_size, dtype='int16') for c in selected_channels}
nb_song = {c:0 for c in selected_channels}


def get_classifications(buffer):
  df = pd.DataFrame()
  df["Channel"] = selected_channels
  df["Threshold"] = [thresholds[c] for c in selected_channels]
  df["Max_buffer_val"] = [np.abs(buffer[c]).max() for c in selected_channels]
  df["Mean_buffer_val"] = [np.abs(buffer[c]).mean() for c in selected_channels]
  df["Threshold_passed"] = df["Mean_buffer_val"] > df["Threshold"]
  df = df.set_index("Channel")
  df["Mean_songfiltered_val"] = [np.abs(signal.sosfilt(song_filter_sos, buffer[c])).mean() if b else None  for c, b in df["Threshold_passed"].items()]
  df["Ratio"] = df["Mean_songfiltered_val"]/df["Mean_buffer_val"] 
  df["Classification"] = np.where(df["Ratio"].isna(), "SILENCE", np.where(df["Ratio"]> song_threshold_ratio, "SONG", "NOISE"))
  logger.debug(f"Classification information is\n{df}")
  return df["Classification"].to_dict()
  

try:
  logger.info(f"Recording on channels {selected_channels}. Press CTRL+C to stop.")
  while True:
    buffer = stream.read(buffer_size)
    buffer = np.frombuffer(buffer, dtype='int16').reshape((-1, nchannels)).T
    classifications = get_classifications(buffer)
    for channel in selected_channels:
      match classifications[channel] != "SILENCE", bool((nb_since_last_noise[channel] > fronttrack_size)):
        case True, False:
          # print("adding current")
          write_data[channel] = np.concatenate([write_data[channel], buffer[channel]])
        case True, True:
          # print(f"adding backtrack + current")
          write_data[channel] = np.concatenate([write_data[channel], backtrack_data[channel].get_data(), buffer[channel]])
        case (False, True):
          #Time to write
          if write_data[channel].size > 0:
            classification = "Song" if nb_song[channel] > nb_song_threshold else "Noise"
            start_recording_time = datetime.now() - timedelta(seconds=write_data[channel].size/rate)
            filename = f"./BirdRecordings/Channel{channel}/{classification}/{start_recording_time.strftime('%Y-%m-%d')}/{start_recording_time.strftime('%H%M%S.%f')}.wav"
            filename = pathlib.Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing to {filename}")
            wf = wave.open(str(filename), 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(paudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(write_data[channel].tobytes())
            wf.close()
            write_data[channel] = np.array([], dtype='int16')
            nb_song[channel] = 0
        case False, False:
          # print("Adding current silence to data")
          write_data[channel] = np.concatenate([write_data[channel], buffer[channel]])

      nb_since_last_noise[channel] = 0 if classifications[channel] != "SILENCE" else nb_since_last_noise[channel]+buffer_size
      backtrack_data[channel].append(buffer[channel])
      if classifications[channel] == "SONG":
          nb_song[channel]+=buffer_size


except KeyboardInterrupt:
  logger.info("Stopping recording. Any not yet written data has been discarded")
