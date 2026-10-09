"""Phase 1: Simulated 4-channel EEG headband broadcasting over LSL (pylsl)."""

# TODO: Learn what an LSL stream is and create a StreamInfo / StreamOutlet.
# TODO: Stream a basic sine wave on 4 channels.

import math
import time
import random

from pylsl import StreamInfo, StreamOutlet

SAMPLE_RATE = 256
CHANNELS = 4
FREQUENCY = 10
AMPLITUDE = 20.0
CHANNEL_AMPLITUDES = [1.0, 0.8, 0.85, 1.1]   # multiplier on AMPLITUDE per channel
CHANNEL_PHASES = [0.0, 0.3, 0.5, 0.9]        # shift in radians per channel
NOISE_LEVEL = 3.0                            # microvolts of random noise

info = StreamInfo(
    name="fakeEEG",
    type="EEG",
    channel_count=CHANNELS,
    nominal_srate=SAMPLE_RATE,
    channel_format="float32",
    source_id="staylocked-sim-001",
)

outlet = StreamOutlet(info)


start = time.perf_counter()
n=0

while True:
    t=n/SAMPLE_RATE
    sample=[]
    for ch in range(CHANNELS):
        wave = AMPLITUDE*CHANNEL_AMPLITUDES[ch]*math.sin(2*math.pi*FREQUENCY*t + CHANNEL_PHASES[ch])
        noise = random.gauss(0, NOISE_LEVEL)
        sample.append(wave + noise)
    outlet.push_sample(sample)

    n+=1
    next_time = start + n/SAMPLE_RATE
    time.sleep(max(0.0,next_time - time.perf_counter()))