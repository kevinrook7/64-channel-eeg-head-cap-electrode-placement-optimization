import mne
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_subject


def prepare_epochs(subject=1):
    """PhysioNet eegmmidb: band-pass filter, drop bad channels, cut left/right epochs."""
    raw, events, event_id = load_subject(subject)
    return epochs_from_raw(raw, events)


FLAT_STD = 1e-6   # volts: below 1 uV of 8-30 Hz activity means no real signal
NOISY_Z = 5.0     # robust z above this means unusually loud (likely a bad contact)


def epochs_from_raw(raw, events):
    """Shared by every dataset: montage, 8-30 Hz filter, bad-channel check, epochs.

    `events` must mark cue onset with code 2 (left) / 3 (right).
    """
    raw_filtered, bad_channels = filter_and_flag(raw)
    return make_epochs(raw_filtered, events, bad_channels)


def filter_and_flag(raw):
    """Attach the montage, band-pass 8-30 Hz, and flag flat or noisy EEG channels.

    One-sided on purpose: a channel that is merely quieter than the rest is kept.
    Low amplitude can be real signal (e.g. BNCI's C5 sits next to the reference
    and still correlates ~0.9 with C3); a loud channel usually means a bad contact.
    Uses no labels, so it can't leak class information.
    """
    montage = mne.channels.make_standard_montage("colin27_1005")
    raw.set_montage(montage)
    # raw.plot_sensors(show_names=True)

    raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0)

    # Only EEG channels take part in the check (EOG / stim channels would skew it).
    eeg_names = [raw_filtered.ch_names[i] for i in mne.pick_types(raw_filtered.info, eeg=True)]
    data = raw_filtered.get_data(picks=eeg_names)
    channel_std = data.std(axis=1)

    median = np.median(channel_std)
    mad = np.median(np.abs(channel_std - median))
    robust_z = (channel_std - median) / (1.4826 * mad)

    bad_channels = []
    print("Median channel std:", round(median * 1e6, 1), "uV")
    for name, std, z in zip(eeg_names, channel_std, robust_z):
        if std < FLAT_STD:
            print(f"FLAT    {name}: std = {std * 1e6:.1f} uV")
            bad_channels.append(name)
        elif z > NOISY_Z:
            print(f"NOISY   {name}: std = {std * 1e6:.1f} uV, robust z = {z:.1f}")
            bad_channels.append(name)
    return raw_filtered, bad_channels


def make_epochs(raw_filtered, events, bad_channels):
    """Cut left/right epochs (-0.5 to 4.0 s around the cue), dropping `bad_channels`."""
    raw_filtered.info["bads"] = list(bad_channels)

    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id={"left_fist": 2, "right_fist": 3},
        tmin=-0.5,
        tmax=4.0,
        baseline=None,
        preload=True,
    )
    epochs = epochs.pick("eeg", exclude="bads")
    return epochs


if __name__ == "__main__":
    epochs = prepare_epochs(1)
    print(epochs)
    print("Data shape:", epochs.get_data().shape)

    # raw.compute_psd(fmax=80).plot()
    # raw_filtered.compute_psd(fmax=80).plot()
    # raw_filtered.plot(picks=["T10", "T8", "TP8", "FT8", "C4"], scalings=dict(eeg=50e-6))

    plt.show()
