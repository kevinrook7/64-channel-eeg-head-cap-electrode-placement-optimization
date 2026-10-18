import mne
from mne.datasets import eegbci

def labels_for_run(run):
    if run in (3, 4, 7, 8, 11, 12):
        return {"T0": "rest", "T1": "left_fist", "T2": "right_fist"}
    if run in (5, 6, 9, 10, 13, 14):
        return {"T0": "rest", "T1": "both_fists", "T2": "both_feet"}
    raise ValueError(f"Run {run} has no movement events")

SUBJECT = 1
RUNS = [4,8,12]
DATA_PATH = "C:/Users/LENOVO/Desktop/Neurotech/StayLocked/data"

EVENT_ID = {"rest": 1, "left_fist": 2, "right_fist": 3, "both_fists": 4, "both_feet": 5}


def load_subject(subject, runs=RUNS):
    raws = []
    for run in runs:
        file = eegbci.load_data(subject, [run], path=DATA_PATH, update_path=False)[0]
        raw = mne.io.read_raw_edf(file, preload=True)
        eegbci.standardize(raw)
        raw.annotations.rename(labels_for_run(run))
        raws.append(raw)

    raw = mne.concatenate_raws(raws)
    events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)
    return raw, events, event_id


if __name__ == "__main__":
    raw, events, event_id = load_subject(SUBJECT)
    print("Duration:", raw.times[-1], "seconds")
    for name, code in event_id.items():
        print(name, "trials:", (events[:, 2] == code).sum())