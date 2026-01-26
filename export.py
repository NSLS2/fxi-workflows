# import databroker
import datetime
# import h5py
# import numpy as np
import os
# import pandas as pd
import re
import httpx

from pathlib import Path
# from PIL import Image
from prefect import task, flow, get_run_logger

from prefect.blocks.system import Secret
from tiled.client import from_profile

api_key = Secret.load("tiled-fxi-api-key", _sync=True).get()
tiled_client = from_profile("nsls2", api_key=api_key)["fxi"]
tiled_client_fxi = tiled_client["raw"]
tiled_client_processed = tiled_client["sandbox"]



@task
def run_export_fxi(uid):
    start_doc = tiled_client_fxi[uid].start
    scan_id = start_doc["scan_id"]
    scan_type = start_doc["plan_name"]
    logger = get_run_logger()
    logger.info(f"Scan ID: {scan_id}")
    logger.info(f"Scan Type: {scan_type}")
    export_scan(uid, filepath=lookup_directory(start_doc) / "exports" / scan_id)
    #logger.info(f"Directory: {lookup_directory(start_doc)}")


@flow
def export(uid):
    run_export_fxi(uid)


def lookup_directory(start_doc):
    """
    Return the path for the proposal directory.

    PASS gives us a *list* of cycles, and we have created a proposal directory under each cycle.
    """
    DATA_SESSION_PATTERN = re.compile("[GUPCpass]*-([0-9]+)")
    client = httpx.Client(base_url="https://api.nsls2.bnl.gov")
    data_session = start_doc[
        "data_session"
    ]  # works on old-style Header or new-style BlueskyRun

    try:
        digits = int(DATA_SESSION_PATTERN.match(data_session).group(1))
    except AttributeError:
        raise AttributeError(f"incorrect data_session: {data_session}")

    response = client.get(f"/v1/proposal/{digits}/directories")
    response.raise_for_status()

    paths = [path_info["path"] for path_info in response.json()["directories"]]

    # Filter out paths from other beamlines.
    paths = [path for path in paths if "fxi-new" == path.lower().split("/")[3]]
    
    # Filter out paths from other cycles and paths for commissioning.
    paths = [
        path
        for path in paths
        if path.lower().split("/")[5] == "commissioning"
        or path.lower().split("/")[5] == start_doc["cycle"]
    ]

    # There should be only one path remaining after these filters.
    # Convert it to a pathlib.Path.
    return Path(paths[0])

def is_legacy(run):
    """
    Check if a run.start document is from a legacy scan.
    """
    t_new = datetime.datetime(2021, 5, 1)
    t = run.start["time"] - 3600 * 60 * 4  # there are 4hour offset
    t = datetime.datetime.utcfromtimestamp(t)
    scan_type = run.start["plan_name"]
    legacy_set = {"tomo_scan", "fly_scan", "xanes_scan", "xanes_scan2"}
    return t < t_new and scan_type in legacy_set


def get_fly_scan_angle(run):
    timestamp_tomo = list(run["primary"]["data"]["Andor_image"])[0]
    assert "zps_pi_r_monitor" in run
    timestamp_mot = run["zps_pi_r_monitor"].read().coords["time"].values
    img_ini_timestamp = timestamp_tomo[0]
    mot_ini_timestamp = timestamp_mot[
        1
    ]  # timestamp_mot[1] is the time when taking dark image

    tomo_time = timestamp_tomo - img_ini_timestamp
    mot_time = timestamp_mot - mot_ini_timestamp

    mot_pos = np.array(run["zps_pi_r_monitor"].read()["zps_pi_r"])
    mot_pos_interp = np.interp(tomo_time, mot_time, mot_pos)

    img_angle = mot_pos_interp
    return img_angle


def convert_AD_timestamps(ts):
    EPICS_EPOCH = datetime(1990, 1, 1, 0, 0)
    return pd.to_datetime(ts, unit="s", origin=EPICS_EPOCH, utc=True).dt.tz_convert(
        "US/Eastern"
    )


def get_img(run, det="Andor", sli=[]):
    "Take in a Header and return a numpy array of detA1 image(s)."
    det_name = f"{det}_image"
    if len(sli) == 2:
        img = np.array(list(run["primary"]["data"][det_name])[sli[0] : sli[1]])
    else:
        img = np.array(list(run["primary"]["data"][det_name]))
    return np.squeeze(img)


def bin_ndarray(ndarray, new_shape=None, operation="mean"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if new_shape == None:
        s = np.array(ndarray.shape)
        s1 = np.int32(s / 2)
        new_shape = tuple(s1)
    operation = operation.lower()
    if not operation in ["sum", "mean"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def export_scan(uid, binning=4, filepath=""):
    # raster_2d_2 scan calls export_raster_2D function even though export_raster_2D_2 function exists.
    # Legacy functions do not exist yet.
    # tiled_client = databroker.from_profile("nsls2", username=None)["fxi"]["raw"]
    run = tiled_client_fxi[uid]
    scan_type = run.start["plan_name"]
    export_function = (
        f"export_{scan_type}_legacy" if is_legacy(run) else f"export_{scan_type}"
    )
    if export_function not in globals().keys():
        print("GLOBAL", globals().keys())
        raise RuntimeError(
            f"Export function {export_function} for scan type {scan_type} not found."
        )
    #globals()[export_function](run, binning=binning, filepath=filepath)
    logger.info(f"File path : {export_function} and Filepath: {filepath}")


def export_tomo_scan(run, filepath="", **kwargs):
    scan_type = "tomo_scan"
    scan_id = run.start["scan_id"]
    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    angle_i = run.start["plan_args"]["start"]
    angle_e = run.start["plan_args"]["stop"]
    angle_n = run.start["plan_args"]["num"]
    img = np.array(list(run["primary"]["data"]["Andor_image"]))
    img = np.array(list(run["primary"]["data"]["Andor_image"]))
    img_tomo = np.median(img, axis=1)
    img = np.array(list(run["dark"]["data"]["Andor_image"]))[0]
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))[0]
    img = np.array(list(run["flat"]["data"]["Andor_image"]))[0]
    img_bkg = np.array(list(run["flat"]["data"]["Andor_image"]))[0]

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)
    img_angle = np.linspace(angle_i, angle_e, angle_n)

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=img_bkg)
        hf.create_dataset("img_dark", data=img_dark)
        hf.create_dataset("img_bkg_avg", data=img_bkg_avg.astype(np.float32))
        hf.create_dataset("img_dark_avg", data=img_dark_avg.astype(np.float32))
        hf.create_dataset("img_tomo", data=img_tomo)
        hf.create_dataset("angle", data=img_angle)


def export_fly_scan(run, filepath="", **kwargs):
    uid = run.start["uid"]
    note = run.start["note"]
    scan_type = "fly_scan"
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]
    x_pos = run["baseline"]["data"]["zps_sx"][1].item()
    y_pos = run["baseline"]["data"]["zps_sy"][1].item()
    z_pos = run["baseline"]["data"]["zps_sz"][1].item()
    r_pos = run["baseline"]["data"]["zps_pi_r"][1].item()
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    x_eng = run.start["XEng"]
    img_angle = get_fly_scan_angle(run)

    img_tomo = np.array(list(run["primary"]["data"]["Andor_image"]))[0]
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))[0]
    img_bkg = np.array(list(run["flat"]["data"]["Andor_image"]))[0]

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=int(scan_id))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.uint16))
        hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.uint16))
        hf.create_dataset("img_bkg_avg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark_avg", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_tomo", data=np.array(img_tomo, dtype=np.uint16))
        hf.create_dataset("angle", data=img_angle)
        hf.create_dataset("x_ini", data=x_pos)
        hf.create_dataset("y_ini", data=y_pos)
        hf.create_dataset("z_ini", data=z_pos)
        hf.create_dataset("r_ini", data=r_pos)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(str(pxl_sz) + "nm"))


def export_fly_scan2(run, filepath="", **kwargs):
    uid = run.start["uid"]
    note = run.start["note"]
    scan_type = "fly_scan2"
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]
    x_pos = run["baseline"]["data"]["zps_sx"][1].item()
    y_pos = run["baseline"]["data"]["zps_sy"][1].item()
    z_pos = run["baseline"]["data"]["zps_sz"][1].item()
    r_pos = run["baseline"]["data"]["zps_pi_r"][1].item()
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    # sanity check: make sure we remembered the right stream name
    assert "zps_pi_r_monitor" in run
    pos = run["zps_pi_r_monitor"].read()
    img_dark = np.array(list(run["primary"]["data"]["Andor_image"])[-1][:])
    img_bkg = np.array(list(run["primary"]["data"]["Andor_image"])[-2][:])
    s = img_dark.shape
    img_dark_avg = np.mean(img_dark, axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])

    imgs = np.array(list(run["primary"]["data"]["Andor_image"])[:-2])
    s1 = imgs.shape
    imgs = imgs.reshape([s1[0] * s1[1], s1[2], s1[3]])

    chunked_timestamps = list(run["primary"]["data"]["Andor_image"])[:-2]

    raw_timestamps = []
    for chunk in chunked_timestamps:
        raw_timestamps.extend(chunk.tolist())

    timestamps = convert_AD_timestamps(pd.Series(raw_timestamps))
    pos["time"] = pos["time"].dt.tz_localize("US/Eastern")

    img_day, img_hour = (
        timestamps.dt.day,
        timestamps.dt.hour,
    )
    img_min, img_sec, img_msec = (
        timestamps.dt.minute,
        timestamps.dt.second,
        timestamps.dt.microsecond,
    )
    img_time = (
        img_day * 86400 + img_hour * 3600 + img_min * 60 + img_sec + img_msec * 1e-6
    )
    img_time = np.array(img_time)

    mot_day, mot_hour = (
        pos["time"].dt.day,
        pos["time"].dt.hour,
    )
    mot_min, mot_sec, mot_msec = (
        pos["time"].dt.minute,
        pos["time"].dt.second,
        pos["time"].dt.microsecond,
    )
    mot_time = (
        mot_day * 86400 + mot_hour * 3600 + mot_min * 60 + mot_sec + mot_msec * 1e-6
    )
    mot_time = np.array(mot_time)

    mot_pos = np.array(pos["zps_pi_r"])
    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)

    pos2 = mot_pos_interp.argmax() + 1
    # img_angle = mot_pos_interp[: pos2 - chunk_size]  # rotation angles
    img_angle = mot_pos_interp[:pos2]
    # img_tomo = imgs[: pos2 - chunk_size]  # tomo images
    img_tomo = imgs[:pos2]

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=int(scan_id))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.uint16))
        hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.uint16))
        hf.create_dataset("img_bkg_avg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark_avg", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_tomo", data=np.array(img_tomo, dtype=np.uint16))
        hf.create_dataset("angle", data=img_angle)
        hf.create_dataset("x_ini", data=x_pos)
        hf.create_dataset("y_ini", data=y_pos)
        hf.create_dataset("z_ini", data=z_pos)
        hf.create_dataset("r_ini", data=r_pos)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(str(pxl_sz) + "nm"))


def export_xanes_scan(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    uid = run.start["uid"]
    note = run.start["note"]
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]

    img_xanes = np.array(list(run["primary"]["data"]["Andor_image"]))
    img_xanes_avg = np.mean(img_xanes, axis=1)
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))
    img_dark_avg = np.mean(img_dark, axis=1)
    img_bkg = np.array(list(run["flat"]["data"]["Andor_image"]))
    img_bkg_avg = np.mean(img_bkg, axis=1)

    eng_list = list(run.start["eng_list"])

    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=eng_list)
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_xanes", data=np.array(img_xanes_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_xanes_scan_img_only(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    #    scan_type = 'xanes_scan'
    uid = run.start["uid"]
    note = run.start["note"]
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]

    img_xanes = np.array(list(run["primary"]["data"]["Andor_image"]))
    img_xanes_avg = np.mean(img_xanes, axis=1)
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))
    img_dark_avg = np.mean(img_dark, axis=1)
    img_bkg_avg = np.ones(img_dark_avg.shape)

    eng_list = list(run.start["eng_list"])

    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0

    filename = os.path.join(
        os.path.abspath(filepath), f"{scan_type}_id_{scan_id}_img_only.h5"
    )

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=eng_list)
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_xanes", data=np.array(img_xanes_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_z_scan(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    scan_id = run.start["scan_id"]
    uid = run.start["uid"]
    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    num = run.start["plan_args"]["steps"]
    note = run.start["plan_args"]["note"] if run.start["plan_args"]["note"] else "None"
    img = np.array(list(run["primary"]["data"]["Andor_image"]))
    img_zscan = np.mean(img[:num], axis=1)
    img_bkg = np.mean(img[num], axis=0, keepdims=True)
    img_dark = np.mean(img[-1], axis=0, keepdims=True)
    img_norm = (img_zscan - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=img_bkg.astype(np.float32))
        hf.create_dataset("img_dark", data=img_dark.astype(np.float32))
        hf.create_dataset("img", data=img_zscan.astype(np.float32))
        hf.create_dataset("img_norm", data=img_norm.astype(np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_z_scan2(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    scan_id = run.start["scan_id"]
    uid = run.start["uid"]
    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    note = run.start["plan_args"]["note"] if run.start["plan_args"]["note"] else "None"
    img = np.mean(np.array(list(run["primary"]["data"]["Andor_image"])), axis=1)
    img = np.squeeze(img)
    img_dark = img[0]
    l1 = np.arange(1, len(img), 2)
    l2 = np.arange(2, len(img), 2)

    img_zscan = img[l1]
    img_bkg = img[l2]

    img_norm = (img_zscan - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset(
            "img_bkg",
            data=np.array(img_bkg.astype(np.float32), dtype=np.float32),
        )
        hf.create_dataset("img_dark", data=img_dark.astype(np.float32))
        hf.create_dataset("img", data=img_zscan.astype(np.float32))
        hf.create_dataset(
            "img_norm",
            data=np.array(img_norm.astype(np.float32), dtype=np.float32),
        )
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_test_scan(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    scan_type = run.start["plan_name"]
    scan_id = run.start["scan_id"]
    uid = run.start["uid"]
    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    num = run.start["plan_args"]["num_img"]
    num_bkg = run.start["plan_args"]["num_bkg"]
    note = run.start["plan_args"]["note"] if run.start["plan_args"]["note"] else "None"
    img = np.squeeze(np.array(list(run["primary"]["data"]["Andor_image"])))
    assert len(img.shape) == 3, "load test_scan fails..."
    img_test = img[:num]
    img_bkg = np.mean(img[num : num + num_bkg], axis=0, keepdims=True)
    img_dark = np.mean(img[-num_bkg:], axis=0, keepdims=True)
    img_norm = (img_test - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("img_bkg", data=img_bkg)
        hf.create_dataset("img_dark", data=img_dark)
        hf.create_dataset("img", data=np.array(img_test, dtype=np.float32))
        hf.create_dataset("img_norm", data=np.array(img_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_count(run, filepath="", **kwargs):
    """
    load images (e.g. RE(count([Andor], 10)) ) and save to .h5 file
    """
    try:
        zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
        DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
        M = (DetU_z_pos / zp_z_pos - 1) * 10.0
        pxl_sz = 6500.0 / M
    except Exception:
        M = 0
        pxl_sz = 0
        print("fails to calculate magnification and pxl size")

    uid = run.start["uid"]
    det = run.start["detectors"][0]
    img = get_img(run, det)
    scan_id = run.start["scan_id"]
    filename = os.path.join(filepath, f"count_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("img", data=img.astype(np.float32))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_delay_count(run, filepath="", **kwargs):
    """
    load images (e.g. RE(count([Andor], 10)) ) and save to .h5 file
    """
    try:
        zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
        DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
        M = (DetU_z_pos / zp_z_pos - 1) * 10.0
        pxl_sz = 6500.0 / M
    except Exception:
        M = 0
        pxl_sz = 0
        print("fails to calculate magnification and pxl size")

    uid = run.start["uid"]
    scan_id = run.start["scan_id"]
    det = run.start["detectors"][0]
    img = get_img(run, det)

    filename = os.path.join(filepath, f"count_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("img", data=img.astype(np.float32))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_delay_scan(run, filepath="", **kwargs):
    det = run.start["detectors"][0]
    scan_type = run.start["plan_name"]
    scan_id = run.start["scan_id"]
    uid = run.start["uid"]
    x_eng = run.start["XEng"]
    note = run.start["plan_args"]["note"] if run.start["plan_args"]["note"] else "None"
    mot_name = run.start["plan_args"]["motor"]
    mot_start = run.start["plan_args"]["start"]
    mot_stop = run.start["plan_args"]["stop"]
    mot_steps = run.start["plan_args"]["steps"]
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    if det == "detA1" or det == "Andor":
        img = get_img(run, det)

        filename = os.path.join(
            os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5"
        )

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("img", data=np.array(img, dtype=np.float32))
            hf.create_dataset("uid", data=uid)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("X_eng", data=x_eng)
            hf.create_dataset("note", data=str(note))
            hf.create_dataset("start", data=mot_start)
            hf.create_dataset("stop", data=mot_stop)
            hf.create_dataset("steps", data=mot_steps)
            hf.create_dataset("motor", data=mot_name)
            hf.create_dataset("Magnification", data=M)
            hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
    else:
        print("no image stored in this scan")


def export_multipos_count(run, filepath="", **kwargs):
    scan_type = run.start["plan_name"]
    scan_id = run.start["scan_id"]
    uid = run.start["uid"]
    num_dark = run.start["num_dark_images"]
    num_of_position = run.start["num_of_position"]
    note = run.start["note"]
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    img_raw = list(run["primary"]["data"]["Andor_image"])
    img_dark = np.squeeze(np.array(img_raw[:num_dark]))
    img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    num_repeat = np.int(
        (len(img_raw) - 10) / num_of_position / 2
    )  # alternatively image and background

    tot_img_num = num_of_position * 2 * num_repeat
    s = img_dark.shape
    img_group = np.zeros([num_of_position, num_repeat, s[1], s[2]], dtype=np.float32)

    for j in range(num_repeat):
        index = num_dark + j * num_of_position * 2
        print(f"processing #{index} / {tot_img_num}")
        for i in range(num_of_position):
            tmp_img = np.array(img_raw[index + i * 2])
            tmp_bkg = np.array(img_raw[index + i * 2 + 1])
            img_group[i, j] = (tmp_img - img_dark_avg) / (tmp_bkg - img_dark_avg)

    filename = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
        for i in range(num_of_position):
            hf.create_dataset(f"img_pos{i+1}", data=np.squeeze(img_group[i]))


def export_grid2D_rel(run, filepath="", **kwargs):
    scan_type = "grid2D_rel"
    scan_id = run.start["scan_id"]
    num1 = run.start["plan_args"]["num1"]
    num2 = run.start["plan_args"]["num2"]
    img = np.squeeze(np.array(list(run["primary"]["data"]["Andor_image"])))

    folder_name = os.path.join(os.path.abspath(filepath), f"{scan_type}_id_{scan_id}")
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    for i in range(num1):
        for j in range(num2):
            filename = os.path.join(folder_name, f"_({i}{j}).tif")
            img = Image.fromarray(img[i * num1 + j])
            img.save(filename)


def export_raster_2D_2(run, binning=4, filepath="", **kwargs):
    from skimage import io

    num_dark = 5
    num_bkg = run.start["plan_args"]["num_bkg"]
    x_eng = run.start["XEng"]
    x_range = run.start["plan_args"]["x_range"]
    y_range = run.start["plan_args"]["y_range"]
    img_sizeX = run.start["plan_args"]["img_sizeX"]
    img_sizeY = run.start["plan_args"]["img_sizeY"]
    pix = run.start["plan_args"]["pxl"]
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    img_raw = np.squeeze(np.array(list(run["primary"]["data"]["Andor_image"])))
    img_dark_avg = np.mean(img_raw[:num_dark], axis=0, keepdims=True)
    s = img_dark_avg.shape
    # img_bkg_avg = np.mean(img_raw[-num_bkg:], axis=0, keepdims = True)
    # img = img_raw[num_dark:-num_bkg]

    num_img = (x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1)
    img = np.zeros([num_img, s[1], s[2]])
    for i in range(num_img):
        index = num_dark + i * num_bkg + i
        img_bkg_avg = np.mean(
            img_raw[index + 1 : index + 1 + num_bkg], axis=0, keepdims=True
        )
        img[i] = (img_raw[index] - img_dark_avg) / (img_bkg_avg - img_dark_avg)

    s = img.shape

    x_num = round((x_range[1] - x_range[0]) + 1)
    y_num = round((y_range[1] - y_range[0]) + 1)
    x_list = np.linspace(x_range[0], x_range[1], x_num)
    y_list = np.linspace(y_range[0], y_range[1], y_num)
    row_size = y_num * s[1]
    col_size = x_num * s[2]
    img_patch = np.zeros([1, row_size, col_size])
    index = 0
    pos_file_for_print = np.zeros([x_num * y_num, 4])
    pos_file = ["cord_x\tcord_y\tx_pos_relative\ty_pos_relative\n"]
    index = 0
    for i in range(int(x_num)):
        for j in range(int(y_num)):
            img_patch[0, j * s[1] : (j + 1) * s[1], i * s[2] : (i + 1) * s[2]] = img[
                index
            ]
            pos_file_for_print[index] = [
                x_list[i],
                y_list[j],
                x_list[i] * pix * img_sizeX / 1000,
                y_list[j] * pix * img_sizeY / 1000,
            ]
            pos_file.append(
                f"{x_list[i]:3.0f}\t{y_list[j]:3.0f}\t{x_list[i]*pix*img_sizeX/1000:3.3f}\t\t{y_list[j]*pix*img_sizeY/1000:3.3f}\n"
            )
            index = index + 1
    s = img_patch.shape
    img_patch_bin = bin_ndarray(
        img_patch, new_shape=(1, int(s[1] / binning), int(s[2] / binning))
    )
    scan_id = run.start["scan_id"]
    fout_tiff = filepath + f"raster2D_scan_{scan_id}_binning_{binning}.tiff"
    fout_txt = filepath + f"raster2D_scan_{scan_id}_cord.txt"
    print(f"{pos_file_for_print}")
    io.imsave(fout_tiff, np.array(img_patch_bin[0], dtype=np.float32))
    with open(f"{fout_txt}", "w+") as f:
        f.writelines(pos_file)
    # tifffile.imsave(fout_tiff, np.array(img_patch_bin, dtype=np.float32))
    num_img = int(x_num) * int(y_num)

    folder_name = os.path.join(os.path.abspath(filepath), f"raster_scan_{scan_id}")
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(folder_name, f"img_{i:02d}_binning_{binning}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("img_patch", data=np.array(img_patch_bin, np.float32))
        hf.create_dataset("img", data=np.array(img, np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, np.float32))
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, np.float32))
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_raster_2D(run, binning=4, filepath="", **kwargs):
    import tifffile

    scan_id = run.start["scan_id"]
    num_dark = run.start["num_dark_images"]
    num_bkg = run.start["num_bkg_images"]
    x_eng = run.start["XEng"]
    x_range = run.start["plan_args"]["x_range"]
    y_range = run.start["plan_args"]["y_range"]
    img_sizeX = run.start["plan_args"]["img_sizeX"]
    img_sizeY = run.start["plan_args"]["img_sizeY"]
    pix = run.start["plan_args"]["pxl"]
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    img_raw = np.squeeze(np.array(list(run["primary"]["data"]["Andor_image"])))
    img_dark_avg = np.mean(img_raw[:num_dark], axis=0, keepdims=True)
    img_bkg_avg = np.mean(img_raw[-num_bkg:], axis=0, keepdims=True)
    img = img_raw[num_dark:-num_bkg]
    s = img.shape
    img = (img - img_dark_avg) / (img_bkg_avg - img_dark_avg)
    x_num = round((x_range[1] - x_range[0]) + 1)
    y_num = round((y_range[1] - y_range[0]) + 1)
    x_list = np.linspace(x_range[0], x_range[1], x_num)
    y_list = np.linspace(y_range[0], y_range[1], y_num)
    row_size = y_num * s[1]
    col_size = x_num * s[2]
    img_patch = np.zeros([1, row_size, col_size])
    index = 0
    pos_file_for_print = np.zeros([x_num * y_num, 4])
    pos_file = ["cord_x\tcord_y\tx_pos_relative\ty_pos_relative\n"]
    index = 0
    for i in range(int(x_num)):
        for j in range(int(y_num)):
            img_patch[0, j * s[1] : (j + 1) * s[1], i * s[2] : (i + 1) * s[2]] = img[
                index
            ]
            pos_file_for_print[index] = [
                x_list[i],
                y_list[j],
                x_list[i] * pix * img_sizeX / 1000,
                y_list[j] * pix * img_sizeY / 1000,
            ]
            pos_file.append(
                f"{x_list[i]:3.0f}\t{y_list[j]:3.0f}\t{x_list[i]*pix*img_sizeX/1000:3.3f}\t\t{y_list[j]*pix*img_sizeY/1000:3.3f}\n"
            )
            index = index + 1
    s = img_patch.shape
    img_patch_bin = bin_ndarray(
        img_patch, new_shape=(1, int(s[1] / binning), int(s[2] / binning))
    )
    fout_tiff = filepath + f"raster2D_scan_{scan_id}_binning_{binning}.tiff"
    fout_txt = filepath + f"raster2D_scan_{scan_id}_cord.txt"
    print(f"{pos_file_for_print}")
    with open(f"{fout_txt}", "w+") as f:
        f.writelines(pos_file)
    tifffile.imsave(fout_tiff, np.array(img_patch_bin, dtype=np.float32))

    folder_name = os.path.join(os.path.abspath(filepath), f"raster_scan_{scan_id}")
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(folder_name, f"img_{i:02d}_binning_{binning}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("img_patch", data=np.array(img_patch_bin, np.float32))
        hf.create_dataset("img", data=np.array(img, np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, np.float32))
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, np.float32))
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_multipos_2D_xanes_scan2(run, filepath="", **kwargs):
    scan_type = run.start["plan_name"]
    uid = run.start["uid"]
    note = run.start["note"]
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]
    #    x_eng = run.start['x_ray_energy']
    num_eng = run.start["num_eng"]
    num_pos = run.start["num_pos"]
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    try:
        repeat_num = run.start["plan_args"]["repeat_num"]
    except Exception:
        repeat_num = 1
    img_xanes = np.array(list(run["primary"]["data"]["Andor_image"]))
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))
    img_bkg = np.array(list(run["flat"]["data"]["Andor_image"]))
    img_xanes = np.mean(img_xanes, axis=1)
    img_dark = np.mean(img_dark, axis=1)
    img_bkg = np.mean(img_bkg, axis=1)
    eng_list = list(run.start["eng_list"])

    for repeat in range(repeat_num):  # revised here
        print(f"repeat: {repeat}")
        id_s = int(repeat * num_eng)
        id_e = int((repeat + 1) * num_eng)
        img_x = img_xanes[id_s * num_pos : id_e * num_pos]  # xanes image
        img_b = img_bkg[id_s:id_e]  # bkg image
        for j in range(num_pos):
            img_p = img_x[j::num_pos]
            img_p_n = (img_p - img_dark) / (img_b - img_dark)
            name = f"{scan_type}_id_{scan_id}_repeat_{repeat:02d}_pos_{j:02d}.h5"
            filename = os.path.join(os.path.abspath(filepath), name)

            with h5py.File(filename, "w") as hf:
                hf.create_dataset("uid", data=uid)
                hf.create_dataset("scan_id", data=scan_id)
                hf.create_dataset("note", data=str(note))
                hf.create_dataset("scan_time", data=scan_time)
                hf.create_dataset("X_eng", data=eng_list)
                hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.float32))
                hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.float32))
                hf.create_dataset("img_xanes", data=np.array(img_p_n, dtype=np.float32))
                hf.create_dataset("Magnification", data=M)
                hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_multipos_2D_xanes_scan3(run, filepath="", **kwargs):
    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    uid = run.start["uid"]
    note = run.start["note"]
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]
    num_eng = run.start["num_eng"]
    num_pos = run.start["num_pos"]
    imgs = np.array(list(run["primary"]["data"]["Andor_image"]))
    imgs = np.mean(imgs, axis=1)
    img_dark = imgs[0]
    eng_list = list(run.start["eng_list"])

    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_bkg = np.zeros([num_eng, imgs.shape[1], imgs.shape[2]])

    index = 1
    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j, i] = imgs[index]
            index += 1

    img_bkg = imgs[-num_eng:]

    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j, i] = (img_xanes[j, i] - img_dark) / (img_bkg[i] - img_dark)
    for j in range(num_pos):
        filename = os.path.join(
            os.path.abspath(filepath), f"{scan_type}_id_{scan_id}_pos_{j}.h5"
        )

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("uid", data=uid)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("note", data=str(note))
            hf.create_dataset("scan_time", data=scan_time)
            hf.create_dataset("X_eng", data=eng_list)
            hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.float32))
            hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.float32))
            hf.create_dataset(
                "img_xanes", data=np.array(img_xanes[j], dtype=np.float32)
            )
            hf.create_dataset("Magnification", data=M)
            hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


def export_user_fly_only(run, filepath="", **kwargs):
    uid = run.start["uid"]
    note = run.start["note"]
    scan_id = run.start["scan_id"]
    scan_time = run.start["time"]
    x_pos = run["baseline"]["data"]["zps_sx"][1].item()
    y_pos = run["baseline"]["data"]["zps_sy"][1].item()
    z_pos = run["baseline"]["data"]["zps_sz"][1].item()
    r_pos = run["baseline"]["data"]["zps_pi_r"][1].item()

    try:
        x_eng = run.start["XEng"]
    except Exception:
        x_eng = run.start["x_ray_energy"]
    # sanity check: make sure we remembered the right stream name
    assert "zps_pi_r_monitor" in run
    pos = run["zps_pi_r_monitor"].read()
    imgs = np.array(list(run["primary"]["data"]["Andor_image"]))

    s1 = imgs.shape
    chunk_size = s1[1]
    imgs = imgs.reshape(-1, s1[2], s1[3])

    # load darks and bkgs
    img_dark = np.array(list(run["dark"]["data"]["Andor_image"]))[0]
    img_bkg = np.array(list(run["flat"]["data"]["Andor_image"]))[0]
    s = img_dark.shape
    img_dark_avg = np.mean(img_dark, axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])

    chunked_timestamps = list(run["primary"]["data"]["Andor_image"])
    raw_timestamps = []
    for chunk in chunked_timestamps:
        raw_timestamps.extend(chunk.tolist())

    timestamps = convert_AD_timestamps(pd.Series(raw_timestamps))
    pos["time"] = pos["time"].dt.tz_localize("US/Eastern")

    img_day, img_hour = (
        timestamps.dt.day,
        timestamps.dt.hour,
    )
    img_min, img_sec, img_msec = (
        timestamps.dt.minute,
        timestamps.dt.second,
        timestamps.dt.microsecond,
    )
    img_time = (
        img_day * 86400 + img_hour * 3600 + img_min * 60 + img_sec + img_msec * 1e-6
    )
    img_time = np.array(img_time)

    mot_day, mot_hour = (
        pos["time"].dt.day,
        pos["time"].dt.hour,
    )
    mot_min, mot_sec, mot_msec = (
        pos["time"].dt.minute,
        pos["time"].dt.second,
        pos["time"].dt.microsecond,
    )
    mot_time = (
        mot_day * 86400 + mot_hour * 3600 + mot_min * 60 + mot_sec + mot_msec * 1e-6
    )
    mot_time = np.array(mot_time)

    mot_pos = np.array(pos["zps_pi_r"])
    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)

    pos2 = mot_pos_interp.argmax() + 1
    img_angle = mot_pos_interp[: pos2 - chunk_size]  # rotation angles
    img_tomo = imgs[: pos2 - chunk_size]  # tomo images

    filename = os.path.join(os.path.abspath(filepath), f"fly_scan_id_{scan_id}.h5")

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=int(scan_id))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.uint16))
        hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.uint16))
        hf.create_dataset("img_bkg_avg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark_avg", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_tomo", data=np.array(img_tomo, dtype=np.uint16))
        hf.create_dataset("angle", data=img_angle)
        hf.create_dataset("x_ini", data=x_pos)
        hf.create_dataset("y_ini", data=y_pos)
        hf.create_dataset("z_ini", data=z_pos)
        hf.create_dataset("r_ini", data=r_pos)


def export_scan_change_expo_time(
    run, filepath="", save_range_x=[], save_range_y=[], **kwargs
):
    from skimage import io

    scan_id = run.start["scan_id"]
    filepath += f"scan_{scan_id}/"
    filepath_t1 = filepath + "t1/"
    filepath_t2 = filepath + "t2/"
    os.makedirs(filepath, exist_ok=True, mode=0o777)
    os.makedirs(filepath_t1, exist_ok=True, mode=0o777)
    os.makedirs(filepath_t2, exist_ok=True, mode=0o777)

    zp_z_pos = run["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = run["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = run.start["plan_name"]
    uid = run.start["uid"]
    note = run.start["plan_args"]["note"]

    x_eng = run.start["x_ray_energy"]

    img_sizeX = run.start["plan_args"]["img_sizeX"]
    img_sizeY = run.start["plan_args"]["img_sizeY"]
    pxl = run.start["plan_args"]["pxl"]
    step_x = img_sizeX * pxl
    step_y = img_sizeY * pxl

    x_range = run.start["plan_args"]["x_range"]
    y_range = run.start["plan_args"]["y_range"]

    imgs = list(run["primary"]["data"]["Andor_image"])
    s = imgs[0].shape

    if len(save_range_x) == 0:
        save_range_x = [0, s[0]]
    if len(save_range_y) == 0:
        save_range_y = [0, s[1]]

    img_dark_t1 = np.median(np.array(imgs[:5]), axis=0)
    img_dark_t2 = np.median(np.array(imgs[5:10]), axis=0)
    imgs = imgs[10:]

    nx = np.abs(x_range[1] - x_range[0] + 1)
    ny = np.abs(y_range[1] - y_range[0] + 1)
    pos_x = np.zeros(nx * ny)
    pos_y = pos_x.copy()

    idx = 0

    for ii in range(nx):
        if not ii % 100:
            print(f"nx = {ii}")
        for jj in range(ny):
            if not jj % 10:
                print(f"ny = {jj}")
            pos_x[idx] = ii * step_x
            pos_y[idx] = jj * step_y
            idx += 1
            id_c = ii * ny * (5 + 5 + 2) + jj * (5 + 5 + 2)
            img_t1 = imgs[id_c]
            img_t2 = imgs[id_c + 1]
            img_bkg_t1 = imgs[(id_c + 2) : (id_c + 7)]
            img_bkg_t1 = np.median(img_bkg_t1, axis=0)
            img_bkg_t2 = imgs[(id_c + 7) : (id_c + 12)]
            img_bkg_t2 = np.median(img_bkg_t2, axis=0)

            img_t1_n = (img_t1 - img_dark_t1) / (img_bkg_t1 - img_dark_t1)
            img_t2_n = (img_t2 - img_dark_t2) / (img_bkg_t2 - img_dark_t2)

            fsave_t1 = filepath_t1 + f"img_t1_{idx:05d}.tiff"
            fsave_t2 = filepath_t2 + f"img_t2_{idx:05d}.tiff"

            im1 = img_t1_n[
                0,
                save_range_x[0] : save_range_x[1],
                save_range_y[0] : save_range_y[1],
            ]
            im2 = img_t2_n[
                0,
                save_range_x[0] : save_range_x[1],
                save_range_y[0] : save_range_y[1],
            ]
            io.imsave(fsave_t1, im1.astype(np.float32))
            io.imsave(fsave_t2, im2.astype(np.float32))
    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("scan_type", data=scan_type)
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("pxl_sz", data=pxl_sz)
        hf.create_dataset("note", data=note)
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("pos_x", data=pos_x)
        hf.create_dataset("pos_y", data=pos_y)
