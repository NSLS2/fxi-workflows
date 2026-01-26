import prefect
from prefect import task, Flow, Parameter
from tiled.client import from_profile
from datetime import datetime
import numpy as np
import pandas as pd


def timestamp_to_float(t):
    tf = []
    for ts in t:
        tf.append(ts)
    return np.array(tf)


def get_fly_scan_angle(input_dict):
    timestamp_tomo = input_dict["timestamp_tomo"]
    pos = input_dict["pos"]
    mot_pos = input_dict["mot_pos"]

    timestamp_mot = timestamp_to_float(pos["time"])

    img_ini_timestamp = timestamp_tomo[0][0]
    mot_ini_timestamp = timestamp_mot[
        1
    ]  # timestamp_mot[1] is the time when taking dark image

    print(f"timestamp_tomo: {timestamp_tomo} img_ini_timestamp: {img_ini_timestamp}")
    tomo_time = timestamp_tomo[0] - img_ini_timestamp
    mot_time = timestamp_mot - mot_ini_timestamp

    mot_pos_interp = np.interp(tomo_time, mot_time, mot_pos)

    img_angle = mot_pos_interp
    return img_angle


@task(log_stdout=True)
def call_find_rot(uid):
    c = from_profile("nsls2", username=None)
    scan_result = c["fxi"][uid]

    logger = prefect.context.get("logger")
    logger.info(scan_result.start)

    # sanity check: make sure we remembered the right stream name
    assert "zps_pi_r_monitor" in scan_result
    pos = scan_result["zps_pi_r_monitor"]["data"]
    logger.info("extracting data from tiled")
    imgs = np.array(list(scan_result["primary"]["data"]["Andor_image"]))

    s1 = imgs.shape
    chunk_size = s1[1]
    imgs = imgs.reshape(-1, s1[2], s1[3])
    logger.info("done with primary images")

    # load darks and bkgs
    img_dark = np.array(list(scan_result["dark"]["data"]["Andor_image"]))[0]
    logger.info("done with darks")
    img_bkg = np.array(list(scan_result["flat"]["data"]["Andor_image"]))[0]
    logger.info("done with background")
    img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.mean(img_bkg, axis=0, keepdims=True)

    chunked_timestamps = list(scan_result["primary"]["data"]["Andor_timestamps"])

    mot_pos = np.array(pos["zps_pi_r"])

    input_dict = {"pos": pos, "timestamp_tomo": chunked_timestamps, "mot_pos": mot_pos}
    img_tomo = np.array(list(scan_result["primary"]["data"]["Andor_image"]))[0]
    logger.info(img_tomo)
    img_angle = get_fly_scan_angle(input_dict)
    logger.info("calculating rotation center")
    img, cen = rotcen_test2(img_tomo, img_bkg_avg, img_dark_avg, img_angle)
    return img, cen


with Flow("test-find-rot") as flow1:
    uid = Parameter("uid")
    call_find_rot(uid)

EPICS_EPOCH = datetime(1990, 1, 1, 0, 0)


def convert_AD_timestamps(ts):
    return pd.to_datetime(ts, unit="s", origin=EPICS_EPOCH, utc=True).dt.tz_convert(
        "US/Eastern"
    )


def get_tomo_images(input_dict):
    pos = input_dict["pos"]
    imgs = input_dict["imgs"]
    chunked_timestamps = input_dict["chunked_timestamps"]
    mot_pos = input_dict["mot_pos"]

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

    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)

    pos2 = mot_pos_interp.argmax() + 1
    img_angle = mot_pos_interp[: pos2 - chunk_size]  # rotation angles
    img_tomo = imgs[: pos2 - chunk_size]  # tomo images
    return img_tomo, img_angle


import tomopy
from scipy.interpolate import interp1d


def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()


def rotcen_test2(
    img_tomo,
    img_bkg_avg,
    img_dark_avg,
    img_angle,
    start=None,
    stop=None,
    steps=None,
    sli=0,
    block_list=[],
    print_flag=1,
    bkg_level=0,
    txm_normed_flag=0,
    denoise_flag=0,
    fw_level=9,
    algorithm="gridrec",
    n_iter=5,
    circ_mask_ratio=0.95,
    options={},
    atten=None,
    clim=[],
    dark_scale=1,
    filter_name="None",
):
    print("beginning of rotcen2")
    s = [1, data.shape[0], data.shape[1]]

    if atten is not None:
        ref_ang = atten[:, 0]
        ref_atten = atten[:, 1]
        fint = interp1d(ref_ang, ref_atten)

    if denoise_flag:
        addition_slice = 100
    else:
        addition_slice = 0

    if sli == 0:
        sli = int(s[1] / 2)
    sli_exp = [
        np.max([0, sli - addition_slice // 2]),
        np.min([sli + addition_slice // 2 + 1, s[1]]),
    ]
    tomo_angle = np.array(img_angle)
    theta = tomo_angle / 180.0 * np.pi
    img_tomo = np.array(img_tomo[:, sli_exp[0] : sli_exp[1], :])

    if txm_normed_flag:
        prj_norm = img_tomo
    else:
        img_bkg = np.array(img_bkg_avg[:, sli_exp[0] : sli_exp[1], :])
        img_dark = np.array(img_dark_avg[:, sli_exp[0] : sli_exp[1], :]) / dark_scale
        prj = (img_tomo - img_dark) / (img_bkg - img_dark)
        if atten is not None:
            for i in range(len(tomo_angle)):
                att = fint(tomo_angle[i])
                prj[i] = prj[i] / att
        prj_norm = -np.log(prj)

    prj_norm = denoise(prj_norm, denoise_flag)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0

    prj_norm -= bkg_level

    print("tomopy prep")
    prj_norm = tomopy.prep.stripe.remove_stripe_fw(
        prj_norm, level=fw_level, wname="db5", sigma=1, pad=True
    )
    """
    if denoise_flag == 1: # denoise using wiener filter
        ss = prj_norm.shape
        for i in range(ss[0]):
           prj_norm[i] = skr.wiener(prj_norm[i], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip)
    elif denoise_flag == 2:
        from skimage.filters import gaussian as gf
        prj_norm = gf(prj_norm, [0, 1, 1])
    """
    s = prj_norm.shape
    if len(s) == 2:
        prj_norm = prj_norm.reshape(s[0], 1, s[1])
        s = prj_norm.shape

    if theta[-1] > theta[1]:
        pos = find_nearest(theta, theta[0] + np.pi)
    else:
        pos = find_nearest(theta, theta[0] - np.pi)
    block_list = list(block_list) + list(np.arange(pos + 1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start == None or stop == None or steps == None:
        start = int(s[2] / 2 - 50)
        stop = int(s[2] / 2 + 50)
        steps = 26
    cen = np.linspace(start, stop, steps)
    img = np.zeros([len(cen), s[2], s[2]])
    print("tomopy start reconstructions")
    for i in range(len(cen)):
        if print_flag:
            print("{}: rotcen {}".format(i + 1, cen[i]))
            if algorithm == "gridrec":
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm="gridrec",
                    filter_name=filter_name,
                )
            elif "astra" in algorithm:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm=tomopy.astra,
                    options=options,
                )
            else:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm=algorithm,
                    num_iter=n_iter,
                    filter_name=filter_name,
                )
    print("tomopy circ_mask")
    img = tomopy.circ_mask(img, axis=0, ratio=circ_mask_ratio)
    return img, cen


def denoise(prj, denoise_flag):
    if denoise_flag == 1:  # Wiener denoise
        import skimage.restoration as skr

        ss = prj.shape
        psf = np.ones([2, 2]) / (2**2)
        reg = None
        balance = 0.3
        is_real = True
        clip = True
        for j in range(ss[0]):
            prj[j] = skr.wiener(
                prj[j], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip
            )
    elif denoise_flag == 2:  # Gaussian denoise
        from skimage.filters import gaussian as gf

        prj = gf(prj, [0, 1, 1])
    return prj
