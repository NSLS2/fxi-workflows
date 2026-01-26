import numpy as np
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
    tomo_angle = np.arrayimg_angle
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
    f.close()

    prj_norm = denoise(prj_norm, denoise_flag)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0

    prj_norm -= bkg_level

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
