#!/opt/anaconda3/envs/p11/bin/python3

import sys
import os
import re
import glob
import statistics
import subprocess
from pyraf import iraf
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
import tkinter as tk
import customtkinter as ctk
from scipy.optimize import curve_fit

import Ha_main
import bottom_a
import starmatch_a as sta


matrix_path   = '/Users/motomo/iriki/matrix/haon2_to_haoff_IRSF_241203.npy'
fits_pattern  = None

force_fwhm_on = None
coef_annulus_on  = 2.5
coef_dannulus_on = 3

force_fwhm_of = None
coef_annulus_of  = 2.5
coef_dannulus_of = 3



def read_coofile(infile):
        with open(infile, 'r') as file:
            flines = file.readlines()
        lines = [line.strip().split() for line in flines if not line.startswith('#')]
        coords = []
        for line in lines:
            if 'INDEF' in line:
                coords.append(line)
                continue
            x = float(line[0]) - 1  # X座標
            y = float(line[1]) - 1  # Y座標
            coords.append([x, y])
        return coords


def estimate_homography(points_src, points_dst):
    A = []
    for (x, y), (x_prime, y_prime) in zip(points_src, points_dst):
        # h11, h12, h13, h21, h22, h23
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
    A = np.array(A)

    b = points_dst.flatten()

    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    H = np.array([
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [0,    0,    1]  
    ])

    return H




def do_estcoords(coords_file_list, H_array):
    outlist = []
    for file in coords_file_list:
        with open(file) as f1:
            lines = f1.readlines()

        header = []
        coords = []
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                header.append(line)
                continue
            varr = line.split()
            x = float(varr[0])
            y = float(varr[1])
            coo0 = np.array([[x], [y], [1]])
            coo = np.dot(H_array, coo0)
            #print(f'{H_array}・{coo0} = {coo}')
            coords.append(f'{coo[0][0]} {coo[1][0]}')

        key = {'haon_':'haoff', 'haoff':'haon_'}
        file2 = re.sub(file[:5], key[file[:5]], file)
        with open(file2, 'w') as f2:
            for varr in header:
                f2.write(f'{varr}')
            for varr in coords:
                f2.write(f'{varr}\n')
        outlist.append(file2)

    return outlist


def do_phot(fitslist, coofilelist, datamax='INDEF', fwhm=5, coef1=2.5, coef2=3):
    bottom_a.setparam()
    magf_list = []
    for fits, coof in zip(fitslist, coofilelist):
        #sigma = bottom_a.skystat(fits, 'stddev')
        #bottom_a.phot(fits, fwhm, sigma)
        iraf.apphot.datapars.unlearn()
        iraf.apphot.centerpars.unlearn()
        iraf.apphot.fitskypars.unlearn()
        iraf.apphot.centerpars.calgorithm = 'centroid'

        iraf.apphot.datapars.datamax = datamax
        iraf.apphot.photpars.apertures = coef1*float(fwhm)
        iraf.apphot.fitskypars.annulus = coef2*float(fwhm)
        iraf.apphot.fitskypars.dannulus = 10
        iraf.apphot.photpars.zmag = 0

        iraf.apphot.phot.interactive = 'no'
        iraf.apphot.phot.cache = 'no'
        iraf.apphot.phot.verify = 'no'
        iraf.apphot.phot.update = 'yes'
        iraf.apphot.phot.verbose = 'no'
        iraf.apphot.phot.mode = 'h'

        iraf.apphot.phot.coords = coof
        iraf.apphot.phot.output = f'{coof[:-4]}.mag.1'
        try:
            os.remove(f'{coof[:-4]}.mag.1')
        except:
            pass
        #print('a')
        iraf.apphot.phot(fits)
        magf_list.append(f'{coof[:-4]}.mag.1')
    return magf_list
        


def do_txdump(magfilelist):
    txd_list = []
    for file in magfilelist:
        #print(f'fileleleelle{file}')
        iraf.apphot.txdump.fields = 'id,xcenter,ycenter,mag,merr'
        iraf.apphot.txdump.expr = 'yes'
        iraf.apphot.txdump.mode = 'h'
        varr = iraf.apphot.txdump(file, Stdout=1)
        outname = re.sub('mag.1', 'txdump', file)
        try:
            os.remove(outname)
        except:
            pass
        with open(outname, 'w') as f1:
            for varr2 in varr:
                f1.write(f'{varr2}\n')
        txd_list.append(outname)
    return txd_list


def do_plot(offtxlist, ontxlist):
    for offile1, onfile1 in zip(offtxlist, ontxlist):
        with open(offile1) as f1, open(onfile1) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        xaxis = []
        yaxis = []
        yerr = []
        #print(f'file = {offile1}, {onfile1}')
        #print(f'lines1 = {len(lines1)}')
        #print(f'lines2 = {len(lines2)}')
        for line1, line2 in zip(lines1, lines2):
            varr1 = line1.split()
            varr2 = line2.split()
            if varr1[2] == 'INDEF' or varr2[2] == 'INDEF':
                continue
            xaxis.append(float(varr1[2]))
            yaxis.append(float(varr1[2])-float(varr2[2]))
            yerr.append((float(varr1[3])**2 + float(varr2[3])**2)**0.5)
        
        #plt.plot(xaxis, yaxis, label='Sample Data', marker='o')
        plt.errorbar(xaxis, yaxis, yerr=yerr, fmt='o', label='Sample Data')  # fmt='o' は点を表す
        plt.xlabel('Haoff mag')
        plt.ylabel('Ha off-on')
        plt.legend()
        plt.savefig(f'{offile1[5:-7]}.png', dpi=300)
        plt.clf()



def glob_pattern(pattern="*.fits"):
    fitslist = glob.glob(pattern)
    grouped_files = {}
    for fitsname in fitslist:
        parts = fitsname.split(day)
        if len(parts)==1:
            continue
        fits_id = parts[1].replace(".fits", "")

        if not fits_id in grouped_files:
            grouped_files[fits_id] = [None, None]

        if 'haon' in parts[0]:
            grouped_files[fits_id][0] = fitsname
        elif 'haoff' in parts[0]:
            grouped_files[fits_id][1] = fitsname

    filtered_files = {k: v for k, v in grouped_files.items() if all(v)}

    return filtered_files


def adapt_matrix(coordsfile, matrix, outputfile):
    with open(coordsfile, 'r') as f1:
        lines1 = f1.readlines()
    coords1 = np.array([list(map(float, line1.split())) for line1 in lines1])
    coords_with_bias = np.hstack([coords1, np.ones((coords1.shape[0], 1))])
    #print(coords_with_bias)
    transformed_coords = coords_with_bias @ matrix.T
    with open(outputfile, 'w') as f_out:
        for coord in transformed_coords:
            f_out.write(" ".join(map(str, coord)) + "\n")
    return outputfile



def moffat_2d(coords, A, alpha, beta, x_c, y_c, offset):
    x, y = coords
    return A * (1 + ((x - x_c)**2 + (y - y_c)**2) / alpha**2) ** (-beta) + offset

def refine_center_2d(fitsname, coof, tol=1e-5, max_iter=10):
    
    data = fits.getdata(fitsname)
    coords = read_coofile(coof)
    data_flat_sorted = np.sort(data.ravel())
    index0 = int(len(data_flat_sorted) / 2)
    lower = data_flat_sorted[:index0]
    offset_fixed = np.median(lower)

    alpha_init = 2.8
    beta_init = 2  # 一般的な初期値

    centroids = []
    for index, coo in enumerate(coords):
        x, y = int(coo[0]), int(coo[1])
        x_start, x_end = x - 8, x + 9
        y_start, y_end = y - 8, y + 9
        if x_start < 0 or y_start < 0 or x_end > data.shape[1] or y_end > data.shape[0]:
            #print(f"Skipping coordinates ({x}, {y}) - slice out of bounds.")
            continue
        slice_image = data[y_start:y_end, x_start:x_end]
        sigma = np.ones_like(slice_image)

        y_indices, x_indices = np.indices(slice_image.shape)
        coords = np.vstack((x_indices.ravel(), y_indices.ravel()))

        x_c, y_c = (slice_image.shape[1] // 2, slice_image.shape[0] // 2)
        A_init = np.max(slice_image) - offset_fixed

        for iteration in range(max_iter):
            # フィッティング対象のMoffat関数（中心を可変）
            def moffat_2d_fixed_offset(coords, A, alpha, beta, x_c, y_c):
                return moffat_2d(coords, A, alpha, beta, x_c, y_c, offset_fixed)

            # フィット実行
            initial_guess = [A_init, alpha_init, beta_init, x_c, y_c]
            try:
                popt, _ = curve_fit(
                    moffat_2d_fixed_offset,
                    coords,
                    slice_image.ravel(),
                    p0=initial_guess,
                    sigma=sigma.ravel(),
                    absolute_sigma=True,
                    bounds=(
                        [0, 0.1, 0.1, 0, 0],  # パラメータの下限
                        [np.inf, 10, 10, slice_image.shape[1], slice_image.shape[0]]  # パラメータの上限
                    )
                )
            except:
                popt = None
                break
            A, alpha, beta, x_c_new, y_c_new = popt
            if np.sqrt((x_c_new - x_c)**2 + (y_c_new - y_c)**2) < tol:
                #print("Converged!")
                break

        if popt is None:
            centroids.append(('INDEF', 'INDEF'))
            continue

        centroids.append((popt[3] + x_start + 1, popt[4] + y_start + 1))
    
    with open(coof, 'w') as f_out:
        for coord in centroids:
            f_out.write(" ".join(map(str, coord)) + "\n")

    return coof


def moffat_1d(distances, A, alpha, beta, offset):
    return A * (1 + (distances / alpha) ** 2) ** (-beta) + offset

def calc_fwhm(fitsname, coof, outname=None):

    satcount = {
        'haoff':param.haoff_satcount, 'haon_':param.haon__satcount,
        }

    result = []
    data = fits.getdata(fitsname)
    coords = read_coofile(coof)
    num_coords = len(coords)
    stack_data = np.zeros((num_coords, 17, 17))
    count = 0
    stack_locoo = []
    stack_glcoo = []
    for i, coo in enumerate(coords):
        if 'INDEF' in coo:
            continue
        x, y = int(coo[0]), int(coo[1])
        x_start, x_end = x - 8, x + 9
        y_start, y_end = y - 8, y + 9
        if x_start < 0 or y_start < 0 or x_end > data.shape[1] or y_end > data.shape[0]:
                #print(f"Skipping coordinates ({x}, {y}) - slice out of bounds.")
                continue
        slice_image = data[y_start:y_end, x_start:x_end]
        stack_data[count] = slice_image
        count += 1
        stack_locoo.append((coo[0] - int(x) + 8, coo[1] - int(y) + 8))
        stack_glcoo.append((coo[0], coo[1]))
    result = result[:count]

    def moffat_1d_fixed_offset(distances, A, alpha, beta):
        return moffat_1d(distances, A, alpha, beta, offset)

    alpha_init = 2.8
    beta_init = 2

    y_indices, x_indices = np.indices((17, 17))
    x = x_indices.ravel()
    y = y_indices.ravel()

    data_flat_sorted = np.sort(data.ravel())
    index0 = int(len(data)/3)
    offset = np.median(data_flat_sorted[:-index0])
    #offset = 0

    popt_list = []
    distances_list = []
    intensities_list = []

    for index1, (sliced_data, locoo) in enumerate(zip(stack_data, stack_locoo)):

        A_init = np.max(sliced_data)
        initial_guess = [A_init, alpha_init, beta_init]

        #xx, yy = np.meshgrid(x, y)
        #distances = np.sqrt((xx - locoo[0])**2 + (yy - locoo[1])**2)

        distances = []
        intensities = []
        sigmas = []

        data_flat_sorted = np.sort(sliced_data.ravel())
        index0 = int(len(sliced_data)/2)
        offset = np.median(data_flat_sorted[:index0])

        # 各ピクセルの重心からの距離を計算
        for y in range(sliced_data.shape[0]):
            for x in range(sliced_data.shape[1]):
                distance = np.sqrt((x - locoo[0])**2 + (y - locoo[1])**2)
                intensity = sliced_data[y, x]

                # データを保存
                distances.append(distance)
                intensities.append(intensity)
                sigmas.append(1 / (distance ** 3))
                #sigmas.append(1)

        distances = np.array(distances)
        intensities = np.array(intensities)
        sigmas = np.array(sigmas)


        """
        print(locoo)
        plt.figure(figsize=(10, 10))
        plt.imshow(sliced_data, origin='lower', cmap='gray', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
        plt.colorbar(label='Pixel Value')
        plt.title('FITS Image with locoo Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.scatter(locoo[0], locoo[1], color='red', marker='o', label='locoo Points')
        plt.legend()
        plt.savefig(f'')
        plt.show()
        """

        try:
            popt, _ = curve_fit(
                moffat_1d_fixed_offset,
                distances.ravel(),
                sliced_data.ravel(),
                p0=initial_guess,
                absolute_sigma=True,
                bounds=(
                    [0, 0.1, 0.1],  # パラメータの下限
                    [np.inf, 10, 10]  # パラメータの上限
                )
            )
        except:
            #print('failed')
            continue
        
        #if 'haoff' in fitsname:
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, intensities, s=1, alpha=0.5, label="Data")
        if not np.isnan(popt).any():
            fit_x = np.linspace(min(distances), max(distances), 500)
            #fit_y = gaussian(fit_x, *popt)
            #fit_y = lorentzian(fit_x, *popt)
            fit_y = moffat_1d(fit_x,offset=offset, *popt)
            plt.plot(fit_x, fit_y, color="gray", label="Fit")
        plt.xlabel("Distance from Centroid (pixels)")
        plt.ylabel("Pixel Intensity")
        plt.title(f"{fitsname} Slice {i+1}, {stack_glcoo[index1]}")
        plt.legend()
        plt.grid()
        #plt.savefig('p0.png')
        plt.show()
        """
        
        popt_list.append(popt)
        distances_list.append(distances)
        intensities_list.append(intensities)

    #sorted_data = sorted(zip(popt_list, distances_list, intensities_list), key=lambda x: x[0][0])
    filterd_data = [item for item in zip(popt_list, distances_list, intensities_list) if item[0][0] <= satcount[fitsname[:5]]]
    sorted_data = sorted(filterd_data, key=lambda x: x[0][0])
    sorted_popts, sorted_distances, sorted_intensities = zip(*sorted_data)

    #print(f'aaaaaaaaaa{len(stack_data)} {len(stack_locoo)}')
    #print(sorted_popts)

    if len(sorted_popts) >= 5:
        fwhms = [2 * popt[1] * np.sqrt(2**(1/popt[2]) - 1) for popt in sorted_popts[-5:]]
        median_fwhm = statistics.median(fwhms)
        plot_fit(fitsname, sorted_popts[-5:], sorted_distances[-5:], sorted_intensities[-5:], offset, outname)
    elif len(sorted_popts)==0:
        print('err calc_fwhm')
        median_fwhm = None
    else:
        fwhms = [2 * popt[1] * np.sqrt(2**(1/popt[2]) - 1) for popt in sorted_popts]
        median_fwhm = statistics.median(fwhms)
        plot_fit(fitsname, sorted_popts, sorted_distances, sorted_intensities, offset, outname)
        
    return median_fwhm


def plot_fit(fitsname, popt_list, distances_list, intensities_list, offset, outname=None):
    #print('ok')
    aaaaaaa=1
    """
    for index, (popt, distances, intensities) in enumerate(zip(popt_list, distances_list, intensities_list)):
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, intensities, s=1, alpha=0.5, label="Data")
        if not np.isnan(popt).any():
            fit_x = np.linspace(min(distances), max(distances), 500)
            #fit_y = gaussian(fit_x, *popt)
            #fit_y = lorentzian(fit_x, *popt)
            fit_y = moffat_1d(fit_x,offset=offset, *popt)
            plt.plot(fit_x, fit_y, color="gray", label="Fit")
        plt.title('FITS Image with locoo Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(f'fwhm{index+1}_{fitsname}.png')
        #plt.show()
        plt.close()
    """
    num_plots = len(popt_list)
    cols = 3  # 1行に表示するプロットの数
    rows = (num_plots + cols - 1) // cols  # 必要な行数を計算
    
    # サブプロットを作成
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # 配列として扱えるようにする

    for index, (popt, distances, intensities, ax) in enumerate(zip(popt_list, distances_list, intensities_list, axes)):
        ax.scatter(distances, intensities, s=1, alpha=0.5, label="Data")
        if not np.isnan(popt).any():
            fit_x = np.linspace(min(distances), max(distances), 500)
            fit_y = moffat_1d(fit_x, offset=offset, *popt)
            ax.plot(fit_x, fit_y, color="gray", label="Fit")
        ax.set_title(f'Plot {index + 1}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Intensity')
        ax.legend()

    # 不要な空のプロットを削除
    for ax in axes[num_plots:]:
        ax.remove()

    # 画像として保存
    plt.tight_layout()
    if outname == None:
        #plt.savefig(f'fwhm_{fitsname}.png')
        pass
    else:
        plt.savefig(outname)
    plt.close()

def gattyanko(first_txdflist, second_txdflist, fwhm1, fwhm2):

    for first_txdf, second_txdf in zip(first_txdflist, second_txdflist):

        outputfile = f'{first_txdf[:-7]}_{second_txdf[:-7]}.txt'

        with open(first_txdf, 'r') as f1:
            lines1 = f1.readlines()
        with open(second_txdf, 'r') as f2:
            lines2 = f2.readlines()

        first_data = {}
        for line in lines1:
            parts = line.split()
            id_num = int(parts[0])
            xcenter = float(parts[1]) if parts[1] != 'INDEF' else 'INDEF'
            ycenter = float(parts[2]) if parts[2] != 'INDEF' else 'INDEF'
            mag = parts[3] if parts[3] != 'INDEF' else 'INDEF'
            merr = parts[4] if parts[4] != 'INDEF' else 'INDEF'
            first_data[id_num] = (xcenter, ycenter, mag, merr)

        second_data = {}
        for line in lines2:
            parts = line.split()
            id_num = int(parts[0])
            xcenter = float(parts[1]) if parts[1] != 'INDEF' else 'INDEF'
            ycenter = float(parts[2]) if parts[2] != 'INDEF' else 'INDEF'
            mag = parts[3] if parts[3] != 'INDEF' else 'INDEF'
            merr = parts[4] if parts[4] != 'INDEF' else 'INDEF'
            second_data[id_num] = (xcenter, ycenter, mag, merr)


        with open(outputfile, 'w') as out:
            out.write(f"\n#1st file: {first_txdf} fwhm={fwhm1}\n#2nd file: {second_txdf} fwhm={fwhm2}\n\n")
            out.write("#ID Xcenter1 Ycenter1 Mag1 Merr1  Xcenter2 Ycenter2 Mag2 Merr2\n\n")

            # ID 順にデータを並べて書き出す
            all_ids = sorted(set(first_data.keys()).union(second_data.keys()))
            for id_num in all_ids:
                # first のデータ
                if id_num in first_data:
                    x1, y1, mag1, merr1 = first_data[id_num]
                else:
                    x1, y1, mag1, merr1 = 'INDEF', 'INDEF', 'INDEF', 'INDEF'

                # second のデータ
                if id_num in second_data:
                    x2, y2, mag2, merr2 = second_data[id_num]
                else:
                    x2, y2, mag2, merr2 = 'INDEF', 'INDEF', 'INDEF', 'INDEF'

                # 一行にまとめて書き込む
                out.write(f"{id_num:3}  {x1:>8} {y1:>8} {mag1:>7} {merr1:>5}  {x2:>8} {y2:>8} {mag2:>7} {merr2:>5}\n")


def genkai(txdflist, fwhm):

    for txdf in txdflist:
        outputfile = f'{txdf[:-7]}_GENKAI.txt'
        with open(txdf, 'r') as f1:
            lines1 = f1.readlines()
        first_data = {}
        for line in lines1:
            parts = line.split()
            id_num = int(parts[0])
            xcenter = float(parts[1]) if parts[1] != 'INDEF' else 'INDEF'
            ycenter = float(parts[2]) if parts[2] != 'INDEF' else 'INDEF'
            mag = parts[3] if parts[3] != 'INDEF' else 'INDEF'
            merr = parts[4] if parts[4] != 'INDEF' else 'INDEF'
            first_data[id_num] = (xcenter, ycenter, mag, merr)
        with open(outputfile, 'w') as out:
            out.write(f"\n#file: {txdf} fwhm={fwhm}\n\n")
            out.write("#ID Xcenter1 Ycenter1 Mag1 Merr1\n\n")

            # ID 順にデータを並べて書き出す
            all_ids = sorted(first_data.keys())
            for id_num in all_ids:
                # first のデータ
                if id_num in first_data:
                    x1, y1, mag1, merr1 = first_data[id_num]
                else:
                    x1, y1, mag1, merr1 = 'INDEF', 'INDEF', 'INDEF', 'INDEF'


                # 一行にまとめて書き込む
                out.write(f"{id_num:3}  {x1:>8} {y1:>8} {mag1:>7} {merr1:>5}\n")





def main(path_2_matrix):
    pixscale = {
        'haoff':param.pixscale_haoff, 'haon_':param.pixscale_haon_,
        }
    satcount = {
        'haoff':param.haoff_satcount, 'haon_':param.haon__satcount,
        }
    
    if isinstance(fits_pattern, str):
        filtered_files = glob_pattern(fits_pattern)
    else:
        filtered_files = glob_pattern()

    for key1, list1 in filtered_files.items():
        print(f'\n-----fitsname {list1}-----')
        first_fits  = list1[0]
        second_fits = list1[1]
        print('[on-off process]')
        print('do starfind')
        results1 = sta.starfind_center3([first_fits], pixscale[first_fits[:5]], satcount[first_fits[:5]], [4, 5, 1], 1000, 2000, 1, enable_progress_bar=False)
        starnum         = results1[0][0]
        first_coof      = results1[1][0]
        threshold_lside = results1[2][0]
        matrix = np.load(path_2_matrix)

        print('adapt matrix')
        second_coof = adapt_matrix(first_coof, matrix, f'{second_fits[:-5]}_DETOTH.coo')

        print('refine center')
        second_coof = refine_center_2d(second_fits, second_coof)

        print('calc fwhm')
        fwhm1 = calc_fwhm(first_fits, first_coof, f'fwhm_{first_fits}.png')
        fwhm2 = calc_fwhm(second_fits, second_coof, f'fwhm_{second_fits}_DETOTH.png')

        if isinstance(force_fwhm_on, (int, float)):
            fwhm1 = force_fwhm_on
        if isinstance(force_fwhm_of, (int, float)):
            fwhm2 = force_fwhm_of
        if None in (fwhm1, fwhm2):
            print('fwhmerr')
            sys.exit()

        print('do phot')
        magf1 = do_phot([first_fits],  [f'{first_fits[:-5]}.coo'], satcount[first_fits[:5]], fwhm1, coef_annulus_on, coef_dannulus_on)
        magf2 = do_phot([second_fits], [f'{second_fits[:-5]}_DETOTH.coo'], satcount[second_fits[:5]], fwhm2, coef_annulus_of, coef_dannulus_of)
        if not os.path.exists(magf1[0]):
            print(f"ファイル {magf1[0]} が存在しません。")
            continue
        if not os.path.exists(magf2[0]):
            print(f"ファイル {magf2[0]} が存在しません。")
            continue

        print('do txdump')
        txdf1 = do_txdump(magf1)
        txdf2 = do_txdump(magf2)
        gattyanko(txdf1, txdf2, fwhm1, fwhm2)

        #ここまでが on - off
        print('[off LMT-mag process]')
        print('do starfind')
        results1 = sta.starfind_center3([second_fits], pixscale[second_fits[:5]], satcount[second_fits[:5]], [4, 5, 1], 1000, 2000, 3, enable_progress_bar=False)
        starnum         = results1[0][0]
        second_coof      = results1[1][0]
        threshold_lside = results1[2][0]

        print('calc fwhm')
        fwhm3 = calc_fwhm(second_fits, second_coof, f'fwhm_{second_fits}.png')
        if isinstance(force_fwhm_of, (int, float)):
            fwhm3 = force_fwhm_of
        if fwhm3 == None:
            print('fwhmerr')
            sys.exit()

        print('do phot')
        magf3 = do_phot([second_fits], [f'{second_fits[:-5]}_DETOTH.coo'], satcount[second_fits[:5]], fwhm3, coef_annulus_of, coef_dannulus_of)
        if not os.path.exists(magf3[0]):
            print(f"ファイル {magf3[0]} が存在しません。")
            continue
        print('do txdump')
        txdf3 = do_txdump(magf3)
        genkai(txdf3, fwhm3)

        
        print('end')
    print()

    #subprocess.run(f'rm {param.work_dir}/*.coo', shell=True, stderr=subprocess.DEVNULL)
    #subprocess.run(f'rm {param.work_dir}/*.mag.1', shell=True, stderr=subprocess.DEVNULL)


    


if __name__ == "__main__":

    argvs = sys.argv
    argc = len(argvs)
    

    if argc == 3:
        
        day = argvs[2]
        objname = argvs[1]
        param = Ha_main.readparam(argvs[2], argvs[1])
        objparam = Ha_main.readobjfile(param, argvs[1])
        path = os.path.join(param.work_dir)
        iraf.chdir(path)
        if not os.path.exists(matrix_path):
            print(f'not exists {matrix_path}')
            sys.exit()
        main(matrix_path)


    else:
        print(f'usage ./{argvs[0]} [object name] [YYMMDD]')