#!/opt/anaconda3/envs/p11/bin/python3

import sys
import os
import re
import glob
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

FONT_TYPE = "meiryo"


class ExecuteFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, fg_color="transparent", *args, **kwargs)
        self.fonts = (FONT_TYPE, 15)
        self.setup_form()

    def setup_form(self):
        self.execform = ExecForm(master=self)
        self.execform.grid(row=0, column=0, columnspan=2, padx=20, pady=(20,10), sticky="w")

        


class ExecForm(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.fonts = (FONT_TYPE, 15)
        self.param = Ha_main.readparam()
        self.executing = False
        self.setup_form()

    def setup_form(self):
        self.objcts = self.get_objnames(self.param)
        self.dates  = self.get_unique_date(self.param)
        self.combox_objects = ctk.CTkComboBox(master=self, values=self.objcts)
        self.combox_date    = ctk.CTkComboBox(master=self, values=self.dates, width=100)
        self.combox_objects.grid(row=0, padx=(10,5), pady=10, column=0, sticky="w")
        self.combox_date.grid(row=0, padx=(5,10), pady=10, column=1, sticky="w")

        self.add_event_button  = ctk.CTkButton(master=self, command=self.AddEvent, text="add queue list", font=self.fonts, width=110)
        self.exec_event_button = ctk.CTkButton(master=self, command=self.toggle, text="▶︎ satrt queue",
                                            fg_color="green", hover_color="green", font=self.fonts, width=100)
        self.add_event_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.exec_event_button.grid(row=0, column=3, padx=10, pady=10, sticky="w")

    def get_objnames(self, param):
        obj_dir = glob.glob(os.path.join(param.objfile_dir, '*'))
        objfiles = [os.path.basename(path) for path in obj_dir if os.path.isfile(path)]
        objfiles.sort()
        return objfiles
    
    def get_unique_date(self, param):
        optvarr0 = param.rawdata_opt.split('/{date}')[0]
        infvarr0 = param.rawdata_infra.split('/{date}')[0]
        optvarr1 = re.sub(r"\{.*?\}", "\*", optvarr0)
        infvarr1 = re.sub(r"\{.*?\}", "\*", infvarr0)
        optvarr2 = glob.glob(optvarr1)
        infvarr2 = glob.glob(infvarr1)
        optvarr3 = [
            name for path in optvarr2 
            for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        infvarr3 = [
            name for path in infvarr2 
            for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        unique_date = list(set(optvarr3) | set(infvarr3))
        unique_date.sort()
        unique_date.append('All')
        return unique_date
    

class 


def read_coofile(infile):
        with open(infile, 'r') as file:
            flines = file.readlines()
        lines = [line.strip().split() for line in flines if not line.startswith('#')]
        coords = np.array([[float(line[0])-1, float(line[1])-1] for line in lines])  # coox, cooy,
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


def do_starfind(fitslist):
    starnumlist, coordsfilelist, iterate = sta.starfind_center3(
        fitslist, param, [2.5, 5.5, 1], 0, 1000
        )
    return coordsfilelist


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


def do_phot(fitslist, coofilelist, fwhm=5):
    bottom_a.setparam()
    magf_list = []
    for fits, coof in zip(fitslist, coofilelist):
        sigma = bottom_a.skystat(fits, 'stddev')
        bottom_a.phot(fits, fwhm, sigma)
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

"""
def main():
    #point
    fits_pattern = "*.fits"
    on_list  = glob.glob(f'haon{fits_pattern}')
    off_list = glob.glob(f'haof{fits_pattern}')
    on_list.sort()
    off_list.sort()

    imexfile = glob.glob("*homo*")
    if len(imexfile) != 1:
        print('imexfile not exist.')
        sys.exit()
    print(imexfile)
    points_src, points_dst = read_imexam_results(imexfile[0])
    H_array = estimate_homography(points_src, points_dst)

    print(H_array)

    oncooflist = do_starfind(on_list)
    ofcooflist = do_estcoords(oncooflist, H_array)

    #print(f'onlist{on_list}')
    #print(f'oncoof{oncooflist}')
    #print(f'offlist{off_list}')
    #print(f'offcoof{ofcooflist}')

    do_phot(on_list, oncooflist)
    do_phot(off_list, ofcooflist)

    magfilelist = glob.glob("*mag*")
    do_txdump(magfilelist)

    offtxlist = glob.glob(f'haoff*.txdump')
    ontxlist  = glob.glob(f'haon_*.txdump')
    offtxlist.sort()
    ontxlist.sort()

    do_plot(offtxlist, ontxlist)
"""


def glob_pattern():
    fitslist = glob.glob("*.fits")
    grouped_files = {}
    for fitsname in fitslist:
        parts = fitsname.split(day)
        if len(parts)==1:
            continue
        fits_id = parts[1].replace(".fits", "")

        if not fits_id in grouped_files:
            grouped_files[fits_id] = [None, None]

        if 'haoff' in parts[0]:
            grouped_files[fits_id][0] = fitsname
        elif 'haon' in parts[0]:
            grouped_files[fits_id][1] = fitsname

    filtered_files = {k: v for k, v in grouped_files.items() if all(v)}

    return filtered_files


def adapt_matrix(coordsfile, matrix, outputfile):
    with open(coordsfile, 'r') as f1:
        lines1 = f1.readlines()
    coords1 = np.array([line1.split() for line1 in lines1])
    coords_with_bias = np.hstack([coords1, np.ones((coords1.shape[0], 1))])
    transformed_coords = coords_with_bias @ matrix.T
    with open(outputfile, 'w') as f_out:
        for coord in transformed_coords:
            f_out.write(" ".join(map(str, coord)) + "\n")
    return outputfile


def moffat(x, A, alpha, beta, offset):
    return A * (1 + ((x) / alpha) ** 2) ** (-beta) + offset

# offset を固定値にするためのラップ関数
def moffat_with_fixed_offset(x, A, alpha, beta, offset):
    return A * (1 + ((x) / alpha) ** 2) ** (-beta) + offset

def moffatfit(distances, intensities, sigma, offset_fixed):
    # offset を固定した関数を定義
    def moffat_fixed(x, A, alpha, beta):
        return moffat_with_fixed_offset(x, A, alpha, beta, offset_fixed)

    # 初期推定値
    initial_guess = [max(intensities), np.std(distances), 2]
    
    # フィッティング
    popt, pcov = curve_fit(
        moffat_fixed, distances, intensities, sigma=sigma, p0=initial_guess, absolute_sigma=True
    )
    return popt, pcov


def moffatf_fwhm(stack_data, stack_coo, fitsname):
    results = []
    for i, slice_data in enumerate(stack_data):
        coo_x, coo_y = stack_coo[i]

        # ローカルなスライス座標における重心の座標を計算
        local_center_x = coo_x - int(coo_x) + 8
        local_center_y = coo_y - int(coo_y) + 8

        distances = []
        intensities = []
        sigmas = []

        # 各ピクセルの重心からの距離を計算
        #for y in range(slice_data.shape[0]):
        for y in range(slice_data.shape[0]):
            for x in range(slice_data.shape[1]):
                distance = np.sqrt((x - local_center_x)**2 + (y - local_center_y)**2)
                intensity = slice_data[y, x]
                distances.append(distance)
                intensities.append(intensity)
                sigmas.append(1 / (distance + 10))

        distances = np.array(distances)
        intensities = np.array(intensities)
        peak = intensities.max()
        sigmas = np.array(sigmas)
        index0 = int(len(intensities)/4)
        offset = np.median(intensities[-index0:])

        try:
            popt, pcov = moffatfit(distances, intensities, sigmas, offset)
            fit_model = moffat(distances, offset=offset, *popt)
            residual = np.sum((intensities - fit_model) ** 2)
            fwhm = 2 * np.sqrt(2**(1/popt[2]) - 1) * popt[1]
            results.append((i + 1, stack_coo[i], residual, fwhm, popt, peak))
        except RuntimeError:
            print(f"Slice {i+1}: フィッティングに失敗しました。")
            popt = [np.nan, np.nan, np.nan, np.nan]

    #filtered_results = [item for item in results if item[4] <= threshold]
    sorted_results1 = sorted(results, key=lambda x: x[5])[-5:]
    #sorted_results = sorted(sorted_results1, key=lambda x: x[2])[:3]
    sorted_results = sorted(results, key=lambda x: x[5])[-5:]
    for res in sorted_results:
        print(f"スライス {res[0]} | 座標: {res[1]} | 残差: {res[2]:.2f} | FWHM: {res[3]:.2f}")

    for res in sorted_results:
        idx = res[0] - 1
        slice_data = stack_data[idx]
        distances, intensities = [], []
        for y in range(slice_data.shape[0]):
            for x in range(slice_data.shape[1]):
                distances.append(np.sqrt((x - local_center_x)**2 + (y - local_center_y)**2))
                intensities.append(slice_data[y, x])

        distances, intensities = np.array(distances), np.array(intensities)
        index0 = int(len(intensities)/4)
        offset = np.median(intensities[-index0:])
        fit_x = np.linspace(min(distances), max(distances), 500)
        fit_y = moffat(fit_x, offset=offset, *res[4])
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, intensities, s=1, alpha=0.5, label="Data")
        plt.plot(fit_x, fit_y, color="gray", label="Fit")
        plt.title(f"{fitsname} Slice {res[0]}: FWHM={res[3]:.2f}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{fitsname}_slice_{res[0]}.png")
        #plt.show()
    
    fwhms = [varr[3] for varr in sorted_results]
    med_fwhm = sorted(fwhms)[int(len(fwhms)/2)]

    return med_fwhm


def calc_fwhm(fitsname, coordsfile):
    data = fits.getdata(fitsname)
    coords = read_coofile(coordsfile)
    slice_size = (17, 17)
    num_slices = len(coords)
    stack_data = np.zeros((num_slices, *slice_size), dtype=data.dtype)

    stack_coo  = []
    valid_slice_count = 0
    for coo in coords:
        x, y = int(coo[0]), int(coo[1])
        x_start, x_end = x - 8, x + 9
        y_start, y_end = y - 8, y + 9
        # スライスが画像の範囲内に収まっているかチェック
        if x_start < 0 or y_start < 0 or x_end > data.shape[1] or y_end > data.shape[0]:
            print(f"Skipping coordinates ({x}, {y}) - slice out of bounds.")
            continue

        stack_data[valid_slice_count] = data[y_start:y_end, x_start:x_end]
        stack_coo.append(coo)
        valid_slice_count += 1
    stack_data = stack_data[:valid_slice_count]
    fwhm = moffatf_fwhm(stack_data, stack_coo, fitsname)

    return fwhm


def gattyanko(first_txdf, second_txdf, outputfile):
    with open(first_txdf, 'r') as f1:
        lines1 = f1.readlines()
    with open(second_txdf, 'r') as f2:
        lines2 = f2.readlines()

    first_data = {}
    for line in lines1:
        parts = line.split()
        id_num = int(parts[0])
        xcenter = float(parts[1])
        ycenter = float(parts[2])
        mag = parts[3] if parts[3] != 'INDEF' else 'INDEF'
        merr = parts[4] if parts[4] != 'INDEF' else 'INDEF'
        first_data[id_num] = (xcenter, ycenter, mag, merr)

    second_data = {}
    for line in lines2:
        parts = line.split()
        id_num = int(parts[0])
        xcenter = float(parts[1])
        ycenter = float(parts[2])
        mag = parts[3] if parts[3] != 'INDEF' else 'INDEF'
        merr = parts[4] if parts[4] != 'INDEF' else 'INDEF'
        second_data[id_num] = (xcenter, ycenter, mag, merr)


    with open(outputfile, 'w') as out:
        out.write(f"\n#First file: {first_txdf}, Second file: {second_txdf}\n")
        out.write("#ID Xcenter1 Ycenter1 Mag1 Merr1 Xcenter2 Ycenter2 Mag2 Merr2\n")

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
            out.write(f"{id_num} {x1} {y1} {mag1} {merr1} {x2} {y2} {mag2} {merr2}\n")



def main(path_2_matrix, fwhm1, fwhm2):
    pixscale = {
        'haoff':param.pixscale_haoff, 'haon_':param.pixscale_haon_,
        }
    satcount = {
        'haoff':param.haoff_satcount, 'haon_':param.haon__satcount,
        }
    filtered_files = glob_pattern()
    for key1, list1 in filtered_files.items():
        first_fits  = list1[0]
        second_fits = list1[1]
        results1 = sta.starfind_center3([first_fits], pixscale[first_fits[:5]],satcount[first_fits[:5]], enable_progress_bar=False)
        starnum         = results1[0][0]
        first_coof      = results1[1][0]
        threshold_lside = results1[2][0]
        matrix = np.load(path_2_matrix)
        second_coof = adapt_matrix(first_coof, matrix, f'{second_fits[:-5]}.coo')
        fwhm1 = calc_fwhm(first_fits, first_coof)
        fwhm2 = calc_fwhm(second_fits, second_coof)
        magf1 = do_phot([first_fits],  [f'{first_fits[:-5]}.coo'],  fwhm1)
        magf2 = do_phot([second_fits], [f'{second_fits[:-5]}.coo'], fwhm2)
        if not os.path.exists(magf1[0]):
            print(f"ファイル {magf1[0]} が存在しません。")
            continue
        if not os.path.exists(magf2):
            print(f"ファイル {magf2} が存在しません。")
            continue
        txdf1 = do_txdump(magf1)
        txdf2 = do_txdump([magf2])
        gattyanko(txdf1, txdf2, f'{txdf1[:-7]}_{txdf2[-7]}.txt')



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CustomTkinter with Matplotlib")
        self.geometry("1200x800")
        self.mainframe = ctk.CTkScrollableFrame(self, width=300, height=200, fg_color="transparent")
        self.mainframe.pack(pady=20, padx=20, fill="both", expand=True)

        # プロットを保持するフレームを作成
        self.gsframe = GetSampleFrame(self.mainframe, width=800, height=600)
        self.gsframe.grid(row=3, column=0, padx=50, pady=20, sticky="nsew")
        self.rfframe = ReadFileFrame(master=self.mainframe, window=self)
        self.rfframe.grid(row=0, column=0, pady=10, padx=20, sticky="ew")


        # 重心検出を実行するボタンを配置
        self.cent_button = ctk.CTkButton(self.mainframe, text="重心検出を実行", command=self.gsframe.perform_centroid_detection)
        self.cent_button.grid(row=1, column=0, pady=10, padx=20, sticky="ew")

        self.homographframe = CalcHomographyFrame(master=self.mainframe, window=self)
        self.homographframe.grid(row=2, column=0, pady=10, padx=20, sticky="ew")

        self.tsframe = TestMatrixFreame(self.mainframe, width=800, height=600)
        self.tsframe.grid(row=6, column=0, padx=50, pady=(10, 50), sticky="nsew")
        self.rfframe2 = ReadFileFrame2(master=self.mainframe, window=self)
        self.rfframe2.grid(row=4, column=0, pady=(50, 10), padx=20, sticky="ew")

        self.exectframe = ExecTestFrame(master=self.mainframe, window=self)
        self.exectframe.grid(row=5, column=0, pady=10, padx=20, sticky="ew")

        # ウィンドウのグリッドレイアウト設定
        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_rowconfigure(1, weight=0)
        self.mainframe.grid_columnconfigure(0, weight=1)

    
    


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
        main()


    else:
        print('usage1 ./ku1mV.py [OBJECT file name] ')
        print('usage2 ./ku1mV.py [object name][YYMMDD]')