#!/opt/anaconda3/envs/p11/bin/python3

import sys
import os
import re
import glob
from pyraf import iraf
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
import tkinter as tk
import customtkinter as ctk

import Ha_main
import bottom_a
import starmatch_a as sta



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
    starnumlist, coordsfilelist, iterate = starmatch_a.starfind_center3(
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
        coordsfile      = results1[1][0]
        threshold_lside = results1[2][0]
        matrix = np.load(path_2_matrix)
        adapt_matrix(coordsfile, matrix, f'{second_fits[:-5]}.coo')
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