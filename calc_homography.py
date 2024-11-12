#!/opt/anaconda3/envs/p11/bin/python3

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
import tkinter as tk
from astropy.io import fits
from astropy.visualization import ZScaleInterval

import starmatch_a as strm
import Ha_main as hma



FONT_TYPE = "meiryo"

def calc(data, coords):

    def generate_slices(coords, size=7):
        half_size = size // 2  # 中心からのオフセット
        slices = []
        for x, y in coords:
            x_start = max(0, int(x) - half_size)
            x_stop = int(x) + half_size + 1
            y_start = max(0, int(y) - half_size)
            y_stop = int(y) + half_size + 1
            slices.append((slice(y_start, y_stop), slice(x_start, x_stop)))
        return slices

    def has_converged(coords1, coords2, tolerance=0.5):
        for (x1, y1), (x2, y2) in zip(coords1, coords2):
            if abs(x1 - x2) > tolerance or abs(y1 - y2) > tolerance:
                return False
        return True

    def centroid_roop(data, coords, max_iterations=100):
        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            slices = generate_slices(coords, 7)
            new_coords = clustar_centroid(data, slices, 0)
            converged = has_converged(coords, new_coords)
            coords = new_coords
            iteration += 1
        if not converged:
            print("最大イテレーション数に達しましたが、収束しませんでした。")
        return coords

    coords = centroid_roop(data, coords)

    coords = np.array(coords)

    return coords


def clustar_centroid(data, slices, padding=2):
    max_y, max_x = data.shape

    centroids = []

    for sl in slices:
        y_slice, x_slice = sl

        y_start = max(y_slice.start - padding, 0)
        y_stop = min(y_slice.stop + padding, max_y)
        x_start = max(x_slice.start - padding, 0)
        x_stop = min(x_slice.stop + padding, max_x)

        data_slice = data[y_start:y_stop, x_start:x_stop]

        total = data_slice.sum()
        if total == 0:
            centroids.append((x_start + (x_stop - x_start) / 2, y_start + (y_stop - y_start) / 2))
            continue

        y_indices, x_indices = np.indices(data_slice.shape)
        y_centroid_local = np.sum(y_indices * data_slice) / total
        x_centroid_local = np.sum(x_indices * data_slice) / total
        y_centroid_global = y_centroid_local + y_start
        x_centroid_global = x_centroid_local + x_start
        centroids.append((x_centroid_global, y_centroid_global))

    return centroids



class ReadFileFrame(ctk.CTkFrame):
    def __init__(self, master=None, window=None, *args, header_name="ReadFileFrame", **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.fonts = (FONT_TYPE, 15)
        self.header_name = header_name
        self.selected_dir = os.path.abspath(os.path.dirname(__file__))
        self.file_name = None
        self.window = window
        self.window.param = hma.readparam()

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(0, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # フレームのラベルを表示
        self.label = ctk.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 12))
        self.label.grid(row=0, column=0, padx=20, sticky="w")

        # ファイルパスを指定するテキストボックス。これだけ拡大したときに、幅が広がるように設定する。
        self.textbox = ctk.CTkEntry(master=self, placeholder_text="fits ファイルを読み込む", width=120, font=self.fonts)
        self.textbox.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")

        # ファイル選択ボタン
        self.button_select = ctk.CTkButton(master=self, 
            fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),   # ボタンを白抜きにする
            command=self.button_select_callback, text="ファイル選択", font=self.fonts)
        self.button_select.grid(row=1, column=1, padx=10, pady=(0,10))
        
        # 開くボタン
        self.button_open1 = ctk.CTkButton(master=self, command=self.button_open_callback1, text="plot inp_frame", font=self.fonts)
        self.button_open1.grid(row=1, column=2, padx=10, pady=(0,10))

        self.button_open2 = ctk.CTkButton(master=self, command=self.button_open_callback2, text="plot ref_frame", font=self.fonts)
        self.button_open2.grid(row=1, column=3, padx=10, pady=(0,10))

    def button_select_callback(self):
        """
        選択ボタンが押されたときのコールバック。ファイル選択ダイアログを表示する
        """
        # エクスプローラーを表示してファイルを選択する
        temp = None
        temp = ReadFileFrame.file_read(self.selected_dir)

        if temp is not None:
            self.file_name = temp
            self.selected_dir = os.path.dirname(self.file_name)
            # ファイルパスをテキストボックスに記入
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, os.path.basename(self.file_name))

    def button_open_callback1(self):
        if self.file_name is not None:
            hdu = fits.open(self.file_name)
            self.window.gsframe.img1 = hdu[0].data
            skycount = float(hdu[0].header.get('SKYCOUNT') or 0)
            self.inpfitsname = os.path.basename(self.file_name)
            paramkey = f'{self.inpfitsname[:5]}_satcount'
            satcount = float(getattr(self.window.param, paramkey, 100000))
            
            # Saturation mask
            satmask = self.window.gsframe.img1 > (satcount - skycount)

            # Get the pixel positions where mask is True
            y, x = np.where(satmask)
            self.window.gsframe.ax1.clear()

            # Plot the image
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(self.window.gsframe.img1)
            self.window.gsframe.ax1.set_title(f"{self.inpfitsname}", fontsize=6)
            self.window.gsframe.ax1.tick_params(labelsize=6)
            self.window.gsframe.ax1.imshow(self.window.gsframe.img1, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            
            # Plot the mask points
            self.window.gsframe.ax1.plot(x, y, marker='*', color='red', markersize=0.5, linestyle='None')

            # Redraw the canvas
            self.window.gsframe.fig.canvas.draw()

    def button_open_callback2(self):
        if self.file_name is not None:
            # 画像データの読み込み
            self.window.gsframe.img2 = fits.getdata(self.file_name)
            
            # 例：SKYCOUNT などの取得（必要に応じて）
            hdu = fits.open(self.file_name)
            skycount = float(hdu[0].header.get('SKYCOUNT') or 0)
            self.reffitsname = os.path.basename(self.file_name)
            paramkey = f'{self.reffitsname[:5]}_satcount'
            satcount = float(getattr(self.window.param, paramkey, 1000000))
            
            # サチュレーションマスクを生成
            satmask = self.window.gsframe.img2 > (satcount - skycount)

            # マスクが True の座標を取得
            y, x = np.where(satmask)
            self.window.gsframe.ax2.clear()

            # 画像のプロット
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(self.window.gsframe.img2)
            self.window.gsframe.ax2.set_title(f"{self.reffitsname}", fontsize=6)
            self.window.gsframe.ax2.tick_params(labelsize=6)
            self.window.gsframe.ax2.imshow(self.window.gsframe.img2, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            
            # マスク位置に点をプロット
            self.window.gsframe.ax2.plot(x, y, marker='*', color='red', markersize=0.5, linestyle='None')

            # キャンバスの再描画
            
            self.window.gsframe.fig.canvas.draw()
            
    @staticmethod
    def file_read(selected_dir):
        """
        ファイル選択ダイアログを表示する
        """
        file_path = tk.filedialog.askopenfilename(filetypes=[("fitsファイル","*.fits")],initialdir=selected_dir)

        if len(file_path) != 0:
            return file_path
        else:
            # ファイル選択がキャンセルされた場合
            return None



class GetSampleFrame(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.img1 = np.random.rand(1000, 1000)
        self.img2 = np.random.rand(1000, 1000)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 2.5), facecolor="#A9A9A9")
        self.ax1.tick_params(labelsize=6)
        self.ax2.tick_params(labelsize=6)
        self.ax1.imshow(self.img1, cmap='gray')
        self.ax2.imshow(self.img2, cmap='gray')

        self.ax1.invert_yaxis()
        self.ax2.invert_yaxis()

        # マーカーを保存するリスト
        self.markers = {self.ax1: [], self.ax2: []}
        self.texts = {self.ax1: [], self.ax2: []}
        self.counter = {self.ax1: 0, self.ax2: 0}
        self.coordinates = {self.ax1: [], self.ax2: []}

        # 現在の選択されたマーカー
        self.selected_marker = None
        self.selected_text = None
        self.selected_index = None

        self.make_canvas()

    def make_canvas(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)

        # Matplotlib Figure を Tkinter のキャンバスとして設定
        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def add_marker(self, ax, x, y):
        self.counter[ax] += 1
        # マーカーと番号を追加
        marker = ax.plot(x, y, marker='o', color='orange', markersize=0.7)[0]
        text = ax.text(x + 20, y + 20, str(self.counter[ax]), color="black", fontsize=7, ha='center', va='center', zorder=2)
        self.markers[ax].append(marker)
        self.texts[ax].append(text)
        self.coordinates[ax].append((x, y))  # 座標をリストに追加

    def remove_marker(self, ax):
        if len(self.markers[ax]) > 0:
            marker = self.markers[ax].pop()
            text = self.texts[ax].pop()
            self.coordinates[ax].pop()  # 座標も削除
            marker.remove()
            text.remove()
            self.counter[ax] -= 1

            # 番号の整理
            for i, text in enumerate(self.texts[ax]):
                text.set_text(str(i + 1))

    def perform_centroid_detection(self):
        # 座標と画像データを取得
        refdata = self.img1
        inpdata = self.img2
        refcoords = self.coordinates[self.ax1]
        inpcoords = self.coordinates[self.ax2]

        # calc 関数を使用して重心を計算
        inp_centroids = calc(inpdata, inpcoords)
        ref_centroids = calc(refdata, refcoords)

        # 初期マーカーの位置を更新（ax1）
        for i, (marker, text) in enumerate(zip(self.markers[self.ax1], self.texts[self.ax1])):
            x, y = ref_centroids[i]
            marker.set_data([x], [y])
            text.set_position((x + 20, y + 20))
            self.coordinates[self.ax1][i] = (x, y)

        # 初期マーカーの位置を更新（ax2）
        for i, (marker, text) in enumerate(zip(self.markers[self.ax2], self.texts[self.ax2])):
            x, y = inp_centroids[i]
            marker.set_data([x], [y])
            text.set_position((x + 20, y + 20))
            self.coordinates[self.ax2][i] = (x, y)

        self.fig.canvas.draw()

    def on_key_press(self, event):
        ax = event.inaxes  # カーソルがどの画像上にあるか
        if not ax:
            return

        if event.key == 'a':  # 'a' を押したらマーカーを追加
            self.add_marker(ax, event.xdata, event.ydata)
        elif event.key == 'backspace':  # 'backspace' を押したらマーカーを削除
            self.remove_marker(ax)

        self.fig.canvas.draw()

    def on_button_press(self, event):
        ax = event.inaxes
        if not ax:
            return

        for i, (marker, text) in enumerate(zip(self.markers[ax], self.texts[ax])):
            contains, _ = marker.contains(event)
            if contains:
                self.selected_marker = marker
                self.selected_text = text
                self.selected_index = i  # インデックスを保存
                break

    def on_mouse_move(self, event):
        ax = event.inaxes
        if self.selected_marker and ax:
            self.selected_marker.set_data([event.xdata], [event.ydata])
            self.selected_text.set_position((event.xdata + 20, event.ydata + 20))
            # 座標を更新
            self.coordinates[ax][self.selected_index] = (event.xdata, event.ydata)
            self.fig.canvas.draw()

    def on_button_release(self, event):
        self.selected_marker = None
        self.selected_text = None
        self.selected_index = None  # リセット



class CalcHomographyFrame(ctk.CTkFrame):
    def __init__(self, master=None, window=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.fonts = (FONT_TYPE, 15)
        self.window = window
        self.header_name = "CalcHomographyFrame"

        self.H = np.zeros((2, 3))

        # UI要素のセットアップ
        self.setup_form()

    def setup_form(self):
        # グリッドレイアウトの設定
        #self.grid_rowconfigure(0, weight=1)
        #self.grid_columnconfigure(0, weight=1)

        # フレームのラベル
        self.label = ctk.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 12))
        self.label.grid(row=0, column=0, padx=20, sticky="w")

        self.matrix_text = ctk.CTkLabel(master=self, text=self.H, font=(FONT_TYPE, 11))
        self.matrix_text.grid(row=1, column=1, padx=10, pady=(0, 10))

        # アフィン変換を推定するボタン
        self.button_estimate = ctk.CTkButton(
            master=self,
            command=self.button_estimate_callback,
            text="calc matrix",
            font=self.fonts
        )
        self.button_estimate.grid(row=1, column=0, padx=10, pady=(0, 10))

        #self.save_name_label = ctk.CTkLabel(self, text=f"matrix file name", font=(FONT_TYPE, 11))
        #self.save_name_label.grid(row=0, column=2, padx=10, pady=10)


        self.savename = ctk.CTkEntry(master=self, width=200, placeholder_text="Enter matrix name")
        self.savename.grid(row=1, column=2, padx=10, pady=10)
        
        self.savebutton = ctk.CTkButton(master=self, command=self.saveaffin, text=f'save affin')
        self.savebutton.grid(row=1, column=3, padx=10, pady=10)

        self.save_status_label = ctk.CTkLabel(master=self, text="", font=(FONT_TYPE, 12))
        self.save_status_label.grid(row=1, column=4, columnspan=2, pady=(10, 0))

    def button_estimate_callback(self):
        # gsframeから座標を取得
        inpcoords = self.window.gsframe.coordinates[self.window.gsframe.ax1]
        refcoords = self.window.gsframe.coordinates[self.window.gsframe.ax2]

        #print(f'inpcoords{inpcoords}')
        #print(f'refcoords{refcoords}')
        if hasattr(self.window.rfframe, 'inpfitsname') and hasattr(self.window.rfframe, 'reffitsname'):
            self.inpfitsname = self.window.rfframe.inpfitsname
            self.reffitsname = self.window.rfframe.reffitsname

        # 対応する点の数が一致するか確認
        if len(inpcoords) != len(refcoords):
            print("対応する点の数が一致しません！")
            return

        if len(inpcoords) == 0:
            print("点がねえ")
            return
        
        name = self.recom_name()
        if name:
            print(f'{name[0]}_to_{name[1]}_{name[2]}')
            self.savename.delete(0, "end")  # 既存の内容を削除
            self.savename.insert(0, f'{name[0]}_to_{name[1]}_{name[2]}')  # 新しい内容を挿入
            self.savename.update_idletasks()

        # 座標をNumPy配列に変換
        inpcoords = np.array(inpcoords)
        refcoords = np.array(refcoords)

        # アフィン変換行列を推定
        self.est_affine(inpcoords, refcoords)

        self.matrix_text.configure(
            #text=f"[[{H[0][0]}, {H[0][1]}, {H[0][2]}], [{H[1][0]}, {H[1][1]}, {H[1][2]}], [{H[2][0]}, {H[2][1]}, {H[2][2]}]]")
            text=self.H)
       

    def est_affine(self, inpcoords, refcoords):
        A = []
        b = []
        for (x, y), (x_prime, y_prime) in zip(inpcoords, refcoords):
            # A行列とbベクトルの構築
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            b.append(x_prime)
            b.append(y_prime)
        A = np.array(A)
        b = np.array(b)

        # 最小二乗法で h を求める
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # アフィン変換行列 H を構築
        self.H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]]
        ])
    
    def recom_name(self):
        if not hasattr(self, 'inpfitsname') or not hasattr(self, 'reffitsname'):
            print(1)
            return None
        match = re.search(r'(\d{6}).*\.fits$', self.reffitsname)
        if match:
            print(2)
            return self.inpfitsname[:5], self.reffitsname[:5], match.group(1)
        else:
            print(3)
            return None
        
    def saveaffin(self):
        savename = f'{self.savename.get()}.npy'
        if savename == '.npy':
            self.save_status_label.configure(text="Enter matrix name", text_color="red")
            return
        
        matpath = os.path.join(self.window.param.matrix_dir, savename)
        
        if os.path.isfile(matpath):
            self.save_status_label.configure(text=f'{savename} already exists.', text_color="red")
            return
        
        # 保存処理
        np.save(matpath, self.H)
        self.save_status_label.configure(text=f'{savename}\nsaved successfully!', text_color="green")

        self.window.exectframe.setup_combobox()


class ReadFileFrame2(ctk.CTkFrame):
    def __init__(self, master=None, window=None, *args, header_name="TestMatrixFreame", **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.fonts = (FONT_TYPE, 15)
        self.header_name = header_name
        self.selected_dir = os.path.abspath(os.path.dirname(__file__))
        self.file_name = None
        self.window = window

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):
        # 行方向のマスのレイアウトを設定する。リサイズしたときに一緒に拡大したい行をweight 1に設定。
        self.grid_rowconfigure(0, weight=1)
        # 列方向のマスのレイアウトを設定する
        self.grid_columnconfigure(0, weight=1)

        # フレームのラベルを表示
        self.label = ctk.CTkLabel(self, text=self.header_name, font=(FONT_TYPE, 12))
        self.label.grid(row=0, column=0, padx=20, sticky="w")

        # ファイルパスを指定するテキストボックス。これだけ拡大したときに、幅が広がるように設定する。
        self.textbox = ctk.CTkEntry(master=self, placeholder_text="fits ファイルを読み込む", width=120, font=self.fonts)
        self.textbox.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")

        # ファイル選択ボタン
        self.button_select = ctk.CTkButton(master=self, 
            fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),   # ボタンを白抜きにする
            command=self.button_select_callback, text="ファイル選択", font=self.fonts)
        self.button_select.grid(row=1, column=1, padx=10, pady=(0,10))
        
        # 開くボタン
        self.button_open1 = ctk.CTkButton(master=self, command=self.button_open_callback1, text="plot inp_frame", font=self.fonts)
        self.button_open1.grid(row=1, column=2, padx=10, pady=(0,10))

        self.button_open2 = ctk.CTkButton(master=self, command=self.button_open_callback2, text="plot ref_frame", font=self.fonts)
        self.button_open2.grid(row=1, column=3, padx=10, pady=(0,10))

    def button_select_callback(self):
        """
        選択ボタンが押されたときのコールバック。ファイル選択ダイアログを表示する
        """
        # エクスプローラーを表示してファイルを選択する
        temp = None
        temp = ReadFileFrame.file_read(self.selected_dir)

        if temp is not None:
            self.file_name = temp
            self.selected_dir = os.path.dirname(self.file_name)
            # ファイルパスをテキストボックスに記入
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, os.path.basename(self.file_name))

    def button_open_callback1(self):
        if self.file_name is not None:
            hdu = fits.open(self.file_name)
            self.window.tsframe.fits1 = self.file_name
            self.window.tsframe.img1 = hdu[0].data
            skycount = float(hdu[0].header.get('SKYCOUNT') or 0)
            self.inpfitsname = os.path.basename(self.file_name)
            paramkey = f'{self.inpfitsname[:5]}_satcount'
            satcount = float(getattr(self.window.param, paramkey, 100000))
            
            # Saturation mask
            satmask = self.window.tsframe.img1 > (satcount - skycount)

            # Get the pixel positions where mask is True
            y, x = np.where(satmask)
            self.window.tsframe.ax1.clear()

            # Plot the image
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(self.window.tsframe.img1)
            self.window.tsframe.ax1.set_title(f"{self.inpfitsname}", fontsize=6)
            self.window.tsframe.ax1.tick_params(labelsize=6)
            self.window.tsframe.ax1.imshow(self.window.tsframe.img1, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            
            # Plot the mask points
            self.window.tsframe.ax1.plot(x, y, marker='*', color='red', markersize=0.5, linestyle='None')

            # Redraw the canvas
            self.window.tsframe.fig.canvas.draw()

    def button_open_callback2(self):
        if self.file_name is not None:
            # 画像データの読み込み
            self.window.tsframe.img2 = fits.getdata(self.file_name)
            
            # 例：SKYCOUNT などの取得（必要に応じて）
            hdu = fits.open(self.file_name)
            self.window.tsframe.fits2 = self.file_name
            skycount = float(hdu[0].header.get('SKYCOUNT') or 0)
            self.reffitsname = os.path.basename(self.file_name)
            paramkey = f'{self.reffitsname[:5]}_satcount'
            satcount = float(getattr(self.window.param, paramkey, 1000000))
            
            # サチュレーションマスクを生成
            satmask = self.window.tsframe.img2 > (satcount - skycount)

            # マスクが True の座標を取得
            y, x = np.where(satmask)
            self.window.tsframe.ax2.clear()

            # 画像のプロット
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(self.window.tsframe.img2)
            self.window.tsframe.ax2.set_title(f"{self.reffitsname}", fontsize=6)
            self.window.tsframe.ax2.tick_params(labelsize=6)
            self.window.tsframe.ax2.imshow(self.window.tsframe.img2, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            
            # マスク位置に点をプロット
            self.window.tsframe.ax2.plot(x, y, marker='*', color='red', markersize=0.5, linestyle='None')

            # キャンバスの再描画
            
            self.window.tsframe.fig.canvas.draw()
            
    @staticmethod
    def file_read(selected_dir):
        """
        ファイル選択ダイアログを表示する
        """
        file_path = tk.filedialog.askopenfilename(filetypes=[("fitsファイル","*.fits")],initialdir=selected_dir)

        if len(file_path) != 0:
            return file_path
        else:
            # ファイル選択がキャンセルされた場合
            return None
        

        
class TestMatrixFreame(ctk.CTkFrame):
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.param = hma.readparam()
        self.fits1 = None
        self.fits2 = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.img1 = np.random.rand(1000, 1000)
        self.img2 = np.random.rand(1000, 1000)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 2.5), facecolor="#A9A9A9")
        self.ax1.tick_params(labelsize=6)
        self.ax2.tick_params(labelsize=6)
        self.ax1.imshow(self.img1, cmap='gray')
        self.ax2.imshow(self.img2, cmap='gray')

        self.ax1.invert_yaxis()
        self.ax2.invert_yaxis()

        self.make_canvas()

    def make_canvas(self):
        # Matplotlib Figure を Tkinter のキャンバスとして設定
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def star_marker(self, lside_threshold, minstarnum, maxstarnum, lowest_threshold):
        if hasattr(self, 'markers1'):
            for marker in self.markers1:
                marker.remove()
            #for text in self.texts:
            #    text.remove()
        if not self.fits1:
            return
        paramkey1 = f'pixscale_{os.path.basename(self.fits1)[:5]}'
        #print(paramkey1)
        paramkey2 = f'{os.path.basename(self.fits1)[:5]}_satcount'
        pixscale_value = float(getattr(self.param, paramkey1, None))
        satcount_value = float(getattr(self.param, paramkey2, None))
        starfind_range = [int(lside_threshold), int(lside_threshold) + 20, 2]
        result = strm.starfind_center3(
            [self.fits1], pixscale_value, satcount_value, starfind_range, int(minstarnum), int(maxstarnum), int(lowest_threshold)
            )
        starnumlist         = result[0]
        coordsfilelist      = result[1]
        threshold_lsidelist = result[2]

        with open(coordsfilelist[0], 'r') as f1:
            lines = f1.readlines()

        self.markers1 = [self.ax1.plot(float(x), float(y), marker='o', markerfacecolor='none', color='orange', markersize=4, markeredgewidth=0.2)[0] for line in lines for x, y in [line.split()]]
        self.canvas.draw()
        """
        texts   = [self.ax1.text(float(x) + 20, float(y) + 20, str(index), color="black", fontsize=7, ha='center', va='center', zorder=2)
         for index, line in enumerate(lines) for x, y in [line.split()]]
        """

class ExecTestFrame(ctk.CTkFrame):
    def __init__(self, master=None, window=None, *args, header_name="ExecTestFrame", **kwargs):
        super().__init__(master, *args, **kwargs)
        
        self.fonts = (FONT_TYPE, 13)
        self.header_name = header_name
        self.selected_dir = os.path.abspath(os.path.dirname(__file__))
        self.file_name = None
        self.window = window

        # フォームのセットアップをする
        self.setup_form()

    def setup_form(self):

        # スライドバー
        self.slider = ctk.CTkSlider(self, from_=0, to=50, command=self.on_slider_change)
        self.slider.set(10)  # 初期値を10に設定
        self.slider.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # スライダーのラベル
        self.slider_label = ctk.CTkLabel(self, text=f"Initial threshold {str(self.slider.get())}", font=self.fonts)
        self.slider_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # エントリーボックス1のラベル
        self.label1 = ctk.CTkLabel(self, text="min starnum", font=self.fonts)
        self.label1.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # エントリーボックス1
        self.entry1 = ctk.CTkEntry(self)
        self.entry1.insert(0, "10")  # 初期値を設定
        self.entry1.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # エントリーボックス2のラベル
        self.label2 = ctk.CTkLabel(self, text="max starnum", font=self.fonts)
        self.label2.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # エントリーボックス2
        self.entry2 = ctk.CTkEntry(self)
        self.entry2.insert(0, "100")  # 初期値を設定
        self.entry2.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # エントリーボックス3のラベル
        self.label3 = ctk.CTkLabel(self, text="lowest threshold", font=self.fonts)
        self.label3.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # エントリーボックス3
        self.entry3 = ctk.CTkEntry(self)
        self.entry3.insert(0, "1")  # 初期値を設定
        self.entry3.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # 実行ボタン
        self.sfind_button = ctk.CTkButton(self, text="starfind", font=self.fonts, command=self.sfind_action)
        self.sfind_button.grid(row=2, column=0, padx=10, pady=10)

        # リセットボタン
        self.reset_button = ctk.CTkButton(self, text="adapt matrix", font=self.fonts, command=self.adapt_action)
        self.reset_button.grid(row=2, column=3, padx=10, pady=10)

        self.seematrix_button = ctk.CTkButton(self, text="print matrix", font=self.fonts, command=self.setup_matrixlabel)
        self.seematrix_button.grid(row=2, column=4, padx=10, pady=10)

        self.setup_combobox()
        self.setup_matrixlabel()

    def setup_combobox(self):
        # コンボボックス
        self.matrixes0 = glob.glob(os.path.join(self.window.param.matrix_dir, '*'))
        self.matrixes1 = [os.path.basename(varr) for varr in self.matrixes0]
        self.matrixbox = ctk.CTkComboBox(master=self, values=self.matrixes1)
        self.matrixbox.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky='ew')
        self.matrixbox.bind("<<ComboboxSelected>>", lambda event: self.setup_matrixlabel())

    def setup_matrixlabel(self):
        H = np.load(os.path.join(self.window.param.matrix_dir, self.matrixbox.get()))
        if hasattr(self, 'matrix_text'):
            self.matrix_text.configure(text=H)
            #print(f'if')
        else:
            self.matrix_text = ctk.CTkLabel(master=self, text=H, font=(FONT_TYPE, 11))
            self.matrix_text.grid(row=1, column=4, padx=10, pady=(0, 10))
            #print(f'else')


    # ボタンに割り当てるアクション

    def on_slider_change(self, value):
        self.slider_label.configure(text=f"Initial threshold {int(self.slider.get())}")

    def sfind_action(self):

        lside_threshold  = self.slider.get()
        minstarnum       = self.entry1.get()
        maxstarnum       = self.entry2.get()
        lowest_threshold = self.entry3.get()
        self.window.tsframe.star_marker(lside_threshold, minstarnum, maxstarnum, lowest_threshold)

    def adapt_action(self):
        #print("リセットボタンが押されました")
        if hasattr(self, 'markers2'):
            for marker in self.markers2:
                marker.remove()
        matrixfile = self.matrixbox.get()
        matrix = np.load(os.path.join(self.window.param.matrix_dir, matrixfile))
        coords = np.array([(marker.get_data()[0][0], marker.get_data()[1][0]) for marker in self.window.tsframe.markers1])
        coords_with_bias = np.hstack([coords, np.ones((coords.shape[0], 1))])
        transformed_coords = coords_with_bias @ matrix.T

        x_min, x_max = self.window.tsframe.ax2.get_xlim()
        y_min, y_max = self.window.tsframe.ax2.get_ylim()

        self.markers2 = [
            self.window.tsframe.ax2.plot(float(x), float(y), marker='o', markerfacecolor='none', color='orange', markersize=4, markeredgewidth=0.2)[0]
            for x, y in transformed_coords
            if x_min <= x <= x_max and y_min <= y <= y_max  # 範囲チェック
        ]

        self.window.tsframe.canvas.draw()

    def exit_action(self):
        print("終了ボタンが押されました")

    def get_values(self):
        slider_value = self.slider.get()
        entry1_value = self.entry1.get()
        entry2_value = self.entry2.get()
        entry3_value = self.entry3.get()
        
        print(f"スライドバーの値: {slider_value}")
        print(f"エントリーボックス1の値: {entry1_value}")
        print(f"エントリーボックス2の値: {entry2_value}")
        print(f"エントリーボックス3の値: {entry3_value}")
    



# メインアプリケーション
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
    app = App()
    app.mainloop()


