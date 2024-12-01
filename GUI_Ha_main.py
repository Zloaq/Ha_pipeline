#!/opt/anaconda3/envs/p11/bin/python3

import tkinter as tk
import customtkinter as ctk
import os
import re
import glob
import queue
import time
import threading
import subprocess

import Ha_main


task_queue = queue.Queue()
lock = threading.Lock()

FONT_TYPE = "meiryo"


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("black")
        ctk.set_default_color_theme("blue")
        self.geometry("800x700")
        self.fonts = (FONT_TYPE, 15)
        self.manu_font = ctk.CTkFont(family='Yu Gothic', size=14, weight='bold')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.csv_filepath = None
        self.main_frame_dict = {'ExecuteFrame':ExecuteFrame, 'Edit_advanced_a.param':EditAdvancedParamFrame}
        self.setup_form()

    def setup_form(self):
        self.main_frames = [func(master=self) for func in self.main_frame_dict.values()]
        self.menu_frame =  MenuFrame(master=self, main_frame_dict=self.main_frame_dict)
        self.menu_frame.grid(row=0, column=0, sticky="nsew")


    def read_param(self):
        current_dir = os.path.abspath(os.path.dirname(__file__))




class MenuFrame(ctk.CTkFrame):
    def __init__(self, master, main_frame_dict, *args, **kwargs):
        super().__init__(master, corner_radius=0, *args, **kwargs)
        self.font1 = ctk.CTkFont(family='Yu Gothic', size=16, weight='bold')
        self.font2 = ctk.CTkFont(family='Yu Gothic', size=14, weight='bold')
        self.menu_list = list(main_frame_dict.keys())
        self.main_frame_dict = main_frame_dict
        self.setup()
    
    def setup(self):
        self.menu_label = ctk.CTkLabel(self, text="■ Pipeline menu", font=self.font1)
        self.menu_label.grid(row=0, column=0, padx=(10,20), pady=20)
        self.menu_button = [ctk.CTkButton(self, corner_radius=0, height=38, border_spacing=6, text=f" {self.menu_list[i]} >",
                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                            anchor="w", font=self.font2, command=lambda i=i: self.select_frame_by_name(i))  # lambdaを使用
                            for i in range(len(self.menu_list))]
        #print(f'self.menu_button {len(self.menu_button)}')
        for i in range(len(self.menu_list)):
            self.menu_button[i].grid(row=i+1, column=0, sticky="ew")
        self.menu_button[0].configure(fg_color=("gray75", "gray25"))
        self.master.main_frames[0].grid(row=0, column=1, sticky="nsew")

    def select_frame_by_name(self, name):
        # set button color for selected button
        for i in range(len(self.menu_list)):
            self.menu_button[i].configure(fg_color=("gray75", "gray25") if name == i else "transparent")
        # show selected frame
        for i, frame in enumerate(self.master.main_frames):
            if name == i:
                frame.grid(row=0, column=1, sticky="nsew")
            else:
                frame.grid_forget()


class ExecuteFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, fg_color="transparent", *args, **kwargs)
        self.fonts = (FONT_TYPE, 15)
        self.setup_form()

    def setup_form(self):
        self.execform = ExecForm(master=self)
        self.execform.grid(row=0, column=0, columnspan=2, padx=20, pady=(20,10), sticky="w")

        self.mpbutton = OnOffButton(master=self, parameter_dict=read_paramfile1(), header_name='なんだそれ')
        self.mpbutton.grid(row=1, column=0, padx=(20, 5), pady=10, sticky="nsw")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, minsize=250)
        
        self.w_ueuelist = Queueue(master=self)
        self.w_ueuelist.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="nsw")
        self.grid_columnconfigure(1, weight=1)


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
    
    def start_thread(self):
        self.Qthread = threading.Thread(target=self.ExecEvent)
        self.Qthread.start()
        
    def toggle(self):
        if self.executing:
            self.exec_event_button.configure(text="▶︎ satrt queue", fg_color="green", hover_color="green")
            self.executing = False
            
        else:
            self.exec_event_button.configure(text="⏹ stop queue", fg_color="#8B0000", hover_color="#8B0000")
            self.executing = True
            self.start_thread()
    
    def AddEvent(self):
        selected_object = self.combox_objects.get()
        selected_date   = self.combox_date.get()
        if selected_date == 'All':
            for selected_date0 in self.dates:
                if selected_date0 == 'All':
                    continue
                task_queue.put(['./Ha_main.py', selected_object, selected_date0])
        else:
                task_queue.put(['./Ha_main.py', selected_object, selected_date])

    def ExecEvent(self):
        while self.executing:
            task = task_queue.get()
            subprocess.run(task)


class Queueue(ctk.CTkScrollableFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, width=300, *args, **kwargs)
        self.label = ctk.CTkLabel(self, text="Queue waiting list")
        self.label.grid(row=0, column=0, sticky="nw")
        self.task_copy = None
        self.update_interval = 1000
        self.get_ueuelist()
        self.schedule_update()

    def get_ueuelist(self):
        with lock:
            self.temp_copy = list(task_queue.queue)

        if self.task_copy == self.temp_copy:
            return
        
        self.task_copy = self.temp_copy

        for label in getattr(self, "labels", []):
            label.destroy()
        for button in getattr(self, "buttons", []):
            button.destroy()

        self.labels = []
        self.buttons = []
        for i, varr in enumerate(self.task_copy):

            label = ctk.CTkLabel(self, corner_radius=0, height=38,
                                 text=f"{varr[0]} {varr[1]} {varr[2]}",
                                 fg_color="transparent", font=("Helvetica", 14))
            label.grid(row=i+1, column=0, sticky="w")
            self.labels.append(label)

            button = ctk.CTkButton(self, text="[×]",width=30, height=28, fg_color="#2F3E46", hover_color="#2F3E46",
                                font=("Arial", 13), command=lambda x=i: self.button_action(x))
            button.grid(row=i+1, column=1, padx=10, sticky="w")
            self.buttons.append(button)

    def button_action(self, index):
        with lock:
            temp_list = []
            while not task_queue.empty():
                temp_list.append(task_queue.get())
            if 0 <= index < len(temp_list):
                temp_list.pop(index)
            for item in temp_list:
                task_queue.put(item)
    
    def schedule_update(self):
        # 一定時間ごとにget_ueuelistを呼び出し
        self.get_ueuelist()
        self.after(self.update_interval, self.schedule_update)


class OnOffButton(ctk.CTkScrollableFrame):
    def __init__(self, master, parameter_dict, header_name, *args, **kwargs):
        super().__init__(master, width=200, *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.parameter_dict = parameter_dict
        self.header_name = header_name
        self.toggle_vars = {}
        self.checkboxes = []
        keys_with_0_or_1 = [key for key, value in parameter_dict.items() if value in (0, 1)]
        for i, varr in enumerate(keys_with_0_or_1):
            toggle_var = tk.IntVar(value=parameter_dict[varr])
            self.toggle_vars[varr] = toggle_var
            
            toggle_switch = ctk.CTkCheckBox(
                master=self,
                text=f"{varr}",
                variable=toggle_var,
                command=lambda key=varr: self.checkbox_changed(key),
                onvalue=1,
                offvalue=0,
                font=("Yu Gothic", 14)
            )
            toggle_switch.grid(row=i, column=0, padx=10, pady=(10, 0), sticky="w")
            self.checkboxes.append(toggle_switch)

    def checkbox_changed(self, key):
        with lock:
            self.parameter_dict[key] = self.toggle_vars[key].get()
            path_program = os.path.abspath(__file__)
            dir_of_program = os.path.dirname(path_program)
            dir1 = os.path.join(dir_of_program, 'main_a.param')
            with open(dir1) as f1:
                lines = f1.readlines()
            new_lines = []
            for line in lines:
                if line.startswith('#') or line.strip() == '':
                    # コメント行や空行はそのままにする
                    new_lines.append(line)
                    continue
                
                varr = line.split('#')
                varr1 = varr[0].split()
                if varr1[0] == key:
                    new_value = self.parameter_dict[key]
                    new_line = f"{key:<15} {new_value}"
                    if len(varr) > 1:
                        new_line += f" #{varr[1]}"
                    new_line += "\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            # 更新後の内容をファイルに書き込む
            with open(dir1, 'w') as f:
                f.writelines(new_lines)

        
            
    def get(self):
        checked_checkboxes = []
        for checkbox in self.checkboxes:
            if checkbox.get() == 1:
                checked_checkboxes.append(checkbox.cget("text"))
        return checked_checkboxes
    
    def save_paramfile(self):
        # パラメータファイルを更新するメソッド
        path_program = os.path.abspath(__file__)
        dir_of_program = os.path.dirname(path_program)
        dir1 = os.path.join(dir_of_program, 'main_a.param')

        with open(dir1, 'w') as f:
            for key, value in self.parameter_dict.items():
                # 行のフォーマット: 変数名 + 値 + コメント（コメントがある場合）
                line = f"{key} {value}\n"
                f.write(line)
    

class CheckProgress(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)



    
class EditAdvancedParamFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, fg_color="transparent", *args, **kwargs)




class NewWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("New Window")
        self.geometry("500x300")
        
        self.label = ctk.CTkLabel(self, text="This is a new window", font=(FONT_TYPE, 15))
        self.label.pack(pady=20)
        
        self.close_button = ctk.CTkButton(self, text="Close", command=self.destroy)
        self.close_button.pack(pady=10)



class paramedit(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Advanced Param")
        self.geometry("500x400")

        self.label = ctk.CTkLabel(self, text="パラメータを編集", font=(FONT_TYPE, 15))
        self.label.pack(pady=10)

        self.textbox = ctk.CTkTextbox(self)
        self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # パラメータファイルを読み込む
        self.load_param_file()

        # ボタンのフレーム
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.back_button = ctk.CTkButton(self.button_frame, text="Back", command=self.destroy)
        self.back_button.pack(side="left", padx=5)

        self.save_button = ctk.CTkButton(self.button_frame, text="Save", command=self.save_param_file)
        self.save_button.pack(side="left", padx=5)

        self.close_button = ctk.CTkButton(self.button_frame, text="Close", command=self.destroy)
        self.close_button.pack(side="left", padx=5)
    
    def load_param_file(self):
        # パラメータファイルをテキストボックスに読み込む
        content = read_paramfile2()
        self.textbox.insert("1.0", content)

    def save_param_file(self):
        # テキストボックスの内容をパラメータファイルに保存
        path_program = os.path.abspath(__file__)
        dir_of_program = os.path.dirname(path_program)
        dir1 = os.path.join(dir_of_program, 'advanced_a.param')
        content = self.textbox.get("1.0", "end-1c")
        with open(dir1, 'w') as f:
            f.write(content)


def read_paramfile1():
    path_program = os.path.abspath(__file__)
    dir_of_program = os.path.dirname(path_program)
    dir1 = os.path.join(dir_of_program, 'main_a.param')
    with open(dir1) as f1:
        lines = f1.readlines()
    param = {}
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        varr = line.split('#')
        varr1 = varr[0].split()
        if varr1[1] == '0' or varr1[1] == '1':
            param[varr1[0]] = int(varr1[1])
        
    return param

def read_paramfile2():
    path_program = os.path.abspath(__file__)
    dir_of_program = os.path.dirname(path_program)
    dir1 = os.path.join(dir_of_program, 'advanced_a.param')
    with open(dir1) as f1:
        cont = f1.read()
    return cont


if __name__ == "__main__":
    app = App()
    app.mainloop()