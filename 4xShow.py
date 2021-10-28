import tkinter as tk
from tkinter.constants import HORIZONTAL
from PIL import Image, ImageTk

from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import os

from codes.test import test_ctunet

class App:
    def __init__(self):
        
        self.p_values = [0,0,0,0]
        self.types = ['cannyedge', 'binary', 'dilation', 'erosion']
        # self.run = test_ctunet(types=self.types)

        # master loop
        self.master = tk.Tk()
        self.master.geometry('875x620')
        self.master['bg'] = 'white'

        # Combo select

        label = ttk.Label(text="Please select module 1")
        # label.pack(fill='x', padx=5, pady=5)

        # comboboxes for selecting modules
        module_names = ['cannyedge', 'binary', 'dilation', 'erosion']
        self.modules = [None]*len(module_names)
        
        for i in range(len(module_names)):
            self.modules[i] = tk.StringVar()
            tk.Label(self.master, text=f'Please select module {i+1}', justify='right').grid(row=0, column=i, padx=5)
            combob1 = ttk.Combobox(self.master, values=module_names, textvariable=self.modules[i], state='readonly').grid( 
                                                                                    row=1, 
                                                                                    column=i,
                                                                                    sticky=tk.W, 
                                                                                    pady=4,
                                                                                    padx = 5)

        # open file dialog
        tk.Button(self.master, 
                text='Cascade Modules', command=self.cascade_modules).grid( row=2, 
                                                                            column=0,
                                                                            sticky=tk.W, 
                                                                            pady=4,
                                                                            padx = 5)


        menu_row = 3
        # open file dialog
        tk.Button(self.master, 
                text='Choose input image', command=self.show_input_image).grid( row=menu_row, 
                                                                                column=0,
                                                                                sticky=tk.W, 
                                                                                pady=4,
                                                                                padx = 5)

        # open file dialog
        tk.Button(self.master, 
                text='Choose GT image', command=self.show_GT_image).grid(   row=menu_row, 
                                                                            column=1,
                                                                            sticky=tk.W, 
                                                                            pady=4,
                                                                            padx = 5)


        # run button
        tk.Button(self.master, 
                text='Run', command=self.run_image).grid(   row=menu_row, 
                                                            column=2, 
                                                            sticky=tk.W, 
                                                            pady=4,
                                                            padx = 5,)

        first_slider_row = 7
        slider_columns = 1
        slider_length = 400
        # slider for 0th p value
        self.slider0 = tk.Scale(self.master, from_=0, to=9, orient=HORIZONTAL, tickinterval=0, length=slider_length)
        self.slider0.bind("<ButtonRelease-1>", self.update0)
        self.slider0.grid(row=first_slider_row, column=slider_columns, padx=0, columnspan=3)
        tk.Label(self.master, text=f"Tuning parameter for module 1:", justify='right').grid(row=first_slider_row, column=0, padx=0)

        # slider for 1st p value
        self.slider1 = tk.Scale(self.master, from_=0, to=9, orient=HORIZONTAL, tickinterval=0, length=slider_length)
        self.slider1.bind("<ButtonRelease-1>", self.update1)
        self.slider1.grid(row=first_slider_row+1, column=slider_columns, padx=0, columnspan=3)
        tk.Label(self.master, text=f"Tuning parameter for module 2:", justify='right').grid(row=first_slider_row+1, column=0, padx=0)

        # slider for 2nd p value
        self.slider2 = tk.Scale(self.master, from_=0, to=9, orient=HORIZONTAL, tickinterval=0, length=slider_length)
        self.slider2.bind("<ButtonRelease-1>", self.update2)
        self.slider2.grid(row=first_slider_row+2, column=slider_columns, padx=0, columnspan=3)
        tk.Label(self.master, text=f"Tuning parameter for module 3:", justify='right').grid(row=first_slider_row+2, column=0, padx=0)

        # slider for 3rd p value
        self.slider3 = tk.Scale(self.master, from_=0, to=9, orient=HORIZONTAL, tickinterval=0, length=slider_length)
        self.slider3.bind("<ButtonRelease-1>", self.update3)
        self.slider3.grid(row=first_slider_row+3, column=slider_columns, padx=0, columnspan=3)
        tk.Label(self.master, text=f"Tuning parameter for module 4:", justify='right').grid(row=first_slider_row+3, column=0, padx=0)

        tk.mainloop()

    def update0(self, x):
        self.p_values[0] = self.slider0.get()
    def update1(self, x):
        self.p_values[1] = self.slider1.get()
    def update2(self, x):
        self.p_values[2] = self.slider2.get()
    def update3(self, x):
        self.p_values[3] = self.slider3.get()

    def run_image(self):

        self.out_img = self.run.run_test(image_path=self.input_image_path, pns=self.p_values)

        render = ImageTk.PhotoImage(self.out_img)
        image = tk.Label(self.master, image=render)
        image.image = render
        image.place(x = 590, y = 320)

        tk.Label(self.master, text="Result image", justify='center').place(x = 680, y = 590)

    # Open and show original image
    def show_input_image(self):
        self.input_image_path = fd.askopenfilename(
            title='Open a file',
            # initialdir='./',
            initialdir=os.getcwd()
            # filetypes=filetypes
            )
        
        print(f'Opening image {self.input_image_path}')

        render = ImageTk.PhotoImage(Image.open(self.input_image_path).resize((256,256)))
        image = tk.Label(self.master, image=render)
        image.image = render
        image.place(x = 50, y = 320)

        tk.Label(self.master, text="Original image", justify='center').place(x = 140, y = 590)
        
    # Open and show GT image
    def show_GT_image(self):
        
        self.gt_image_path = fd.askopenfilename(
            title='Open a file',
            # initialdir='./',
            initialdir=os.getcwd()
            # filetypes=filetypes
            )
            
        print(f'Opening image {self.gt_image_path}')

        render = ImageTk.PhotoImage(Image.open(self.gt_image_path).resize((256,256)))
        image = tk.Label(self.master, image=render)
        image.image = render
        image.place(x = 320, y = 320)

        tk.Label(self.master, text="GT image", justify='center').place(x = 410, y = 590)

        # test_binary(image_path.get())

    def cascade_modules(self):
        self.types = [module.get() for module in self.modules]
        print(self.types)
        self.run = test_ctunet(types=self.types)
        print('Done cascading!')

app = App()
