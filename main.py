from tkinter import *
from tkinter import filedialog
from tkinter import _setit

import numpy as np
from PIL import Image, ImageTk

import convolution
import detection
import tools
import segmentation
import noise
import image_pairs
import morphologic


class mainWindow():
    thumbnail_width = 250  # --- side length of thumbnail images
    figure_index = 0  # -------- figure index is used to open multiple figures simultaneously
    root = None  # ------------- tkinter gui root element
    canvas_input = None  # ----- canvas for input image
    canvas_output = None  # ---- canvas for output image
    in_image_mat_1 = None  # --- float matrix input image slot 1 (float64[:,:,:] 1 or 3 channels)
    in_image_mat_2 = None  # --- float matrix input image slot 2 (float64[:,:,:] 1 or 3 channels)
    out_image_mat_1 = None  # -- float matrix output image slot 1 (float64[:,:,:] 1 or 3 channels)
    out_image_mat_2 = None  # -- float matrix output image slot 2 (float64[:,:,:] 1 or 3 channels)
    left_slot_var = None  # -----tkinter integer variable [1 or 2]
    right_slot_var = None  # ----tkinter integer variable [1 or 2]
    rgb_var = None  # ---------- tkinter string variable [RGB or GRAY]
    group_var = None  # ----=--- tkinter string variable [names of implemented groups]
    operation_var = None  # ---- tkinter string variable [operations of current group]
    menu_operation = None  # --- tkinter option menu holding the option_var
    label_p1 = None  # --------- tkinter label for parameter 1
    label_p2 = None  # --------- tkinter label for parameter 2
    label_p3 = None  # --------- tkinter label for parameter 3
    entry_p1 = None  # --------- tkinter entry for parameter 1
    entry_p2 = None  # --------- tkinter entry for parameter 2
    entry_p3 = None  # --------- tkinter entry for parameter 3

    def __init__(self):
        # ============== GUI ROOT ELEMENT ================

        self.root = Tk()
        self.root.resizable(0, 0)

        # ============== THUMBNAILS =================

        self.root.in_img = PhotoImage(width=self.thumbnail_width, height=self.thumbnail_width)
        self.canvas_input = Canvas(self.root, width=self.thumbnail_width, height=self.thumbnail_width, bg="#000000")
        self.canvas_input.create_image((0, 0), anchor='nw', image=self.root.in_img, state="normal")
        self.canvas_input.grid(row=2, column=0, rowspan=8, columnspan=2, padx=3, pady=1)

        self.root.out_img = PhotoImage(width=self.thumbnail_width, height=self.thumbnail_width)
        self.canvas_output = Canvas(self.root, width=self.thumbnail_width, height=self.thumbnail_width, bg="#000000")
        self.canvas_output.create_image((0, 0), anchor='nw', image=self.root.out_img, state="normal")
        self.canvas_output.grid(row=2, column=4, rowspan=8, columnspan=2, padx=3, pady=1)

        # ============= UPPER BUTTONS ===============

        Button(self.root, text="Load", command=self.load_image, width=12).grid(row=0, column=0,
                                                                               sticky='NSEW', padx=5, pady=5)
        Button(self.root, text="Save", command=self.save_image).grid(row=0, column=4, columnspan=2,
                                                                     sticky='NSEW', padx=5, pady=5)
        self.rgb_var = IntVar(self.root)
        rgb_checkbox = Checkbutton(self.root, text="load color", variable=self.rgb_var)
        rgb_checkbox.grid(row=0, column=1, sticky='NSEW', padx=5, pady=5)

        # =============== SLOT VARS =================

        self.left_slot_var = IntVar(self.root)
        rb = Radiobutton(self.root, text="input 1", variable=self.left_slot_var, value=1, width=15,
                         command=self.left_slot_changed)
        rb.grid(row=1, column=0, sticky=E, pady=1)
        rb.invoke()
        rb = Radiobutton(self.root, text="input 2", variable=self.left_slot_var, value=2, width=15,
                         command=self.left_slot_changed)
        rb.grid(row=1, column=1, sticky=W, pady=1)

        self.right_slot_var = IntVar(self.root)
        rb = Radiobutton(self.root, text="output 1", variable=self.right_slot_var, value=1, width=15,
                         command=self.right_slot_changed)
        rb.grid(row=1, column=4, sticky=E, pady=1)
        rb.invoke()
        rb = Radiobutton(self.root, text="output 2", variable=self.right_slot_var, value=2, width=15,
                         command=self.right_slot_changed)
        rb.grid(row=1, column=5, sticky=W, pady=1)

        # ============= LOWER BUTTONS ===============

        last_row = 10

        b = Button(self.root, text="Show", command=lambda: self.show_full(True), width=12)
        b.grid(row=last_row, column=0, sticky='NSEW', padx=5, pady=5)

        b = Button(self.root, text="Histogram",
                   command=lambda: self.show_histogram(True), width=12)
        b.grid(row=last_row, column=1, sticky='NSEW', padx=5, pady=5)

        b = Button(self.root, text="Show", command=lambda: self.show_full(False), width=12)
        b.grid(row=last_row, column=4, sticky='NSEW', padx=5, pady=5)

        b = Button(self.root, text="Histogram",
                   command=lambda: self.show_histogram(False), width=12)
        b.grid(row=last_row, column=5, sticky='NSEW', padx=5, pady=5)

        # ======== IMAGE OPERATION SETTINGS =========

        self.group_var = StringVar(self.root)
        choices = {convolution.GROUPNAME, detection.GROUPNAME, tools.GROUPNAME, segmentation.GROUPNAME, noise.GROUPNAME,
                   image_pairs.GROUPNAME, morphologic.GROUPNAME}
        self.group_var.set(convolution.GROUPNAME)  # set default option
        menu_group = OptionMenu(self.root, self.group_var, *choices, command=self.group_changed)
        menu_group.grid(row=1, column=2, columnspan=2, sticky='NSEW', pady=1)

        self.operation_var = StringVar(self.root)
        choices = convolution.OPS

        self.operation_var.set(convolution.OPS[0])
        self.menu_operation = OptionMenu(self.root, self.operation_var, *choices)
        self.menu_operation.grid(row=2, column=2, columnspan=2, sticky='NSEW', pady=1, padx=10)
        self.operation_var.trace('w', self.operation_changed)

        self.label_p1 = Label(self.root, text="filter size\n(N >= 3)", justify='left')
        self.label_p1.grid(row=3, column=2, sticky=N + S + W)
        self.label_p2 = Label(self.root, text="-\n", justify='left')
        self.label_p2.grid(row=4, column=2, sticky=N + S + W)
        self.label_p3 = Label(self.root, text="border\n(valid, zero, same)", justify='left')
        self.label_p3.grid(row=5, column=2, sticky=N + S + W)
        self.label_p4 = Label(self.root, text="-\n", justify='left')
        self.label_p4.grid(row=6, column=2, sticky=N + S + W)

        self.entry_p1 = Entry(self.root, text='')
        self.entry_p1.grid(row=3, column=3, sticky=N + S + E + W)
        self.entry_p2 = Entry(self.root, text='')
        self.entry_p2.grid(row=4, column=3, sticky=N + S + E + W)
        self.entry_p3 = Entry(self.root, text='')
        self.entry_p3.grid(row=5, column=3, sticky=N + S + E + W)
        self.entry_p4 = Entry(self.root, text='')
        self.entry_p4.grid(row=6, column=3, sticky=N + S + E + W)

        self.entry_p1.delete(0, END)
        self.entry_p1.insert(0, "3")
        self.entry_p2.delete(0, END)
        self.entry_p2.insert(0, "")
        self.entry_p3.delete(0, END)
        self.entry_p3.insert(0, "same")
        self.entry_p4.delete(0, END)
        self.entry_p4.insert(0, "")

        Button(self.root, text="== Apply =>", command=self.apply).grid(row=7, column=2, columnspan=2,
                                                                       sticky='NSEW', pady=5, padx=10)
        Button(self.root, text="<= Copy ==", command=self.copy).grid(row=8, column=2, columnspan=2,
                                                                     sticky='NSEW', pady=5, padx=10)

        # =============== START GUI =================

        self.root.mainloop()

    def draw_matrix_to_thumbnail(self, source_mat_f, input):
        if source_mat_f is None:
            source_mat = np.zeros((self.thumbnail_width, self.thumbnail_width, 3), dtype=np.ubyte)
        else:
            source_mat = source_mat_f.astype(np.ubyte)

        if source_mat.shape[2] == 1:
            data = source_mat[:, :, 0]
            im = Image.fromarray(data, mode='L')
        else:
            data = source_mat[:, :, :]
            im = Image.fromarray(data, mode='RGB')

        im.thumbnail((self.thumbnail_width, self.thumbnail_width), Image.ANTIALIAS)

        if input:
            self.imTk_in = ImageTk.PhotoImage(im)
            self.canvas_input.create_image((0, 0), anchor='nw', image=self.imTk_in, state="normal")
        else:
            self.imTk_out = ImageTk.PhotoImage(im)
            self.canvas_output.create_image((0, 0), anchor='nw', image=self.imTk_out, state="normal")

    def load_image(self):
        path = filedialog.askopenfilename()
        if len(path) == 0:
            return

        input = tools.imread3D(path)
        if not self.rgb_var.get() and input.shape[2] == 3:
            input = tools.convert_to_1channel(input)

        if self.left_slot_var.get() == 1:
            self.in_image_mat_1 = input
            self.draw_matrix_to_thumbnail(self.in_image_mat_1, True)
            print('input 1 <- ', input.shape)
        else:
            self.in_image_mat_2 = input
            self.draw_matrix_to_thumbnail(self.in_image_mat_2, True)
            print('input 2 <- ', input.shape)

    def save_image(self):
        if self.right_slot_var.get() == 1:
            image = self.out_image_mat_1
        else:
            image = self.out_image_mat_2
        if image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(
                ("Portable Network Graphics", "*.png"), ("Joint Photographic Experts Group", "*.jpeg"),
                ("All Files", "*.*")))
            tools.imsave3D(path, image)

    def show_histogram(self, isinput):
        if isinput:
            name = 'input image'
            if self.left_slot_var.get() == 1:
                image = self.in_image_mat_1
            else:
                image = self.in_image_mat_2
        else:
            name = 'output image'
            if self.right_slot_var.get() == 1:
                image = self.out_image_mat_1
            else:
                image = self.out_image_mat_2
        if image is None:
            return
        if image.shape[2] == 1:
            gray = image.astype(np.ubyte)
            hist = np.zeros(256, dtype=np.int64)
            for i in range(256):
                hist[i] = np.sum((gray == i).astype(np.int64))
            tools.plot_hist(hist, name)

        else:
            gray = tools.convert_to_1channel(image).astype(np.ubyte)
            red = image[:, :, 0].astype(np.ubyte)
            green = image[:, :, 1].astype(np.ubyte)
            blue = image[:, :, 2].astype(np.ubyte)

            hist_gray = np.zeros(256, dtype=np.int64)
            hist_red = np.zeros(256, dtype=np.int64)
            hist_green = np.zeros(256, dtype=np.int64)
            hist_blue = np.zeros(256, dtype=np.int64)
            for i in range(256):
                hist_gray[i] = np.sum((gray == i).astype(np.int64))
                hist_red[i] = np.sum((red == i).astype(np.int64))
                hist_green[i] = np.sum((green == i).astype(np.int64))
                hist_blue[i] = np.sum((blue == i).astype(np.int64))
            tools.plot_hist3(hist_gray, hist_red, hist_green, hist_blue, name)

    def show_full(self, left):
        if left:
            if self.left_slot_var.get() == 1:
                image = self.in_image_mat_1
            else:
                image = self.in_image_mat_2
        else:
            if self.right_slot_var.get() == 1:
                image = self.out_image_mat_1
            else:
                image = self.out_image_mat_2

        if image is None:
            return

        self.figure_index += 1
        tools.plot(image, str(self.figure_index))

    def left_slot_changed(self):
        if self.left_slot_var.get() == 1:
            self.draw_matrix_to_thumbnail(self.in_image_mat_1, True)
        else:
            self.draw_matrix_to_thumbnail(self.in_image_mat_2, True)
        return

    def right_slot_changed(self):
        if self.right_slot_var.get() == 1:
            self.draw_matrix_to_thumbnail(self.out_image_mat_1, False)
        else:
            self.draw_matrix_to_thumbnail(self.out_image_mat_2, False)
        return

    def group_changed(self, _):
        self.operation_var.set('')
        self.menu_operation['menu'].delete(0, 'end')

        operation = self.group_var.get()
        if operation == convolution.GROUPNAME:
            new_choices = convolution.OPS

        elif operation == detection.GROUPNAME:
            new_choices = detection.OPS

        elif operation == tools.GROUPNAME:
            new_choices = tools.OPS

        elif operation == segmentation.GROUPNAME:
            new_choices = segmentation.OPS

        elif operation == noise.GROUPNAME:
            new_choices = noise.OPS

        elif operation == image_pairs.GROUPNAME:
            new_choices = image_pairs.OPS

        elif operation == morphologic.GROUPNAME:
            new_choices = morphologic.OPS

        for choice in new_choices:
            self.menu_operation['menu'].add_command(label=choice, command=_setit(self.operation_var, choice))

        self.operation_var.set(new_choices[0])

    def operation_changed(self, _, __, ___):
        group = self.group_var.get()
        operation = self.operation_var.get()

        self.label_p1['text'] = '-\n'
        self.label_p2['text'] = '-\n'
        self.label_p3['text'] = '-\n'
        self.label_p4['text'] = '-\n'

        self.entry_p1.delete(0, END)
        self.entry_p2.delete(0, END)
        self.entry_p3.delete(0, END)
        self.entry_p4.delete(0, END)

        if group == convolution.GROUPNAME:
            self.label_p3['text'] = 'border\n(valid, zero, same)'
            self.entry_p3.insert(0, "same")

            if operation == convolution.BLOCK or \
                    operation == convolution.BINOMIAL or \
                    operation == convolution.MINIMUM or \
                    operation == convolution.MEDIAN or \
                    operation == convolution.MAXIMUM:
                self.label_p1['text'] = 'filter size\n(N >= 3)'
                self.entry_p1.insert(0, "3")

            elif operation == convolution.GAUSS or \
                    operation == convolution.LAPLACIAN_OF_GAUSSIAN:
                self.label_p1['text'] = 'sigma\n(s > 0.4)'
                self.entry_p1.insert(0, "2.0")

            elif operation == convolution.DERIVATIVE_OF_GAUSSIAN:
                self.label_p1['text'] = 'sigma\n(s > 0.4)'
                self.label_p4['text'] = 'compute\n(x, y, amp, dir, color)'
                self.entry_p1.insert(0, "2.0")
                self.entry_p4.insert(0, "color")

            elif operation == convolution.SIMPLE_FIRST_DERIVATIVE or \
                    operation == convolution.SOBEL:
                self.label_p1['text'] = '-\n'
                self.label_p4['text'] = 'compute\n(x, y, amp, dir, color)'
                self.entry_p1.insert(0, "")
                self.entry_p4.insert(0, "color")

            elif operation == convolution.SIMPLE_SECOND_DERIVATIVE:
                self.label_p1['text'] = '-\n'
                self.label_p4['text'] = 'compute\n(xx, yy, xy, det)'
                self.entry_p1.insert(0, "")
                self.entry_p4.insert(0, "xx")

            elif operation == convolution.SECOND_DERIVATIVE_OF_GAUSSIAN:
                self.label_p1['text'] = 'sigma\n(s > 0.4)'
                self.label_p4['text'] = 'compute\n(xx, yy, xy, det)'
                self.entry_p1.insert(0, "2.0")
                self.entry_p4.insert(0, "xx")

            elif operation == convolution.DIFFERENCE_OF_GAUSSIAN:
                self.label_p1['text'] = 'sigma high\n(s_h > s_l)'
                self.label_p2['text'] = 'sigma low\n(s_l > 0.4)'
                self.entry_p1.insert(0, "2.1")
                self.entry_p2.insert(0, "2.0")

            elif operation == convolution.SHARPEN:
                self.label_p1['text'] = 'sigma \n(s > 0.4)'
                self.label_p2['text'] = 'gamma\n(y > 0.0)'
                self.entry_p1.insert(0, "1.5")
                self.entry_p2.insert(0, "5")

        elif group == detection.GROUPNAME:

            if operation == detection.BEAUDET:
                self.label_p1['text'] = 'sigma\n(s > 0.4)'
                self.label_p4['text'] = 'num. best points\n(n > 0)'
                self.entry_p1.insert(0, "1")
                self.entry_p4.insert(0, "25")
            elif operation == detection.HARRIS:
                self.label_p1['text'] = 'filter size\n(N >= 1)'
                self.label_p2['text'] = 'kappa\n(k > 0.0)'
                self.label_p3['text'] = 'min. cornerness\n(r >= 0.0)'
                self.label_p4['text'] = 'num. best points\n(n > 0)'
                self.entry_p1.insert(0, "7")
                self.entry_p2.insert(0, "0.06")
                self.entry_p3.insert(0, "3")
                self.entry_p4.insert(0, "25")
            elif operation == detection.SCALE_ADAPTED_HARRIS:
                self.label_p1['text'] = 'differentiation scale\n(sigma >= 0.4)'
                self.label_p2['text'] = 'kappa\n(k > 0.0)'
                self.label_p3['text'] = 'min. cornerness\n(r >= 0.0)'
                self.label_p4['text'] = 'num. best points\n(n > 0)'
                self.entry_p1.insert(0, "1")
                self.entry_p2.insert(0, "0.06")
                self.entry_p3.insert(0, "3")
                self.entry_p4.insert(0, "25")
            elif operation == detection.FOERSTNER:
                self.label_p1['text'] = 'differentiation scale\n(sigma_d >= 0.4)'
                self.label_p2['text'] = 'min. weight\n(omega >= 0.0)'
                self.label_p3['text'] = 'min. isotropy\n(1 >= q >= 0.0)'
                self.label_p4['text'] = 'num. best points\n(n > 0)'
                self.entry_p1.insert(0, "1.0")
                self.entry_p2.insert(0, "5.0")
                self.entry_p3.insert(0, "0.75")
                self.entry_p4.insert(0, "25")
            elif operation == detection.DIFFERENCE_OF_GAUSSIAN:
                self.label_p1['text'] = 'octaves\n(o >= 1)'
                self.label_p2['text'] = 'scales per octave\n(s >= 2)'
                self.label_p3['text'] = 'use best p percent\n(100 >= p > 0)'
                self.label_p4['text'] = 'min. DoG response\n(t >= 0.0))'
                self.entry_p1.insert(0, "3")
                self.entry_p2.insert(0, "4")
                self.entry_p3.insert(0, "10")
                self.entry_p4.insert(0, "5")
            elif operation == detection.LOG_DETECTOR:
                self.label_p1['text'] = 'sigma_min\n(s_min > 0.4)'
                self.label_p2['text'] = 'sigma_max\n(s_max > s_min)'
                self.label_p3['text'] = 'num. sigmas\n(s >= 1)'
                self.label_p4['text'] = 'num. best points\n(n > 0)'
                self.entry_p1.insert(0, "1")
                self.entry_p2.insert(0, "12")
                self.entry_p3.insert(0, "8")
                self.entry_p4.insert(0, "25")
            elif operation == detection.EDGE_PIXELS:
                self.label_p1['text'] = 'gauss scale\n(sigma >= 0.4)'
                self.label_p2['text'] = 'minimum amplitude\n(t_amp >= 0.0)'
                self.label_p3['text'] = '-\n'
                self.label_p4['text'] = '-\n'
                self.entry_p1.insert(0, "3.0")
                self.entry_p2.insert(0, "2.0")
                self.entry_p3.insert(0, "")
                self.entry_p4.insert(0, "")
            elif operation == detection.HOUGH_LINES:
                self.label_p1['text'] = 'gauss scale\n(sigma >= 0.4)'
                self.label_p2['text'] = 'show hough space\n(0,1)'
                self.label_p3['text'] = 'amplitude threshold\n(t >= 0.0)'
                self.label_p4['text'] = 'num lines\n(n > 0)'
                self.entry_p1.insert(0, "1.0")
                self.entry_p2.insert(0, "0")
                self.entry_p3.insert(0, "10.0")
                self.entry_p4.insert(0, "7")

        elif group == tools.GROUPNAME:
            if operation == tools.EXTRACT_CHANNEL:
                self.label_p1['text'] = 'channel\n(c >= 0)'
                self.entry_p1.insert(0, "0")
            elif operation == tools.HISTOGRAM_NORMALISATION:
                self.label_p1['text'] = 'outlier fraction\n(1.0 > p >= 0.0)'
                self.entry_p1.insert(0, "0.04")
            elif operation == tools.HISTOGRAM_EQUALIZATION:
                self.label_p1['text'] = 'alpha\n(1.0 > a >= 0.0)'
                self.entry_p1.insert(0, "0.5")
            elif operation == tools.LOCAL_HISTOGRAM_EQUALIZATION:
                self.label_p1['text'] = 'alpha\n(1.0 > a >= 0.0)'
                self.entry_p1.insert(0, "0.5")
                self.label_p2['text'] = 'context window size\n(M > 0)'
                self.entry_p2.insert(0, "100")
            elif operation == tools.EXTEND:
                self.label_p1['text'] = 'size\n(N > 0)'
                self.entry_p1.insert(0, "15")
                self.label_p2['text'] = 'type\n(same, zero)'
                self.entry_p2.insert(0, "same")
            elif operation == tools.RESCALE:
                self.label_p1['text'] = 'factor\n(f > 0.0)'
                self.entry_p1.insert(0, "2")
            elif operation == tools.RGB2HSV:
                self.label_p1['text'] = 'display\n(h,s,v,all)'
                self.entry_p1.insert(0, "s")
            elif operation == tools.RGB2LAB:
                self.label_p1['text'] = 'display\n(l,a,b,all)'
                self.entry_p1.insert(0, "l")

        elif group == segmentation.GROUPNAME:
            if operation == segmentation.WATERSHED:
                self.label_p1['text'] = 'sigma\n(s > 0.4)'
                self.entry_p1.insert(0, "2.0")
                self.label_p2['text'] = 'shed only\n(0,1)'
                self.entry_p2.insert(0, "0")
                self.label_p3['text'] = 'new seed threshold\n(t >= 0.0)'
                self.entry_p3.insert(0, "1.5")
            elif operation == segmentation.GRAY_THRESHOLDING:
                self.label_p1['text'] = 'min. gray value\n(255.0 >= g_min >= 0.0)'
                self.entry_p1.insert(0, "100")
                self.label_p2['text'] = 'max. gray value\n(255.0 >= g_max >= 0.0)'
                self.entry_p2.insert(0, "255")
            elif operation == segmentation.RGB_THRESHOLDING:
                self.label_p1['text'] = 'red: min,max\n(255.0 >= r >= 0.0)'
                self.entry_p1.insert(0, "100,255")
                self.label_p2['text'] = 'green: min,max\n(255.0 >= g >= 0.0)'
                self.entry_p2.insert(0, "0,255")
                self.label_p3['text'] = 'blue: min,max\n(255.0 >= b >= 0.0)'
                self.entry_p3.insert(0, "0,255")
            elif operation == segmentation.HSV_THRESHOLDING:
                self.label_p1['text'] = 'hue: min,max\n(360.0 >= h >= 0.0)'
                self.entry_p1.insert(0, "100.0,140.0")
                self.label_p2['text'] = 'saturation: min,max\n(1.0 >= s >= 0.0)'
                self.entry_p2.insert(0, "0.0,1.0")
                self.label_p3['text'] = 'value: min,max\n(1.0 >= v >= 0.0)'
                self.entry_p3.insert(0, "0.0,1.0")

            elif operation == segmentation.REGION_GROWING:
                self.label_p1['text'] = 'max. dist. color/gray\n(h_c > 0.0)'
                self.entry_p1.insert(0, "15")
                self.label_p4['text'] = 'color segments\n(random,average)'
                self.entry_p4.insert(0, "average")

            elif operation == segmentation.MEAN_SHIFT:
                self.label_p1['text'] = 'bandwith color/gray\n(h_c > 0.0)'
                self.entry_p1.insert(0, "15")
                self.label_p2['text'] = 'use spacial\n(0,1)'
                self.entry_p2.insert(0, "0")
                self.label_p3['text'] = 'bandwith space\n(h_s > 0.0)'
                self.entry_p3.insert(0, "25")
            elif operation == segmentation.SLIC:
                self.label_p1['text'] = 'number of superpixels\n(K >= 1)'
                self.entry_p1.insert(0, "500")
                self.label_p2['text'] = 'compactness\n(m > 0.0)'
                self.entry_p2.insert(0, "100")
                self.label_p3['text'] = 'L1 Threshold\n(E > 0.0)'
                self.entry_p3.insert(0, "10")
                self.label_p4['text'] = 'color segments\n(random,average)'
                self.entry_p4.insert(0, "average")
            elif operation == segmentation.FILTER_BANK_CLUSTERING:
                self.label_p1['text'] = 'sigma min (s_min > 1.0)'
                self.entry_p1.insert(0, "1.0")
                self.label_p2['text'] = 'sigma max (s_max > s_min)'
                self.entry_p2.insert(0, "5.0")
                self.label_p3['text'] = 'nr. of clusters (k > 1)'
                self.entry_p3.insert(0, "5")
                self.label_p4['text'] = 'k-means threshold (th >= 0.0)'
                self.entry_p4.insert(0, "0.01")

        elif group == noise.GROUPNAME:
            if operation == noise.WHITE:
                self.label_p1['text'] = 'sigma\n(s > 0.0)'
                self.entry_p1.insert(0, "10.0")
            elif operation == noise.SALTPEPPER:
                self.label_p1['text'] = 'probability\n(1.0 > p > 0.0)'
                self.entry_p1.insert(0, "0.05")

        elif group == image_pairs.GROUPNAME:
            if operation == image_pairs.SIFT_MATCH:
                self.label_p1['text'] = 'octaves\n(o >= 1)'
                self.label_p2['text'] = 'scales per octave\n(s >= 2)'
                self.label_p3['text'] = 'use best p percent\n(100 >= p > 0)'
                self.label_p4['text'] = 'min. DoG response\n(t >= 0.0))'
                self.entry_p1.insert(0, "3")
                self.entry_p2.insert(0, "4")
                self.entry_p3.insert(0, "15")
                self.entry_p4.insert(0, "10")

        elif group == morphologic.GROUPNAME:
            if operation == morphologic.DISTANCE_MAP:
                self.label_p1['text'] = 'distance\n(manhattan, euclidian)'
                self.label_p2['text'] = 'search width (euclidian)\n(N >= 3)'
                self.entry_p1.insert(0, "euclidian")
                self.entry_p2.insert(0, "2")
            else:
                self.label_p1['text'] = 'shape\n(circle, square)'
                self.label_p2['text'] = 'sidelength of S.E.\n(N >= 3)'
                self.label_p3['text'] = 'iterations\n(i >= 1)'
                self.entry_p1.insert(0, "circle")
                self.entry_p2.insert(0, "7")
                self.entry_p3.insert(0, "1")

    def apply(self):
        in_slot = self.left_slot_var.get()
        out_slot = self.right_slot_var.get()
        operation = self.group_var.get()
        op_type = self.operation_var.get()

        if in_slot == 1:
            image = self.in_image_mat_1
        else:
            image = self.in_image_mat_2
        if image is None:
            return

        image_float = image.astype(np.float64)
        p1 = self.entry_p1.get()
        p2 = self.entry_p2.get()
        p3 = self.entry_p3.get()
        p4 = self.entry_p4.get()
        try:
            if operation == convolution.GROUPNAME:
                result = convolution.apply(image_float, op_type, p1, p2, p3, p4)
            elif operation == detection.GROUPNAME:
                result = detection.apply(image_float, op_type, p1, p2, p3, p4)
            elif operation == tools.GROUPNAME:
                result = tools.apply(image_float, op_type, p1, p2, p3, p4)
            elif operation == segmentation.GROUPNAME:
                result = segmentation.apply(image_float, op_type, p1, p2, p3, p4)
            elif operation == noise.GROUPNAME:
                result = noise.apply(image_float, op_type, p1, p2, p3, p4)
            elif operation == image_pairs.GROUPNAME:
                if in_slot == 1:
                    image_2 = self.in_image_mat_2
                else:
                    image_2 = self.in_image_mat_1
                if image_2 is None:
                    return
                image_2_float = image_2.astype(np.float64)
                result = image_pairs.apply(image_float, image_2_float, op_type, p1, p2, p3, p4)
            elif operation == morphologic.GROUPNAME:
                result = morphologic.apply(image_float, op_type, p1, p2, p3, p4)

        except ValueError:
            print('Value Error')
            return

        if out_slot == 1:
            self.out_image_mat_1 = result
        else:
            self.out_image_mat_2 = result
        self.draw_matrix_to_thumbnail(result, False)

    def copy(self):
        in_slot = self.left_slot_var.get()
        out_slot = self.right_slot_var.get()
        if out_slot == 1:
            out_image = self.out_image_mat_1
        else:
            out_image = self.out_image_mat_2

        if out_image is None:
            return
        if in_slot == 1:
            self.in_image_mat_1 = out_image.copy()
            self.draw_matrix_to_thumbnail(self.in_image_mat_1, True)
        else:
            self.in_image_mat_2 = out_image.copy()
            self.draw_matrix_to_thumbnail(self.in_image_mat_2, True)


if __name__ == "__main__":
    mainWindow()
