from tkinter import *
from tkinter import filedialog
from tkinter import _setit

import numpy as np
from PIL import Image, ImageTk

import tools


class mainWindow():
    thumbnail_width = 500  # --- side length of thumbnail image
    image_width = 500  # ------- width of full size image
    image_height = 500  # ------ height of full size image
    root = None  # ------------- tkinter gui root element
    canvas_input = None  # ----- canvas for input image
    in_image_mat = None  # ----- float matrix input image slot 1 (float64[:,:,:] 1 or 3 channels)
    rgb_var = None  # ---------- tkinter string variable [RGB or GRAY]
    group_var = None  # ----=--- tkinter string variable [names of implemented groups]
    operation_var = None  # ---- tkinter string variable [operations of current group]
    menu_operation = None  # --- tkinter option menu holding the option_var

    def __init__(self):
        # ============== GUI ROOT ELEMENT ================

        self.root = Tk()
        self.root.resizable(0, 0)

        # ============== THUMBNAILS =================

        self.root.in_img = PhotoImage(width=self.thumbnail_width, height=self.thumbnail_width)
        self.root.in_img_overlay = PhotoImage(width=self.thumbnail_width, height=self.thumbnail_width)
        self.canvas_input = Canvas(self.root, width=self.thumbnail_width, height=self.thumbnail_width, bg="#000000")
        self.canvas_input.create_image((0, 0), anchor='nw', image=self.root.in_img, state="normal")
        self.canvas_input.grid(row=1, column=0, rowspan=8, columnspan=2, padx=3, pady=1)
        # create a transparent overlay to which we can draw
        self.canvas_input.create_image((0, 0), anchor='nw', image=self.root.in_img_overlay, state="normal")

        # add mouse event to canvas
        self.canvas_input.bind("<Button-1>", self.mouse_click)

        # add mouse move event to canvas
        self.canvas_input.bind("<B1-Motion>", self.mouse_click)

        # ============= UPPER BUTTONS ===============

        Button(self.root, text="Load", command=self.load_image, width=12).grid(row=0, column=0,
                                                                               sticky='NSEW', padx=5, pady=5)

        # #add dropdown menu to switch between coorinate modes
        # self.coordmode_var = StringVar(self.root)
        # self.coordmode_var.set("Pixel")  # default value
        # self.coordmode_menu = OptionMenu(self.root, self.coordmode_var, "Pixel", "Percent")
        # self.coordmode_menu.grid(row=0, column=1, sticky='NSEW', padx=5, pady=5)

        # ============= LABELS ===============

        label_height = Label(self.root, text="Height", justify='left')
        label_height.grid(row=1, column=2, sticky=N + S + W)
        self.label_height_value = Label(self.root, text="0", justify='left')
        self.label_height_value.grid(row=1, column=3, sticky=N + S + W)

        label_width = Label(self.root, text="Width", justify='left')
        label_width.grid(row=2, column=2, sticky=N + S + W)
        self.label_width_value = Label(self.root, text="0", justify='left')
        self.label_width_value.grid(row=2, column=3, sticky=N + S + W)

        label_row = Label(self.root, text="Row:", justify='left')
        label_row.grid(row=3, column=2, sticky=N + S + W)
        self.label_row_value = Label(self.root, text="0", justify='left')
        self.label_row_value.grid(row=3, column=3, sticky=N + S + W)
        self.label_row_value_pct = Label(self.root, text="0", justify='left')
        self.label_row_value_pct.grid(row=3, column=4, sticky=N + S + W)

        label_col = Label(self.root, text="Col:", justify='left')
        label_col.grid(row=4, column=2, sticky=N + S + W)
        self.label_col_value = Label(self.root, text="0", justify='left')
        self.label_col_value.grid(row=4, column=3, sticky=N + S + W)
        self.label_col_value_pct = Label(self.root, text="0", justify='left')
        self.label_col_value_pct.grid(row=4, column=4, sticky=N + S + W)

        label_x = Label(self.root, text="X", justify='left')
        label_x.grid(row=5, column=2, sticky=N + S + W)
        self.label_x_value = Label(self.root, text="0", justify='left')
        self.label_x_value.grid(row=5, column=3, sticky=N + S + W)

        label_y = Label(self.root, text="Y", justify='left')
        label_y.grid(row=6, column=2, sticky=N + S + W)
        self.label_y_value = Label(self.root, text="0", justify='left')
        self.label_y_value.grid(row=6, column=3, sticky=N + S + W)

        label_normed_x = Label(self.root, text="X/W", justify='left')
        label_normed_x.grid(row=7, column=2, sticky=N + S + W)
        self.label_normed_x_value = Label(self.root, text="0", justify='left')
        self.label_normed_x_value.grid(row=7, column=3, sticky=N + S + W)

        label_normed_y = Label(self.root, text="Y/W", justify='left')
        label_normed_y.grid(row=8, column=2, sticky=N + S + W)
        self.label_normed_y_value = Label(self.root, text="0", justify='left')
        self.label_normed_y_value.grid(row=8, column=3, sticky=N + S + W)

        # ============= LOWER BUTTONS ===============

        last_row = 9
        b = Button(self.root, text="Show", command=lambda: self.show_full(True), width=12)
        b.grid(row=last_row, column=0, sticky='NSEW', padx=5, pady=5)

        # add textboxes from which coordinates can be copied
        self.textbox_xy = Entry(self.root, width=10)
        self.textbox_xy.grid(row=last_row, column=3, sticky='NSEW', padx=5, pady=5)
        self.textbox_xy.bind("<Button-1>", self.copy_textbox)

        # =============== START GUI =================

        self.root.mainloop()

    def mouse_click(self, event):

        # draw a cross on the overlay
        self.canvas_input.delete("cross")
        self.canvas_input.create_line(event.x - 10, event.y, event.x + 10, event.y, fill="red", tags="cross")
        self.canvas_input.create_line(event.x, event.y - 10, event.x, event.y + 10, fill="red", tags="cross")

        # convert to image coordinates
        x = event.x * self.image_width / self.thumbnail_width
        y = event.y * self.image_width / self.thumbnail_width

        self.label_row_value.config(text=str(int(y)))
        self.label_col_value.config(text=str(int(x)))

        self.label_row_value_pct.config(text=f"({int(y / self.image_height * 100)}%)")
        self.label_col_value_pct.config(text=f"({int(x / self.image_width * 100)}%)")

        # x is horizontal, y is vertical, origin is bottom left
        self.label_x_value.config(text=str(int(x)))
        self.label_y_value.config(text=str(int(self.image_height - y)))

        # x,y normalised to image width
        self.label_normed_x_value.config(text=f"{x / self.image_width:.2f}")
        self.label_normed_y_value.config(text=f"{int(self.image_height - y) / self.image_width:.2f}")

        #copy normalized coordinates to textbox
        self.textbox_xy.delete(0, END)
        self.textbox_xy.insert(0, f"{x / self.image_width:.2f},{int(self.image_height - y) / self.image_width:.2f}")

    def copy_textbox(self, event):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.textbox_xy.get())

    def draw_matrix_to_thumbnail(self, source_mat_f):
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

        self.image_width = im.size[0]
        self.image_height = im.size[1]

        self.label_height_value.config(text=str(self.image_height))
        self.label_width_value.config(text=str(self.image_width))

        im.thumbnail((self.thumbnail_width, self.thumbnail_width), Image.ANTIALIAS)

        self.imTk_in = ImageTk.PhotoImage(im)
        self.canvas_input.create_image((0, 0), anchor='nw', image=self.imTk_in, state="normal")

    def load_image(self):
        path = filedialog.askopenfilename()
        if len(path) == 0:
            return

        input = tools.imread3D(path)

        self.in_image_mat_1 = input
        self.draw_matrix_to_thumbnail(self.in_image_mat_1)

    def show_full(self, _):
        image = self.in_image_mat_1

        if image is None:
            return

        tools.plot(image, str("Image"))


if __name__ == "__main__":
    mainWindow()
