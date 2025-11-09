# 📸 Image Augmentation System for CV/ML

This project implements an **augmentation system for images** used in Computer Vision and Machine Learning tasks.  
It automatically loads `.jpg` images from a selected folder, applies a set of configurable augmentation algorithms, and saves the results in a new output folder.

---

## ✅ Project Spec

- Python and OpenCV installed on the local machine.
- A `Poze` directory containing ~5 images (`.jpg`), preferably **640x480**, with various content.
- A program that:
  - lets the user select a directory (Tkinter dialog),
  - reads all `.jpg` images from that directory,
  - applies predefined augmentation algorithms with parameters (from a config file),
  - saves results in `<input_folder>_aug`,
  - uses file names of the form:
    `originalName_OperationName_counter.jpg`  
    with a **global counter** starting from `_1`.

This implementation **meets and extends** those requirements.

---

## 🧠 Features Overview

- 🗂️ **Interactive folder & config selection** via Tkinter.
- ⚙️ **Config-driven pipeline**: algorithms and parameters are defined in a text file.
- 🔗 **Chain processing support (bonus)** using `|` to apply multiple operations in sequence.
- 🧮 **Pixel-level and geometric augmentations** implemented modularly.
- 🧱 **Low-level implementations** (direct pixel operations) for required algorithms.

---

## 🧩 Supported Operations

All operations are defined in `OPS` and can be used in the config file by name.

### 🔹 Pixel-level / Color 

| Name         | Syntax                 | Description                              | Implementation |
|-------------|------------------------|------------------------------------------|----------------|
| `Dummy`     | `Dummy`                | Copies the input image (test only).      | direct copy    |
| `Brightness`| `Brightness value`     | Adds `value` to all pixels, clamped.     | low-level      |
| `Contrast`  | `Contrast factor`      | Scales around 128: `(p-128)*a+128`.      | low-level      |
| `Grayscale` | `Grayscale`            | Converts to grayscale (weighted RGB).    | low-level      |

### 🔹 Geometric 

| Name       | Syntax                  | Description                              | Implementation |
|------------|-------------------------|------------------------------------------|----------------|
| `FlipH`    | `FlipH`                 | Horizontal flip (mirror).                | low-level      |
| `Rotation` | `Rotation angle`        | Rotates by `angle` degrees (center).     | OpenCV         |
| `Resize`   | `Resize W x H`          | Resizes to `W`×`H` (e.g. `640x480`).     | OpenCV         |

---

📚 Context

This project was developed as part of the Fundamentals of Computer Vision course, targeting:

data augmentation for ML,

modular design,

pixel-level and geometric processing,

configuration-driven workflows,

and bonus support for chained operations.

---

## 🔗 Config File Language

The **config file** is a simple text file.  
Each line defines **one step** that is applied to **all images**.

You can specify:
- a **single operation** per line, or
- a **chain** of operations using `|`.

Comments starting with `#` are ignored.


▶️ How to Run

Install dependencies:

pip install opencv-python numpy tk


Prepare:

a folder with .jpg images (e.g. Test/)

a config file (e.g. config_example.txt) with operations like in the examples above.

Run the script:

python augmentor.py


When prompted:

select the config file,

select the input folder with .jpg images.

The program will create an output folder:

<input_folder>_aug


and save files like:

img1_Dummy_1.jpg
img1_Brightness_2.jpg
img1_Brightness+Contrast+Grayscale_3.jpg
