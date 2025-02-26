# Cell Segmentation and Tracking Analysis

This project is designed for segmenting and tracking cells in time-lapse microscopy images using Otsu thresholding and [Cell ACDC](https://github.com/SchmollerLab/Cell_ACDC) tracking. It includes additional functions for filtering tracked cells, calculating their movement speeds, and visualizing results.

## Features
- **Image Preprocessing**: Rescales intensity and applies Gaussian filtering.
- **Cell Segmentation**: Uses Otsu thresholding and region filtering.
- **Cell Tracking**: Integrates with [Cell ACDC](https://github.com/SchmollerLab/Cell_ACDC) for object tracking.
- **Filtering**: Removes objects not present for a minimum number of frames.
- **Speed Calculation**: Computes cell movement speeds over time.
- **Data Visualization**: Generates boxplots and fits Michaelis-Menten kinetics.

## Installation Guide

### Step 1: Install Python
Ensure you have Python installed (recommended version: 3.13+). You can download Python from [python.org](https://www.python.org/downloads/). Make sure to check the box for python to be added to system PATH!

### Step 2: Download files and navigate there
[Download](https://github.com/Teranis/molmot/archive/refs/heads/master.zip) the files form this repository in a zip folder (or clone the repository). Unpack the folder into the desired location. Open the command prompt. Navigate to the folder into which you have extracted the data using the commands:

- **Windows:**
  ```sh
  cd path\to\extracted\folder
  ```
  For example:
    ```sh
  cd C:\molmot-master
  ```
- **Mac/Linux:**
  ```sh
  cd path/to/extracted/folder
  ```

### Step 3: Create a Virtual Environment
It is recommended to create a virtual environment to manage dependencies.

```sh
python -m venv venv
```

Activate the virtual environment:
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source venv/bin/activate
  ```

### Step 4: Install Dependencies

Run the following command to install all required Python libraries:

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not available, install dependencies manually:

```sh
pip install numpy tqdm scikit-image pandas matplotlib seaborn scipy
```

## Usage Guide

1. Place your image files inside the `imgs/` directory.
2. Modify the configuration parameters in the script as needed (e.g., segmentation settings, time interval, pixel size, etc.).
3. Open the command prompt and move to the folder where you placed the files during installation. View Step 2 of the installation guide to see how to navigate there.
4. Run the script:

```sh
python main.py
```
5. For tracking, please use [Cell ACDC](https://github.com/SchmollerLab/Cell_ACDC). The created data structure should also allow users to easily view and edit data in Cell_ACDC.

### Optional Features
- **Restructure data for ACDC:** Set `restruc_for_acdc_toggle = True`.
- **Segment images:** Set `segment_toggle = True`.
- **Filter tracked IDs:** Set `filter_cont_tracking_toggle = True`.
- **Calculate cell speeds:** Set `get_speed_toggle = True`.
- **Generate boxplots:** Set `boxplot_toggle = True`.

## Output
- **Segmented cell masks** (`.npz` files)
- **Filtered tracking data** (`.npz` files)
- **Speed CSV files**
- **Plots and fitted kinetic models**

## License
This project is released under the MIT License.