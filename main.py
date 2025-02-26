import os
import numpy as np
import tqdm
import concurrent.futures
import functools
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

MAX_WORKERS = 8
dir = os.path.join(os.path.dirname(__file__), 'imgs')
end_name = "molmot.tif" # for both data restructure and segmentation, the end name of the image files

restruc_for_acdc_toggle = False # toggle for restructuring the data for ACDC

segment_toggle = False # toggle for segmenting the images
segm_ending = "molmot_segm.npz" # ending for the segm files
neighborhood = skimage.morphology.disk(15) # neighborhood for otsu thresholding

# now use Cell ACDC to track the cells (https://github.com/SchmollerLab/Cell_ACDC)
# classic acdc tracker used (originally developed for budding yeast)
# with an overlap th of 0.4

filter_cont_tracking_toggle = False # toggle for filtering the IDs which are present in all frames
tracked_ending = "molmot_segm_tracked.npz" # ending for the tracked segm files
filter_length = 60 # frames that the object has to be present in to be considered
filtered_mask_ending = "molmot_segm_filtered.npz" # ending for the filtered segm files

get_speed_toggle = False # toggle for getting the speed of the cells
time_interval = 0.1 # time interval between frames in seconds
pixel_size = 100/150 # pixel size in micrometers per pixel

boxplot_toggle = True # toggle for creating a boxplot of the speeds

del_files_toggle = False # toggle for deleting all npz files in a directory, for a clean start ^^
del_files_toggle_2 = False # just to make sure one doesnt delete all npz files by accident
del_end = ".npz" # ending for the files to be deleted

list_files_toggle = False # list all files in a directory
list_end_name = ".npz" # ending for the files to be listed

#####################################################################################################

def del_files(dir, del_end):
    for path in os.listdir(dir):
        if path.endswith(del_end):
            os.remove(os.path.join(dir, path))
        elif os.path.isdir(os.path.join(dir, path)):
            del_files(os.path.join(dir, path), del_end)

def list_files(dir, list_end_name):
    for path in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, path)):
            list_files(os.path.join(dir, path), list_end_name)
        elif path.endswith(list_end_name):
            print(os.path.join(dir, path))

def restruc_for_acdc(dir, end_name):
    for folder in os.listdir(dir):
        tif_files = [file for file in os.listdir(os.path.join(dir, folder)) if file.endswith(".tif")]
        for i, file in enumerate(tif_files):
            os.makedirs(os.path.join(dir, folder, f"Position_{i}", f"Images"), exist_ok=True)
            os.rename(os.path.join(dir, folder, file), os.path.join(dir, folder, f"Position_{i}", f"Images", end_name))

def segm_vid(img_path, neighborhood, end_name, segm_ending):
    img = skimage.io.imread(img_path)
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 1))
    img = skimage.util.img_as_ubyte(img)
    img = np.expand_dims(img, axis=0) if img.ndim == 2 else img

    segm_mask = np.zeros_like(img, dtype=np.int32)

    for i, frame in enumerate(img):
        # preprocess the image
        frame = skimage.filters.gaussian(frame, sigma=1)
        frame = skimage.filters.unsharp_mask(frame, radius=1, amount=1)

        # Apply Otsu's thresholding
        frame = skimage.util.img_as_ubyte(frame)  # Convert to uint8 before applying rank filters
        thresh = skimage.filters.rank.otsu(frame, neighborhood)
        binary = frame > thresh
        labels = skimage.measure.label(binary)
        regions = skimage.measure.regionprops(labels)

        # filter for long, thin regions
        for region in regions:
            if region.minor_axis_length > 20:
                continue
            if region.area < 50:
                continue
            segm_mask[i][labels == region.label] = region.label
    
    segm_mask = np.squeeze(segm_mask)
    save_path = img_path.replace(end_name, segm_ending)
    np.savez(save_path, segm_mask)

def segment(root_dir, neighborhood, end_name, segm_ending):
    imgs_paths = [os.path.join(root_dir, folder, subfolder, "Images", end_name) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder)) for subfolder in os.listdir(os.path.join(root_dir, folder)) if os.path.isdir(os.path.join(root_dir, folder, subfolder))]
    imgs_paths = [img for img in imgs_paths if img.endswith(end_name)]
    print(f"Found {len(imgs_paths)} images to process:")
    [print(path) for path in imgs_paths]

    partial_func = functools.partial(segm_vid,
                                            neighborhood=neighborhood,
                                            end_name=end_name,
                                            segm_ending=segm_ending)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(partial_func, img_path) for img_path in imgs_paths]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                raise e

def filter_vid(path, filter_length, tracked_ending, filtered_mask_ending):
    segm_mask = np.load(path)
    segm_mask = segm_mask["arr_0"]

    segm_mask = np.expand_dims(segm_mask, axis=0) if segm_mask.ndim == 2 else segm_mask

    IDs_per_frame = [np.unique(frame) for frame in segm_mask]
    IDs_per_frame_1D  = []
    for IDs in IDs_per_frame:
        IDs_per_frame_1D.extend(IDs)

    counted_IDs = np.unique(IDs_per_frame_1D, return_counts=True)
    filtered_IDs = set(counted_IDs[0][counted_IDs[1] >= filter_length])

    segm_mask_filtered = np.zeros_like(segm_mask, dtype=np.int32)
    for i, frame in enumerate(segm_mask):
        mask = np.isin(frame, list(filtered_IDs))
        segm_mask_filtered[i][mask] = frame[mask]

    print(f"Found {len(filtered_IDs)} IDs for {path}")

    save_path = path.replace(tracked_ending, filtered_mask_ending)
    np.savez(save_path, segm_mask_filtered)

def filter_cont_tracking(dir, tracked_ending, filtered_mask_ending):
    segm_paths = [os.path.join(dir, folder, subfolder, "Images", tracked_ending) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder)) for subfolder in os.listdir(os.path.join(dir, folder)) if os.path.isdir(os.path.join(dir, folder, subfolder))]
    segm_paths = [path for path in segm_paths if path.endswith(tracked_ending)]
    segm_paths = [path for path in segm_paths if os.path.exists(path)]

    print(f"Found {len(segm_paths)} files to process:")
    [print(path) for path in segm_paths]

    partial_func = functools.partial(filter_vid,tracked_ending=tracked_ending,
                                     filter_length=filter_length,
                                     filtered_mask_ending=filtered_mask_ending)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(partial_func, path) for path in segm_paths]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering images"):
            try:
                future.result()
            except Exception as e:
                raise e

def get_speeds_vid(path, time_interval, pixel_size):
    segm_vid = np.load(path)
    segm_vid = segm_vid["arr_0"]

    if segm_vid.ndim == 2:
        return
    
    if segm_vid.shape[0] == 1:
        return

    unique_IDs = np.unique(segm_vid)
    unique_IDs = unique_IDs[unique_IDs != 0]
    index = np.arange(len(segm_vid))
    df = pd.DataFrame(index=index, columns=unique_IDs, dtype=np.float32)
    for i, lab in enumerate(segm_vid):
        regions = skimage.measure.regionprops(lab)

        if i == 0:
            centroids_last_lab = {region.label: region.centroid for region in regions}          
            continue

        for region in regions:
            if region.label not in centroids_last_lab.keys():
                continue
            
            centroid = region.centroid
            last_centroid = centroids_last_lab[region.label]
            distance = np.linalg.norm(np.array(centroid) - np.array(last_centroid))
            speed = np.float32(distance * pixel_size / time_interval)
            df.at[i, region.label] = speed

        centroids_last_lab = {region.label: region.centroid for region in regions}          

    save_path = path.replace(filtered_mask_ending, f".speeds.csv")
    df.index.rename("frame_i", inplace=True)
    df.to_csv(save_path)

def get_speeds(dir, filtered_mask_ending, time_interval, pixel_size):
    segm_paths = [os.path.join(dir, folder, subfolder, "Images", filtered_mask_ending) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder)) for subfolder in os.listdir(os.path.join(dir, folder)) if os.path.isdir(os.path.join(dir, folder, subfolder))]
    segm_paths = [path for path in segm_paths if path.endswith(filtered_mask_ending)]

    print(f"Found {len(segm_paths)} files to process:")
    [print(path) for path in segm_paths]

    partial_func = functools.partial(get_speeds_vid, time_interval=time_interval, pixel_size=pixel_size)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(partial_func, path) for path in segm_paths]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating speeds"):
            try:
                future.result()
            except Exception as e:
                raise e

def preprep_data(all_speeds):
   # Create a DataFrame for plotting
        plot_data = pd.DataFrame([(folder, speed) for folder, speeds in all_speeds.items() for speed in speeds], columns=["ATP concentration (µM)", "Speed µm/s"])
        plot_data["ATP concentration (µM)"] = plot_data["ATP concentration (µM)"].astype(int)
        plot_data["Speed µm/s"] = plot_data["Speed µm/s"].astype(float)
        plot_data = plot_data.sort_values(by="ATP concentration (µM)")
        plot_data = plot_data.dropna()
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
        plot_data = plot_data[plot_data["Speed µm/s"] > 0]
        return plot_data

def boxplot(dir, th_cutoff=1.5):
    all_speeds_cut = {}
    all_speeds_complete = {}
    for folder in os.listdir(dir):
        speeds = []
        speeds_complete = []
        if not os.path.isdir(os.path.join(dir, folder)):
            continue

        for subfolder in os.listdir(os.path.join(dir, folder)):
            speeds_path = os.path.join(dir, folder, subfolder, "Images", ".speeds.csv")
            if not os.path.exists(speeds_path):
                continue

            print(f"Processing {speeds_path}")
            df = pd.read_csv(speeds_path, index_col="frame_i")
            speeds_local = df.to_numpy().flatten()
            speeds_local = speeds_local[~np.isnan(speeds_local)]
            speeds_complete.extend(speeds_local.copy())
            
            lower_cutoff = np.percentile(speeds_local, 25) - th_cutoff * (np.percentile(speeds_local, 75) - np.percentile(speeds_local, 25))
            upper_cutoff = np.percentile(speeds_local, 75) + th_cutoff * (np.percentile(speeds_local, 75) - np.percentile(speeds_local, 25))
            speeds_local = speeds_local[(speeds_local > lower_cutoff) & (speeds_local < upper_cutoff)]
            speeds.extend(speeds_local)

            all_speeds_cut[folder] = speeds
            all_speeds_complete[folder] = speeds_complete

    plot_data = preprep_data(all_speeds_cut)
    plot_data_complete = preprep_data(all_speeds_complete)        

    def michaelis_menten(x, Km, Vmax):
        return Vmax * x / (Km + x)
    
    def lineweaver_burk(x, a, b):
        return a * x + b

    # ##################
    # fit cut dataq
    x = np.array(plot_data["ATP concentration (µM)"])
    y = np.array(plot_data["Speed µm/s"])
    p, cov = curve_fit(michaelis_menten, x, y,)
    unc = np.sqrt(np.diag(cov))
    print("michaelis_menten cut data")
    print(f"(Km: {p[0]:.4f} ± {unc[0]:.4f}) μM")
    print(f"(Vmax: {p[1]:.4f} ± {unc[1]:.4f}) μM/s")


    # same for complete data
    x = np.array(plot_data_complete["ATP concentration (µM)"])
    y = np.array(plot_data_complete["Speed µm/s"])   
    p_comp, cov_comp = curve_fit(michaelis_menten, x, y,)
    unc_comp = np.sqrt(np.diag(cov_comp))
    print("michaelis_menten complete data")
    print(f"Km: ({p_comp[0]:.4f} ± {unc_comp[0]:.4f}) μM")
    print(f"Vmax: ({p_comp[1]:.4f} ± {unc_comp[1]:.4f}) μM/s")

    ##################
    # Fit Lineweaver-Burk
    x = np.array(plot_data["ATP concentration (µM)"])
    y = np.array(plot_data["Speed µm/s"])

    inverse_x = 1 / x
    inverse_y = 1 / y
    b0 = 1 / p[1]
    a0 = p[0] * b0
    p0 = (a0, b0)

    p_lin_raw, cov_lin_raw = curve_fit(lineweaver_burk, inverse_x, inverse_y, p0=p0)

    a, b = p_lin_raw
    sigma_a, sigma_b = np.sqrt(np.diag(cov_lin_raw))
    v_max = 1 / b
    k_m = a / b
    sigma_vmax = sigma_b / b**2
    sigma_km = k_m * np.sqrt((sigma_a / a)**2 + (sigma_b / b)**2)

    print("Lineweaver-Burk Fit Results:")
    print(f"Km: ({k_m:.4f} ± {sigma_km:.4f}) ) μM")
    print(f"Vmax: ({v_max:.4f} ± {sigma_vmax:.4f}) μM/s")
    p_lin = np.array([k_m, v_max])
    unc_lin = np.array([sigma_km, sigma_vmax])


    # Fit Lineweaver-Burk complete data
    x = np.array(plot_data_complete["ATP concentration (µM)"])
    y = np.array(plot_data_complete["Speed µm/s"])

    inverse_x = 1 / x
    inverse_y = 1 / y
    b0 = 1 / p_comp[1]
    a0 = p_comp[0] * b0
    p0 = (a0, b0)

    p_lin_raw, cov_lin_raw = curve_fit(lineweaver_burk, inverse_x, inverse_y, p0=p0)

    a, b = p_lin_raw 
    sigma_a, sigma_b = np.sqrt(np.diag(cov_lin_raw))
    v_max = 1 / b
    k_m = a / b

    sigma_vmax = sigma_b / b**2
    sigma_km = k_m * np.sqrt((sigma_a / a)**2 + (sigma_b / b)**2)

    print("Lineweaver-Burk Fit Results:")
    print(f"Km: ({k_m:.4f} ± {sigma_km:.4f}) μM")
    print(f"Vmax: ({v_max:.4f} ± {sigma_vmax:.4f}) μM/s ")
    p_lin_comp = np.array([k_m, v_max])
    unc_lin_comp = np.array([sigma_km, sigma_vmax])

    # save the data
    plot_data.to_csv(os.path.join(dir, "plot_data.csv"))
    plot_data_complete.to_csv(os.path.join(dir, "plot_data_complete.csv"))

    # save the fit information
    with open(os.path.join(dir, "fit.txt"), "w") as f:
        f.write(f"Michaelis-Menten Cut Data\n")
        f.write(f"Km: ({p[0]} ± {unc[0]}) 1e-6 M\n")
        f.write(f"Vmax: ({p[1]} ± {unc[1]}) 1e-6 M/s\n\n")

        f.write(f"Michaelis-Menten Complete Data\n")
        f.write(f"Km: ({p_comp[0]} ± {unc_comp[0]}) 1e-6 M\n")
        f.write(f"Vmax: ({p_comp[1]} ± {unc_comp[1]}) 1e-6 M/s\n\n")

        f.write(f"Lineweaver-Burk Cut Data\n")
        f.write(f"Km: ({p_lin[0]} ± {unc_lin[0]}) 1e-6 M\n")
        f.write(f"Vmax: ({p_lin[1]} ± {unc_lin[1]}) 1e-6 M/s\n\n")

        f.write(f"Lineweaver-Burk Complete Data\n")
        f.write(f"Km: ({p_lin_comp[0]} ± {unc_lin_comp[0]}) 1e-6 M\n")
        f.write(f"Vmax: ({p_lin_comp[1]} ± {unc_lin_comp[1]}) 1e-6 M/s\n")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="ATP concentration (µM)", y="Speed µm/s", data=plot_data, linewidth = 2, showcaps = True, boxprops=dict(alpha=.3), width = 0.4, fliersize=0)
    sns.violinplot(x="ATP concentration (µM)", y="Speed µm/s", data=plot_data, inner=None, color=".8")#, alpha=0.6)

     # plot the fit
    ATP_concentrations = plot_data["ATP concentration (µM)"]
    x = np.unique(np.array(ATP_concentrations))
    x_plot = np.arange(len(x))
    y_fit  = michaelis_menten(x, *p)
    y_upper = michaelis_menten(x, p[0] + unc[0], p[1] + unc[1])
    y_lower = michaelis_menten(x, p[0] - unc[0], p[1] - unc[1])
    plt.plot(x_plot, y_fit, color="black", linestyle="--", label=f"Fit: Km = ({p[0]:.4f} pm {unc[0]:.4f}) μM, Vmax = ({p[1]:.4f} pm {unc[1]:.4f}) μM/s")
    plt.fill_between(x_plot, y_upper, y_lower, color="black", alpha=0.2)

    # also plot lb fit
    y_fit  = michaelis_menten(x, *p_lin)
    y_upper = michaelis_menten(x, p_lin[0] + unc_lin[0], p_lin[1] + unc_lin[1])
    y_lower = michaelis_menten(x, p_lin[0] - unc_lin[0], p_lin[1] - unc_lin[1])
    plt.plot(x_plot, y_fit, color="blue", linestyle="--", label=f"Linearized fit: Km = ({p_lin[0]:.4f} pm {unc_lin[1]:.4f}) μM, Vmax = ({p_lin[1]:.4f} pm {unc_lin[1]:.4f}) μM/s")
    plt.fill_between(x_plot, y_upper, y_lower, color="blue", alpha=0.2)

    # add n to the plot
    n = plot_data.groupby("ATP concentration (µM)").size()
    for i, txt in enumerate(n):
        plt.annotate(f"n = {txt}", (i, 0), xytext=(0, -20), textcoords="offset points", ha='center', va='bottom')

    plt.legend()
    plt.title("Speed of Actin in Relation to ATP Concentration with Outliers Removed")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(dir, "speeds.png"), dpi=300)

    ####################################
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="ATP concentration (µM)", y="Speed µm/s", data=plot_data_complete, linewidth = 2, showcaps = True, boxprops=dict(alpha=.3), width = 0.4, fliersize=0)
    sns.violinplot(x="ATP concentration (µM)", y="Speed µm/s", data=plot_data_complete, inner=None, color=".8")#, alpha=0.6)

    # plot the fit
    y_fit = michaelis_menten(x, *p_comp)
    y_upper = michaelis_menten(x, p_comp[0] + unc_comp[0], p_comp[1] + unc_comp[1])
    y_lower = michaelis_menten(x, p_comp[0] - unc_comp[0], p_comp[1] - unc_comp[1])
    plt.plot(x_plot, y_fit, color="black", linestyle="--", label=f"Fit: Km = ({p_comp[0]:.4f} pm {unc_comp[1]:.4f}) μM, Vmax = ({p_comp[1]:.4f} pm {unc_comp[1]:.4f}) μM/s")
    plt.fill_between(x_plot, y_upper, y_lower, color="black", alpha=0.2)

    # also plot lb fit
    y_fit = michaelis_menten(x, *p_lin_comp)
    y_upper = michaelis_menten(x, p_lin_comp[0] + unc_lin_comp[0], p_lin_comp[1] + unc_lin_comp[1])
    y_lower = michaelis_menten(x, p_lin_comp[0] - unc_lin_comp[0], p_lin_comp[1] - unc_lin_comp[1])
    plt.plot(x_plot, y_fit, color="blue", linestyle="--", label=f"Linearized fit: Km = ({p_lin_comp[0]:.4f} pm {unc_lin_comp[1]:.4f}) μM, Vmax = ({p_lin_comp[1]:.4f} pm {unc_lin_comp[1]:.4f}) μM/s")
    plt.fill_between(x_plot, y_upper, y_lower, color="blue", alpha=0.2)

    n = plot_data_complete.groupby("ATP concentration (µM)").size()
    for i, txt in enumerate(n):
        plt.annotate(f"n = {txt}", (i, 0), xytext=(0, -20), textcoords="offset points", ha='center', va='bottom')

    plt.legend()
    plt.title("Speed of Actin in Relation to ATP Concentration with Outliers")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(dir, "speeds_complete.png"), dpi=300)

if __name__ == "__main__":
    if restruc_for_acdc_toggle:
        restruc_for_acdc(dir, end_name)
    if segment_toggle:
        segment(dir, neighborhood, end_name, segm_ending)
    if filter_cont_tracking_toggle:
        filter_cont_tracking(dir, tracked_ending, filtered_mask_ending)
    if get_speed_toggle:
        get_speeds(dir, filtered_mask_ending, time_interval, pixel_size)
    if boxplot_toggle:
        boxplot(dir)
    if del_files_toggle and del_files_toggle_2:
        del_files(dir, del_end)
    if list_files_toggle:
        list_files(dir, list_end_name)