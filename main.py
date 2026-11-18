import os
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))

gt_dir = os.path.join(project_dir, "GT_formatted")
meas_dir = os.path.join(project_dir, "meas_formatted")

gt_files = os.listdir(gt_dir)
meas_files = os.listdir(meas_dir)

gt_data = []
meas_data = []

for file in gt_files:
    file_path = os.path.join(gt_dir, file)
    gt_data.append(np.load(file_path))

for file in meas_files:
    file_path = os.path.join(meas_dir, file)
    meas_data.append(np.load(file_path))
    

print(meas_files[0],meas_data[0].shape)    