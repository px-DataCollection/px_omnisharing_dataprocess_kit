import h5py
import os
import shutil
import argparse
import numpy as np
import json
import wave
import subprocess
from PIL import Image

def save_wav(path, sample_rate, data):
    """
    Save a NumPy int16 array to a .wav file.
    """
    if data.dtype != np.int16:
        raise ValueError("Audio data must be int16")
    if data.ndim == 1:
        nchannels = 1
        frames = data.tobytes()
    else:
        nchannels = data.shape[1]
        frames = data.reshape(-1).tobytes()
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(frames)

def parse_group(node, out_path):
    """
    Recursively parse an HDF5 group or dataset
    """
    if isinstance(node, h5py.Dataset):
        data = node[()]
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            if data.ndim <= 2:
                path = out_path + '.txt'
                arr = data if data.ndim == 1 else data.reshape(data.shape[0], -1)
                np.savetxt(path, arr, fmt='%.6f', delimiter=' ')
            else:
                np.save(out_path + '.npy', data)
        elif data.dtype.kind in ('S', 'U', 'O') or isinstance(data, (bytes, str)):
            path = out_path + '.txt'
            with open(path, 'w', encoding='utf-8') as f:
                if data.shape == ():
                    s = data.decode('utf-8') if isinstance(data, (bytes, np.bytes_)) else str(data)
                    f.write(s)
                else:
                    for x in data:
                        s = x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x)
                        f.write(f"{s}\n")
        else:
            np.save(out_path + '.npy', data)

        return

    os.makedirs(out_path, exist_ok=True)
    for name, item in node.items():
        parse_group(item, os.path.join(out_path, name))


def parse_processed_h5(h5_path: str, results_dir: str, return_vision: bool = True):
    """
    Parse ONLY the videos from raw input
    """
    with h5py.File(h5_path, 'r') as h5f:
        if return_vision:
            image_group = h5f["dataset/observation/image"]
            os.makedirs(results_dir, exist_ok=True)

            for video_name in image_group.keys():
                print('video_name', video_name)
                video_dir = os.path.join(results_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)

                if video_name.startswith("RGBD_"):  # Process color videos from RGBD cameras
                    d_color = image_group[f"{video_name}/color/data"][:]
                    
                    # Save the color data as .h265
                    color_raw_path = os.path.join(video_dir, f"{video_name}_color.h265")
                    with open(color_raw_path, "wb") as f:
                        f.write(d_color)

                    # Convert .h265 to .mp4 using ffmpeg 
                    color_mp4_path = os.path.join(video_dir, f"{video_name}_color.mp4")
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-fflags", "+genpts", 
                        "-r", "30", 
                        "-vsync", "2", 
                        "-i", color_raw_path,
                        "-c:v", "copy",
                        color_mp4_path
                    ], check=True)

                    os.remove(color_raw_path) 

                elif video_name.startswith("RGB_"):  # write .h265 into .mp4 files
                    if video_name in ("RGB_Camera5", "RGB_Camera7"):
                        continue
                    video_dir = os.path.join(results_dir, video_name)
                    os.makedirs(video_dir, exist_ok=True)

                    raw_path = os.path.join(video_dir, f"{video_name}.h265")
                    d = image_group[f"{video_name}/data"][:]
                    with open(raw_path, "wb") as ff:
                        ff.write(d)

                    mp4_path = os.path.join(video_dir, f"{video_name}.mp4")
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-fflags", "+genpts", 
                        "-r", "30", 
                        "-vsync", "2", 
                        "-i", raw_path,
                        "-c:v", "copy",
                        mp4_path
                    ], check=True)

                    os.remove(raw_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',  type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    for filename in os.listdir(args.source_dir):
        if not filename.endswith('.hdf5'):
            continue
        base = os.path.splitext(filename)[0]
        tgt = os.path.join(args.results_dir, base)
        if os.path.exists(tgt):
            print(f"Skipping existing: {base}")
            continue
        os.makedirs(tgt, exist_ok=True)
        print('Processing', filename)
        parse_processed_h5(os.path.join(args.source_dir, filename), tgt, return_vision=True)



