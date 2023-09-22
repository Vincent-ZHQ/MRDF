import csv
import os
import pandas as pd

# data_file = '/scratch/users/ntu/heqing00/Research/Dataset/FakeAVCeleb_v1.2/meta_data.csv'
# data = pd.read_csv(data_file,  dtype={'source': str,'target1': str,'target2': str,'method': str,'category': str,'type': str,'race': str,'gender': str, 'vid':str, 'path':str})
# print(data.loc[21565])
#
# fold_path = '/scratch/users/ntu/heqing00/Research/Dataset/FakeAVCeleb_v1.2'
# for i in range(21566):
#     file_path = os.path.join(fold_path, data.loc[i]['path'], data.loc[i]['vid'])
#     file_path = file_path.replace(' (', '-')
#     file_path = file_path.replace(')', '')
#     new_path = file_path.replace('.mp4', '.wav')
#     cmd = "ffmpeg -y -i " + file_path + " -ac 1 -ar 16000 "  + new_path
#     if not os.system(cmd):
#         print(file_path)


# data_file = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/metadata.csv'
# data = pd.read_csv(data_file,  dtype={'filename': str,'label': str,'audio_label': str,'video_label': str,'original': str})
# print(data.loc[119145])
#
# fold_path = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/train'
# for i in range(119146):
#     file_path = os.path.join(fold_path, data.loc[i]['filename'])
#     new_path = file_path.replace('.mp4', '.wav')
#     cmd = "ffmpeg -y -i " + file_path + " -ac 1 -ar 16000 "  + new_path
#     if not os.system(cmd):
#         print(file_path)


# data_file = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/val_labels.csv'
# data = pd.read_csv(data_file,  dtype={'filename': str,'label': str})
# print(data.loc[3999])
#
# fold_path = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/val'
# for i in range(4000):
#     file_path = os.path.join(fold_path, data.loc[i]['filename'])
#     new_path = file_path.replace('.mp4', '.wav')
#     cmd = "ffmpeg -y -i " + file_path + " -ac 1 -ar 16000 "  + new_path
#     if not os.system(cmd):
#         print(file_path)







import subprocess
import sys
import re

def get_video_duration(video_file):
    try:
        cmd = ["ffprobe", "-i", video_file, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        duration = float(output.strip())
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def generate_silence_wav(video_file, output_wav):
    duration = get_video_duration(video_file)
    if duration is not None:
        subprocess.run(["ffmpeg", "-f", "lavfi", "-y", "-i", f"anullsrc=r=16000:cl=stereo", "-t", str(duration), "-acodec", "pcm_s16le", output_wav])

if __name__ == "__main__":
    data_file = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/metadata.csv'
    data = pd.read_csv(data_file, dtype={'filename': str, 'label': str})
    print(data.loc[119145])

    fold_path = '/scratch/users/ntu/heqing00/Research/Dataset/DFDC/train'

    for i in range(119146):
        file_path = os.path.join(fold_path, data.loc[i]['filename'])
        new_path = file_path.replace('.mp4', '.wav')
        generate_silence_wav(file_path, new_path)
        cmd = "ffmpeg -y -i " + file_path + " -vn -acodec pcm_s16le -ac 1 -ar 16000 " + new_path
        if not os.system(cmd):
            print(file_path)


