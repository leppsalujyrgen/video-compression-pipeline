import subprocess
import os
import re
from typing import List
import cv2

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def get_image_paths(directories: List[str], sort: bool = False) -> List[str]:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_paths = []

    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

    if sort:
        image_paths.sort(key=natural_sort_key)

    return image_paths

def create_video(source_file: str, video_fps: int, codec: str, video_output_path: str, logging_enabled: bool = False, **kwargs) -> List[str]:
    print(f"Creating video...")
    if source_file.endswith(".txt"):
        command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", source_file, "-vf", f"fps={video_fps}", "-c:v", codec]
    elif source_file.endswith((".mp4", ".mkv", "avi")):
        command = ["ffmpeg", "-y", "-i", source_file, "-vf", f"fps={video_fps}", "-c:v", codec]
    
    if codec in ["libx264", "libx265"]:
        if "preset" in kwargs:
            command.extend(["-preset", kwargs["preset"]])
        if "tune" in kwargs:
            command.extend(["-tune", kwargs["tune"]])
        if "pix_fmt" in kwargs:
            command.extend(["-pix_fmt", kwargs["pix_fmt"]])
        if "bitrate" in kwargs:
            command.extend(["-b:v", kwargs["bitrate"]])
        if "crf" in kwargs:
            command.extend(["-crf", kwargs["crf"]])
    elif codec == "mjpeg":
        if "quality" in kwargs:
            command.extend(["-q:v", str(kwargs["quality"])])
        if "pix_fmt" in kwargs:
            command.extend(["-pix_fmt", kwargs["pix_fmt"]])
    command.append(video_output_path)
    print(f"- FFMPEG command created. Command: {command}")
    
    video_output_dir = os.path.dirname(video_output_path)
    os.makedirs(video_output_dir, exist_ok=True)
    if logging_enabled:
        log_file_path = video_output_path.split(".")[0] + "_ffmpeg_encoding_benchmark.log"
        log_file = open(log_file_path, "w") 
        log_file.write(f"ffmpeg command: {command}\n\n")
        print(f"- Logging enabled. Benchmarking log file: {log_file_path}")
        subprocess.run(command, check=True, stdout=log_file, stderr=log_file)
        log_file.close()
    else:
        subprocess.run(command, check=True)
    print(f"Created video: {video_output_path}")


def decode_video(input_video, output_dir, fps=10, logging_enabled=True):
    print("Decoding video...")
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = f"{output_dir}/%d.png"    
    # Run FFmpeg command using subprocess
    command = [
        'ffmpeg', '-i', input_video, '-vf', f'fps={fps}', output_pattern
    ]

    if logging_enabled:
        log_file_path = input_video.split(".")[0] + "_ffmpeg_decoding_benchmark.log"
        log_file = open(log_file_path, "w") 
        log_file.write(f"ffmpeg command: {command}\n\n")
        print(f"- Logging enabled. Benchmarking log file: {log_file_path}")
        subprocess.run(command, check=True, stdout=log_file, stderr=log_file)
        log_file.close()
    else:
        subprocess.run(command, check=True)
    print(f"Decoded frames: {output_dir}")
    


dataset_source_path = "/data/RUP_Data/CARLA_dataset/2025-02-07_nuscenes_video/train"
bag_dirs = os.listdir(dataset_source_path)
camera_dirs = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

dataset_bag_paths = [os.path.join(dataset_source_path, bag_dir) for bag_dir in  bag_dirs]
dataset_images = get_image_paths(dataset_bag_paths, sort=True)

for bag_dir in bag_dirs:
    for cam_dir in camera_dirs:
        camera_fps = 10
        camera_image_paths = [camera_image_path for camera_image_path in dataset_images if (f"/{cam_dir}/" in camera_image_path) and (f"/{bag_dir}/" in camera_image_path) and ("bevs" not in camera_image_path) and ("visibility_masks" not in camera_image_path) and ("fov_masks" not in camera_image_path) and ("/video/" not in camera_image_path)]
        camera_image_file = 'image_list.txt'
        print(f"Dataset containes {len(camera_image_paths)} for camera {cam_dir}.")
        with open(camera_image_file, 'w') as file:
            for image_path in camera_image_paths:
                file.write(f"file '{os.path.abspath(image_path)}'\n")
                file.write(f"duration {1/camera_fps}\n")
        
        ####
        ## Lossless video
        ####

        # Create FFV1 Video
        encoder_name = "ffv1"
        lossless_video_path = os.path.join(dataset_source_path, bag_dir, cam_dir, "video", f"{cam_dir}_ffv1.mkv")
        create_video(
            source_file=camera_image_file, 
            video_fps=camera_fps,  
            codec="ffv1",
            video_output_path=lossless_video_path,
            logging_enabled=True
        )


        ####
        ## Lossy videos
        ###

        # Create H.264 and H.265 Videos
        encoder_names = ["libx264", "libx265"] 
        video_pixel_formats = ["yuvj420p", "yuvj444p"]
        crf_values = ["2", "23", "51"]

        for encoder_name in encoder_names:
            for video_pixel_format in video_pixel_formats:
                for crf_value in crf_values:
                    
                    video_path = os.path.join(dataset_source_path, bag_dir, cam_dir, "video", f"{cam_dir}_{encoder_name}_ultrafast_zerolatency_crf{crf_value}_{video_pixel_format}.mp4")
                    create_video(
                        source_file=lossless_video_path, 
                        video_fps=camera_fps,  
                        codec=encoder_name,
                        video_output_path=video_path,
                        preset="ultrafast",
                        tune="zerolatency",
                        pix_fmt=video_pixel_format,
                        crf=crf_value,
                        logging_enabled= True
                    )
                    decoded_frames_output_dir = video_path.split(".")[0]
                    decode_video(
                        video_path, decoded_frames_output_dir
                    )

        # Create MJPEG Videos
        encoder_names = ["mjpeg"]
        video_pixel_formats = ["yuvj420p", "yuvj444p"]
        q_values = ["1", "16", "31"]

        for encoder_name in encoder_names:
            for video_pixel_format in video_pixel_formats:
                for q in q_values:
                    video_path = os.path.join(dataset_source_path, bag_dir, cam_dir, "video", f"{cam_dir}_{encoder_name}_q{q}_{video_pixel_format}.mp4")
                    create_video(
                        source_file=lossless_video_path, 
                        video_fps=camera_fps,  
                        codec=encoder_name,
                        video_output_path=video_path,
                        pix_fmt=video_pixel_format,
                        quality=q,
                        logging_enabled= True
                    )
                    decoded_frames_output_dir = video_path.split(".")[0]
                    decode_video(
                        video_path, decoded_frames_output_dir
                    )