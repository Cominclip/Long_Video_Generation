import os
import cv2
import torch
import argparse
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image

def extract_last_frame(video_path, iteration, path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_to_capture = total_frames - 1
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
    ret, frame = video.read()
    if ret:
        image_path = path + 'frame_part' + str(iteration + 1) + '.jpg'
        cv2.imwrite(image_path, frame)
        print("Last frame saved at:", image_path)
    else:
        print("Failed to extract last frame")
    video.release()

def trim_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        print(f"Not enough video frames: {video_path}")
        return None
    
    frames = []
    for _ in range(frame_count - 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def concatenate_videos(video_paths, output_path, frame_rate):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_size = None
    all_frames = []
    for video_path in video_paths:
        frames = trim_last_frame(video_path)
        if frames is not None:
            all_frames.extend(frames)
            if video_size is None:
                video_size = (frames[0].shape[1], frames[0].shape[0])
    
    if len(all_frames) == 0:
        print("No valid video frames")
        return

    out = cv2.VideoWriter(output_path, fourcc, frame_rate, video_size)
    for frame in all_frames:
        out.write(frame)
    out.release()

def inference(meta, args):
    os.makedirs(meta['foldername'], exist_ok=True)
    # load pretrained models: t2i: stable diffusion xl; i2v: i2vgen-xl

    pipeline = StableVideoDiffusionPipeline.from_pretrained("/home/minkai/video_generation/models/stable-video-diffusion-img2vid-xt-1-1/", torch_dtype=torch.float16, variant="fp16").to("cuda")
    generator = torch.manual_seed(args.seed)

    all_video_paths = []
    for i in range(4):
        if i == 0:
            image = load_image(meta['foldername'] + 'people2.png').convert("RGB")
        else:
            image = load_image(meta['foldername'] + 'frame_part' + str(i) + '.jpg').convert("RGB")
        frames = pipeline(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
        save_path = meta['foldername'] + 'generated_part_' + str(i) + '.mp4'
        all_video_paths.append(save_path)
        export_to_video(frames, save_path, fps=args.fps)
        extract_last_frame(save_path, i, meta['foldername'])
    concatenate_videos(all_video_paths, meta['foldername'] + 'final.mp4', args.fps)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1222, help="random seed")
    parser.add_argument('--test_initial_prompt',action='store_true', help="use prompt to generate")
    parser.add_argument("--fps", type=int, default=16, help="fps of the generated video")
    args = parser.parse_args()

    meta = dict(
                foldername = 'generation/people2_svd/'
            )   

    inference(meta, args)
