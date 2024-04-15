import os
import cv2
import torch
import argparse
from diffusers import DiffusionPipeline
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

def extract_last_frame(video_path, iteration, path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_to_capture = total_frames - 2
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
    sdxl_base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    sdxl_refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=sdxl_base.text_encoder_2, vae=sdxl_base.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/I2VGen-XL", torch_dtype=torch.float16, variant="fp16").to("cuda")
    # pipeline.enable_model_cpu_offload()
    generator = torch.manual_seed(args.seed)

    if args.test_initial_prompt:
        prompt = meta['prompt']
    else:
        prompt = meta['llm_prompt']

    image = sdxl_base(prompt=prompt[0], width=1280, height=720, num_inference_steps=40, denoising_end=0.8, output_type="latent").images
    image = sdxl_refiner(prompt=prompt[0], num_inference_steps=40, denoising_start=0.8, image=image).images[0].save(meta['foldername'] + 'frame_part0.jpg')

    all_video_paths = []
    for i in range(len(prompt)):
        image = load_image(meta['foldername'] + 'frame_part' + str(i) + '.jpg').convert("RGB")
        frames = pipeline(
            prompt=prompt[0],
            image=image,
            height=720,
            width=1280,
            target_fps = args.fps,
            num_inference_steps=50,
            negative_prompt=meta['negative_prompt'],
            guidance_scale=9.0,
            generator=generator
        ).frames[0]
        save_path = meta['foldername'] + 'generated_part_' + str(i) + '.mp4'
        all_video_paths.append(save_path)
        export_to_video(frames, save_path, fps=args.fps)
        extract_last_frame(save_path, i, meta['foldername'])
    concatenate_videos(all_video_paths, meta['foldername'] + 'final.mp4', args.fps)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=522112, help="random seed")
    parser.add_argument('--test_initial_prompt',action='store_true', help="use prompt to generate")
    parser.add_argument("--fps", type=int, default=16, help="fps of the generated video")
    args = parser.parse_args()

    meta = dict(
                prompt = ["A grizzly bear hunting for fish in a river at the edge of a waterfall"],
                llm_prompt = ["In the scenic wilderness, a majestic grizzly bear stands at the edge of a breathtaking waterfall, surveying the rushing river below",
                              "With focused determination, the bear dives into the crystal-clear water, skillfully navigating the strong currents as it searches for fish",
                              "Using its powerful paws and sharp claws, the bear swiftly catches a leaping fish from the river, showcasing its exceptional hunting skills and primal strength"], 
                negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
                foldername = 'generation/bear_hunting_for_fish/'
            )   

    inference(meta, args)
