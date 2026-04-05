#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import threading
import subprocess
from queue import Queue, Empty

# ---- NumPy compatibility shim (for old deps like skvideo) ----
if not hasattr(np, "float"):
	np.float = float
if not hasattr(np, "int"):
	np.int = int
if not hasattr(np, "bool"):
	np.bool = bool
if not hasattr(np, "complex"):
	np.complex = complex
if not hasattr(np, "object"):
	np.object = object
# --------------------------------------------------------------

import skvideo.io
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def run_ffmpeg(args_list):
	"""Run ffmpeg with subprocess, raise on failure."""
	result = subprocess.run(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result.returncode != 0:
		raise RuntimeError(
			f"ffmpeg failed ({result.returncode}): {' '.join(args_list)}\n"
			f"stderr:\n{result.stderr.decode(errors='ignore')}"
		)

def transferAudio(sourceVideo, targetVideo):
	import shutil

	temp_dir = "./temp"
	os.makedirs(temp_dir, exist_ok=True)

	# 1) Try lossless audio copy (container must support stream-copy)
	audio_copy = os.path.join(temp_dir, "audio.mkv")
	run_ffmpeg(["ffmpeg", "-y", "-i", sourceVideo, "-c:a", "copy", "-vn", audio_copy])

	targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
	if os.path.exists(targetNoAudio):
		os.remove(targetNoAudio)
	os.rename(targetVideo, targetNoAudio)

	try:
		run_ffmpeg(["ffmpeg", "-y", "-i", targetNoAudio, "-i", audio_copy, "-c", "copy", targetVideo])
	except Exception:
		# 2) Fallback: transcode audio to AAC
		audio_aac = os.path.join(temp_dir, "audio.m4a")
		run_ffmpeg(["ffmpeg", "-y", "-i", sourceVideo, "-c:a", "aac", "-b:a", "160k", "-vn", audio_aac])
		try:
			run_ffmpeg(["ffmpeg", "-y", "-i", targetNoAudio, "-i", audio_aac, "-c", "copy", targetVideo])
			print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
		except Exception:
			# If still fails, keep the no-audio video
			os.rename(targetNoAudio, targetVideo)
			print("Audio transfer failed. Interpolated video will have no audio")
			shutil.rmtree(temp_dir, ignore_errors=True)
			return

	# success path — remove the audio-less video
	if os.path.exists(targetNoAudio):
		os.remove(targetNoAudio)

	# cleanup
	shutil.rmtree(temp_dir, ignore_errors=True)

parser = argparse.ArgumentParser(description='Interpolation for a pair of images / a video using RIFE')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None, help='Folder of 0000001.png, 0000002.png, ...')
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode (faster on Tensor Cores)')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='deprecated: remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='write PNG sequence instead of video')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1, help='2**exp = multi')
parser.add_argument('--multi', dest='multi', type=int, default=2, help='fps upscaling factor')

args = parser.parse_args()

if args.exp != 1:
	args.multi = (2 ** args.exp)

if (args.video is None) and (args.img is None):
	raise SystemExit("You must provide either --video <file> or --img <folder>")

if args.skip:
	print("`--skip` flag is abandoned, please refer to issue #207.")

if args.UHD and args.scale == 1.0:
	args.scale = 0.5

if args.scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
	raise SystemExit("Invalid --scale. Choose from [0.25, 0.5, 1.0, 2.0, 4.0]")

if args.img is not None:
	args.png = True  # image sequence mode => PNG output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	if args.fp16:
		torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
	model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

# ---- Input handling
fpsNotAssigned = False
if args.video is not None:
	if not os.path.exists(args.video):
		raise FileNotFoundError(f"Input video not found: {args.video}")

	videoCapture = cv2.VideoCapture(args.video)
	if not videoCapture.isOpened():
		raise RuntimeError(f"Failed to open video: {args.video}")

	fps = videoCapture.get(cv2.CAP_PROP_FPS) or 0.0
	tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	videoCapture.release()

	if args.fps is None:
		fpsNotAssigned = True
		args.fps = max(1.0, fps) * args.multi  # guard against 0 fps metadata
	else:
		fpsNotAssigned = False

	videogen = skvideo.io.vreader(args.video)
	try:
		lastframe = next(videogen)
	except StopIteration:
		raise RuntimeError("No frames found in input video.")

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video_path_wo_ext, in_ext = os.path.splitext(args.video)
	print(f'{video_path_wo_ext}.{args.ext}, {tot_frame} frames in total, {fps:.3f} FPS to {args.fps:.3f} FPS')
	if (not args.png) and fpsNotAssigned:
		print("The audio will be merged after interpolation process")
	else:
		print("Will not merge audio because using png or fps flag!")
else:
	# folder of PNGs: 0000001.png, 0000002.png, ...
	if not os.path.isdir(args.img):
		raise FileNotFoundError(f"Image folder not found: {args.img}")
	videogen_list = sorted(
		[f for f in os.listdir(args.img) if f.lower().endswith(".png")],
		key=lambda x: int(os.path.splitext(x)[0])
	)
	if not videogen_list:
		raise RuntimeError(f"No PNG frames found in folder: {args.img}")
	tot_frame = len(videogen_list)
	lastframe = cv2.imread(os.path.join(args.img, videogen_list[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
	videogen_list = videogen_list[1:]
	videogen = videogen_list  # reuse read loop
	# FPS is irrelevant for PNG output; args.png is already forced True

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
	os.makedirs('vid_out', exist_ok=True)
else:
	vid_out_name = args.output if args.output else f'{video_path_wo_ext}_{args.multi}X_{int(np.round(args.fps))}.{args.ext}'
	vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
	if not vid_out.isOpened():
		raise RuntimeError(f"Failed to open video writer for: {vid_out_name}")

# ---- Helpers
def pad_image(img):
	if args.fp16:
		return F.pad(img, padding).half()
	return F.pad(img, padding)

def build_read_buffer(user_args, read_buffer, vg):
	try:
		if user_args.video is not None:
			for frame in vg:
				if user_args.montage:
					frame = frame[:, left: left + w]
				read_buffer.put(frame)
		else:
			# folder of PNGs
			for f in vg:
				frame = cv2.imread(os.path.join(user_args.img, f), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
				if user_args.montage:
					frame = frame[:, left: left + w]
				read_buffer.put(frame)
	except Exception:
		pass
	finally:
		read_buffer.put(None)

def clear_write_buffer(user_args, write_buffer, writer):
	cnt = 0
	while True:
		item = write_buffer.get()
		if item is None:
			break
		if user_args.png:
			cv2.imwrite(f'vid_out/{cnt:0>7d}.png', item[:, :, ::-1])
			cnt += 1
		else:
			writer.write(item[:, :, ::-1])

def make_inference(I0, I1, n):
	global model
	if model.version >= 3.9:
		res = []
		for i in range(n):
			res.append(model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), args.scale))
		return res
	else:
		middle = model.inference(I0, I1, args.scale)
		if n == 1:
			return [middle]
		first_half = make_inference(I0, middle, n=n // 2)
		second_half = make_inference(middle, I1, n=n // 2)
		if n % 2:
			return [*first_half, middle, *second_half]
		else:
			return [*first_half, *second_half]

# ---- Geometry & progress
if args.montage:
	left = w // 4
	w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

pbar = tqdm(total=tot_frame, desc="Interpolating", unit="frame")
if args.montage:
	lastframe = lastframe[:, left: left + w]

# ---- Queues & threads
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)

reader = threading.Thread(target=build_read_buffer, args=(args, read_buffer, videogen), daemon=True)
writer = threading.Thread(target=clear_write_buffer, args=(args, write_buffer, vid_out), daemon=True)

reader.start()
writer.start()

# ---- Main loop
I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
I1 = pad_image(I1)
temp = None  # save lastframe when processing static frame

while True:
	if temp is not None:
		frame = temp
		temp = None
	else:
		frame = read_buffer.get()
	if frame is None:
		break

	I0 = I1
	I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
	I1 = pad_image(I1)

	I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
	I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
	ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

	break_flag = False
	if ssim > 0.996:
		# Potential static/redundant frame, peek next
		frame2 = read_buffer.get()
		if frame2 is None:
			break_flag = True
			frame2 = lastframe
		else:
			temp = frame2

		I1 = torch.from_numpy(np.transpose(frame2, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
		I1 = pad_image(I1)
		I1 = model.inference(I0, I1, scale=args.scale)

		I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
		ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
		frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

	if ssim < 0.2:
		# If frames are very dissimilar, fill with previous frames to keep timing
		output = [I0 for _ in range(args.multi - 1)]
	else:
		output = make_inference(I0, I1, args.multi - 1)

	if args.montage:
		write_buffer.put(np.concatenate((lastframe, lastframe), 1))
		for mid in output:
			mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
			write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
	else:
		write_buffer.put(lastframe)
		for mid in output:
			mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
			write_buffer.put(mid[:h, :w])

	pbar.update(1)
	lastframe = frame
	if break_flag:
		break

# flush tail frame(s)
if args.montage:
	write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
	write_buffer.put(lastframe)
write_buffer.put(None)

# Wait for threads to finish
reader.join()
writer.join()

pbar.close()
if vid_out is not None:
	vid_out.release()

# ---- Audio merge (only for video path, original-fps case)
if (not args.png) and fpsNotAssigned and (args.video is not None):
	try:
		transferAudio(args.video, vid_out_name)
	except Exception as e:
		print(f"Audio transfer failed. Interpolated video will have no audio. Reason: {e}")
		targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
		if os.path.exists(targetNoAudio):
			os.rename(targetNoAudio, vid_out_name)

# Optional: free VRAM if running multiple times in one session
del model
if torch.cuda.is_available():
	torch.cuda.empty_cache()
