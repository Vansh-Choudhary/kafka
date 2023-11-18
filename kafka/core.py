#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import kafka.globals
import kafka.metadata
import kafka.ui as ui
from kafka.predictor import predict_image, predict_video
from kafka.processors.frame.core import get_frame_processors_modules
from kafka.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{kafka.metadata.name} {kafka.metadata.version}')

    args = program.parse_args()

    kafka.globals.source_path = args.source_path
    kafka.globals.target_path = args.target_path
    kafka.globals.output_path = normalize_output_path(kafka.globals.source_path, kafka.globals.target_path, args.output_path)
    kafka.globals.headless = kafka.globals.source_path is not None and kafka.globals.target_path is not None and kafka.globals.output_path is not None
    kafka.globals.frame_processors = args.frame_processor
    kafka.globals.keep_fps = args.keep_fps
    kafka.globals.keep_frames = args.keep_frames
    kafka.globals.skip_audio = args.skip_audio
    kafka.globals.many_faces = args.many_faces
    kafka.globals.reference_face_position = args.reference_face_position
    kafka.globals.reference_frame_number = args.reference_frame_number
    kafka.globals.similar_face_distance = args.similar_face_distance
    kafka.globals.temp_frame_format = args.temp_frame_format
    kafka.globals.temp_frame_quality = args.temp_frame_quality
    kafka.globals.output_video_encoder = args.output_video_encoder
    kafka.globals.output_video_quality = args.output_video_quality
    kafka.globals.max_memory = args.max_memory
    kafka.globals.execution_providers = decode_execution_providers(args.execution_provider)
    kafka.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if kafka.globals.max_memory:
        memory = kafka.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = kafka.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'kafka.CORE') -> None:
    print(f'[{scope}] {message}')
    if not kafka.globals.headless:
        ui.update_status(message)


def start() -> None:
    for frame_processor in get_frame_processors_modules(kafka.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(kafka.globals.target_path):
        if predict_image(kafka.globals.target_path):
            destroy()
        shutil.copy2(kafka.globals.target_path, kafka.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(kafka.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(kafka.globals.source_path, kafka.globals.output_path, kafka.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(kafka.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if predict_video(kafka.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(kafka.globals.target_path)
    # extract frames
    if kafka.globals.keep_fps:
        fps = detect_fps(kafka.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(kafka.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(kafka.globals.target_path)
    # process frame
    temp_frame_paths = get_temp_frame_paths(kafka.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(kafka.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(kafka.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    # create video
    if kafka.globals.keep_fps:
        fps = detect_fps(kafka.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(kafka.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(kafka.globals.target_path)
    # handle audio
    if kafka.globals.skip_audio:
        move_temp(kafka.globals.target_path, kafka.globals.output_path)
        update_status('Skipping audio...')
    else:
        if kafka.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(kafka.globals.target_path, kafka.globals.output_path)
    # clean temp
    update_status('Cleaning temporary resources...')
    clean_temp(kafka.globals.target_path)
    # validate video
    if is_video(kafka.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if kafka.globals.target_path:
        clean_temp(kafka.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(kafka.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if kafka.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
