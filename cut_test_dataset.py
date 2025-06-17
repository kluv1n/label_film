import cv2
import os
import json
import traceback

def extract_all_frames(video_path, output_dir, fps=1):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть видео: {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            print(f"[ОШИБКА] Не удалось определить FPS для: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        frame_interval = max(int(video_fps / fps), 1)
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_count = 0
        saved_count = 0

        print(f"[ИНФО] Обработка видео: {video_path}")
        print(f"[ИНФО] Длительность: {duration:.2f} сек, Всего кадров: {total_frames}")
        print(f"[ИНФО] Сохранение с частотой {fps} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                current_time = frame_count / video_fps
                hours = int(current_time // 3600)
                minutes = int((current_time % 3600) // 60)
                seconds = int(current_time % 60)

                frame_filename = os.path.join(
                    output_dir,
                    f"{video_name}_{hours:02d}-{minutes:02d}-{seconds:02d}_{frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1
            if frame_count >= total_frames:
                break

        cap.release()
        print(f"[УСПЕХ] Сохранено {saved_count} кадров из {video_path}\n")

    except Exception as e:
        print(f"[ОШИБКА] Ошибка при обработке видео {video_path}: {e}")
        print(traceback.format_exc())

def process_all_videos(input_dir, output_dir, target_fps=1):
    if not os.path.exists(input_dir):
        print(f"[ОШИБКА] Папка с видео не найдена: {input_dir}")
        return

    for video_file in os.listdir(input_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]

            video_output_dir = os.path.join(output_dir, video_name)
            extract_all_frames(video_path, video_output_dir, target_fps)

if __name__ == "__main__":
    try:
        INPUT_VIDEOS_DIR = "video/data_test_short"  # Папка с исходными видео
        OUTPUT_FRAMES_DIR = "frames/test_frames_all"  # Папка для сохранения кадров
        TARGET_FPS = 1                               # Частота кадров (1 кадр в секунду)

        process_all_videos(INPUT_VIDEOS_DIR, OUTPUT_FRAMES_DIR, TARGET_FPS)

    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА] Ошибка при запуске: {e}")
        print(traceback.format_exc())