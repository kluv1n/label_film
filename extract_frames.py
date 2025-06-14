import cv2
import os

def process_all_videos(base_folder, output_folder):
    for subfolder in os.listdir(base_folder):
        full_subfolder = os.path.join(base_folder, subfolder)
        if os.path.isdir(full_subfolder):
            for film in os.listdir(full_subfolder):
                if film.endswith(".mp4"):
                    film_dir = os.path.join(full_subfolder, film)
                    full_output_dir = os.path.join(output_folder, subfolder)
                    extract_video(film_dir, full_output_dir)

def extract_video(film_dir, full_oytput_dir):
    cap = cv2.VideoCapture(film_dir)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Не удалось определить FPS для: {film_dir}")
        return

    frame_interval = fps//2
    frame_count = 0
    saved_count = 0

    os.makedirs(full_oytput_dir)

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(full_oytput_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Сохранено {saved_count} кадров из {film_dir}")


process_all_videos("video/data_test_short", "frames/test_frames")
process_all_videos("video/data_train_short", "frames/train_frames")