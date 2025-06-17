import cv2
import os
import json
import traceback

def time_to_second(time):
    try:
        h, m, s = map(int, time.strip().split(":"))
        return h * 3600 + m * 60 + s
    except Exception as e:
        print(f"[ОШИБКА] Ошибка в формате времени '{time}': {e}")
        return 0

def extract_video(film_dir, full_output_dir, start_time, end_time, video_id):
    try:
        cap = cv2.VideoCapture(film_dir)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть видео: {film_dir}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"[ОШИБКА] FPS не определён для: {film_dir}")
            return

        frame_interval = max(int(fps) // 4, 1)
        frame_count = 0
        saved_count = 0

        os.makedirs(full_output_dir, exist_ok=True)

        while True:
            success, frame = cap.read()
            if not success:
                break

            current_time = frame_count / fps

            if current_time > end_time + (end_time - start_time + 15):
                break

            if frame_count % frame_interval == 0 and start_time + (end_time - start_time + 10) < current_time <= end_time + (end_time - start_time + 15):
                frame_filename = os.path.join(full_output_dir, f"{video_id}_all_frame_{saved_count:06d}.jpg")
                try:
                    cv2.imwrite(frame_filename, frame)
                    saved_count += 1
                except Exception as e:
                    print(f"[ПРЕДУПРЕЖДЕНИЕ] Ошибка при сохранении кадра: {e}")

            frame_count += 1

        cap.release()
        print(f"[ГОТОВО] Сохранено {saved_count} кадров из {film_dir}")

    except Exception as e:
        print(f"[ОШИБКА] Ошибка при обработке видео {film_dir}: {e}")
        print(traceback.format_exc())

def process_all_videos(base_folder, output_folder, labels):
    for subfolder in os.listdir(base_folder):
        full_subfolder = os.path.join(base_folder, subfolder)
        if os.path.isdir(full_subfolder):
            for film in os.listdir(full_subfolder):
                if film.endswith(".mp4"):
                    film_dir = os.path.join(full_subfolder, film)
                    full_output_dir = os.path.join(output_folder, subfolder)

                    video_id = os.path.splitext(film)[0]
                    if video_id not in labels:
                        print(f"[ПРОПУСК] {video_id} отсутствует в JSON, пропускаем.")
                        continue

                    try:
                        start_time = time_to_second(labels[video_id]["start"])
                        end_time = time_to_second(labels[video_id]["end"])

                        print(f"[ОБРАБОТКА] {film_dir}: с {labels[video_id]['start']} до {labels[video_id]['end']}")
                        extract_video(film_dir, full_output_dir, start_time, end_time, video_id)
                    except Exception as e:
                        print(f"[ОШИБКА] Ошибка при обработке {film_dir}: {e}")
                        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        with open("video/labels_json/test_labels.json", "r", encoding="utf-8") as f:
            labels_test = json.load(f)

        with open("video/labels_json/train_labels.json", "r", encoding="utf-8") as f:
            labels_train = json.load(f)

        #process_all_videos("video/data_test_short", "frames/test_frames_intro", labels_test)
        process_all_videos("video/data_train_short", "frames/train_frames_all", labels_train)

    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА] Ошибка при запуске: {e}")
        print(traceback.format_exc())
