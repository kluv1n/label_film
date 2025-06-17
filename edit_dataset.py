import os
import shutil

def flatten_image_structure(class_dir):
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                full_path = os.path.join(root, file)
                # Целевой путь — переместить в корень class_dir
                target_path = os.path.join(class_dir, file)

                # Избегаем перезаписи
                if os.path.abspath(full_path) != os.path.abspath(target_path):
                    counter = 1
                    name, ext = os.path.splitext(file)
                    while os.path.exists(target_path):
                        new_name = f"{name}_{counter}{ext}"
                        target_path = os.path.join(class_dir, new_name)
                        counter += 1

                    shutil.move(full_path, target_path)

    # Удаление пустых папок
    for root, dirs, files in os.walk(class_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

# Пути к папкам
intro_path = "frames/train_frames_intro"
non_intro_path = "frames/train_frames_all"

# Обработка
flatten_image_structure(intro_path)
flatten_image_structure(non_intro_path)

print("Готово")
