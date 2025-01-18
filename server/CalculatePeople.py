import os
from ultralytics import YOLO
import cv2
import telebot
import numpy as np
import psycopg2
from datetime import datetime
from datetime import time as dt_time
import time
import requests
import shutil
from sklearn.cluster import KMeans

host_ = '109.71.243.170'
database_ = 'calc_people_camera'
user_ = 'gen_user'
password_ = 'rftk56^67'
port_ = '5432'

TOKEN = '7067388213:AAEWGOHb1uOzmlOzczYbKYx8_D5IG57W6rs'
bot = telebot.TeleBot(TOKEN)

#model_yolo = YOLO('yolov8x.pt')
model_yolo = YOLO('yolov8x-seg.pt')  # Модель сегментации


class Konstant_class:
    name = ''
    value = ''


class Kamera_class:
    id = 0
    name = ''
    url = ''
    active_st = dt_time(hour=0, minute=0, second=0)  # начало действия
    active_end = dt_time(hour=0, minute=0, second=0)  # окончание действия
    usl_send_less = 0   # сообщение уйдёт если значение будет меньше заданного
    usl_send_more = 0   # сообщение уйдёт если значение будет больше заданного
    usl_change_min = 0  # ухудшение ситуации на величину
    usl_norm_less = 0   # нормализация условий после снижения
    usl_norm_more = 0   # нормализация условий псле превышения
    cnt_people = 0      # сколько человек определено на картинке
    folder_name = ''    # папка куда загружен файл
    file_name = ''      # имя загруженного файла с картинкой
    file_name_obr = ''  # имя обработанного фала (расчерчены люди)
    # условия на превышение
    last_send_more_usl_dt = datetime(1900, 1, 1)   # посл сообщ по условиям
    last_send_more_usl_cnt = 0   # количество людей в последнем сообщении
    last_send_more_norm_dt = datetime(1900, 1, 1)  # посл сообщ по нормал
    last_send_more_norm_cnt = 0  # кол-во лиц в последнем сообщении по нормал
    # условия на снижение
    last_send_less_usl_dt = datetime(1900, 1, 1)   # посл сообщ по условиям
    last_send_less_usl_cnt = 0   # количество людей в последнем сообщении
    last_send_less_norm_dt = datetime(1900, 1, 1)  # посл сообщ по нормал
    last_send_less_norm_cnt = 0  # кол-во лиц в последнем сообщении по нормал
    # Создать сообщения для услолвия превышения:
    b_add_mess_usl_more = 0       # создать сообщение для условия превышения
    b_add_mess_usl_more_norm = 0  # создать сообщение о нормализации
    # Создать сообщения для услолвия снижения:
    b_add_mess_usl_less = 0       # создать сообщение для условия превышения
    b_add_mess_usl_less_norm = 0  # создать сообщение о нормализации
    id_class_yolo_coco = 0        # класс в йоло, который выявлят будем

# функция по выделению людей на картинке
def find_people(kamera):
    if kamera.file_name == 'error':
        kamera.cnt_people = -1
        return

    # Загрузка изображения
    image = cv2.imread(kamera.folder_name + '/' + kamera.file_name)
    
    #height, width, channels = image.shape

    # Подготовка изображения для детекции
    #blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0),
    #                             True, crop=False)
    #net.setInput(blob)
    #outs = net.forward(output_layers)

    results = model_yolo(source=image,  
                        imgsz=max(image.shape[:2]))  # максимальный из размеров входного изображения (оригинальный размер)
    
    # Обнаружение объектов на изображении
    class_ids = [] # класс объекта
    confidences = []
    boxes = [] # область где нарисован объект
    masks = []
    for r in results:
        for box_num, box in enumerate(r.boxes):
            class_id = int(box.cls[0])  # Класс (0 - человек, 2 - авто)
            confidence = float(box.conf[0])  # Уверенность
            if class_id == kamera.id_class_yolo_coco and confidence > 0.2:  # Если обнаружен человек (вероятность выше 0,2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Верхний левый и нижний правый угол
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                masks.append(r.masks.data[box_num])
                    
            

    # Отрисовка рамки вокруг людей на изображении
    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        #if i in indexes:
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        slabel = 'class - ' + str(kamera.id_class_yolo_coco)
        if kamera.id_class_yolo_coco == 0: slabel = 'person'
        if kamera.id_class_yolo_coco == 2: 
            slabel = 'car'
            
            
            mask_np = masks[i].cpu().numpy().astype('uint8')
            # Находим контуры
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Рисуем контуры на изображении
            for contour in contours:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Красные контуры
            binary_mask = np.zeros(mask_np.shape, dtype=np.uint8)
            # Рисуем полигон, заполняя его белым цветом на бинарной маске
            cv2.drawContours(binary_mask, contours, -1, 255, thickness=cv2.FILLED)
            
            # Находим координаты всех ненулевых точек (т.е., внутри контура)
            points = np.column_stack(np.where(binary_mask > 0))  # Строки и столбцы ненулевых пикселей
            image_object = np.zeros_like(image)
            # Распаковываем координаты points (строки и столбцы)
            rows, cols = points[:, 0], points[:, 1]
            # Заполняем только нужные точки из исходного изображения
            image_object[rows, cols] = image[rows, cols]
            
            color_car_rgb = get_color_car(image_object[y1:y2, x1:x2], kamera) # функция для определения цвета авто
            cv2.rectangle(image, (x2, y1), (x2 + 20, y1 + 20), color_car_rgb[::-1], -1)
            rgb_text = f" RGB({color_car_rgb[0]}, {color_car_rgb[1]}, {color_car_rgb[2]})"
            
            

        slabel = slabel + str(round(confidences[i], 2))
            
        if kamera.id_class_yolo_coco == 2:
            slabel =  slabel + rgb_text
        cv2.putText(image, slabel, (x1, y1), font, 1, (255, 0, 0), 2)

    kamera.cnt_people = len(boxes)

    # Сохраняем изображение
    file_name_new = kamera.file_name[:-4] + '__' + str(kamera.cnt_people)
    prefix = ''
    i = 0
    while os.path.exists(kamera.folder_name + '/' +
                         file_name_new + prefix + '.jpg'):
        i += 1
        prefix = '_' + str(i)

    kamera.file_name_obr = file_name_new + prefix + '.jpg'
    cv2.imwrite(kamera.folder_name + '/' + kamera.file_name_obr, image)

    kamera.file_name_obr = kamera.file_name_obr


# находим цвет авто
def get_color_car(image, kamera):
    
    # Преобразуем изображение в цветовую модель LAB
    filtered_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # сглаживаем изображение
    filtered_lab = cv2.medianBlur(filtered_lab, 5)  # Применяем медианный фильтр
    
    # Преобразуем пиксели в одномерный массив для кластеризации
    pixels = filtered_lab.reshape((-1, 3))  # Преобразуем в BGR формат
    pixels = np.float32(pixels)
    # цвет должен быть "осовным", поэтому боьшоре количество кластеров будет избыточным
    # попробуем 5
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pixels)
    
    # Найдём самый большой кластер
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    
    # Средний цвет доминирующего кластера (LAB)
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    
    # Преобразование из Lab в BGR
    dominant_color_bgr = cv2.cvtColor(
        np.uint8([[dominant_color]]), cv2.COLOR_LAB2BGR)[0][0]
    
    # Преобразование из BGR в RGB
    dominant_color_rgb = cv2.cvtColor(
        np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2RGB)[0][0]
    # Возвращаем цвет кластера
    return tuple(map(int, dominant_color_rgb))

    
# проверка является ли значение числом
def is_number(test_value):
    test_value = str(test_value)
    if test_value.isdigit():
        return True
    if len(test_value) > 1:
        if test_value[0].startswith('-') and test_value[1:].isdigit():
            return True
    try:
        float(test_value)  # Пробуем преобразовать строку в число
        return True
    except ValueError:
        return False


# РАБОТА С КОНСТАНТАМИ
# получаем все константы из базы
def get_all_konstant(conn):
    konstants = []
    cur = conn.cursor()
    sql = """ select name, value
                 from public.calc_people_camera_param
                 """

    cur.execute(sql)
    conn.commit()
    rows = cur.fetchall()
    for row in rows:
        konstant = Konstant_class()
        konstant.name = row[0]
        konstant.value = row[1]
        konstants.append(konstant)
    cur.close()

    return konstants


# получаем значене константы
# type_out 1 - строка, 2 - число, 3 - дата
def get_konstant(konstant_name, konstants, type_out):
    for konstant in konstants:
        if konstant_name == konstant.name:
            if type_out == 1:
                return konstant.value
            elif type_out == 2:
                if is_number(konstant.value):
                    return int(konstant.value)
            elif type_out == 3:
                try:
                    return datetime.strptime(konstant.value, '%d.%m.%Y')
                except Exception as e:
                    print(f"Ошибка преобразования в дату: {e}")
                    return ''
    return ''


# Получаем список опрашиваемых камер
def get_spisok_kamer(conn):
    kameras = []
    cur = conn.cursor()
    sql = """
        select C.id, C.name, C.url,
               C.active_st, C.active_end,
               C.usl_send_less, C.usl_send_more,
               C.usl_change_min, C.usl_norm_less, C.usl_norm_more,
               t_usl_more_last.date_time last_send_more_usl_dt,
               t_usl_more_last.cnt_people last_send_more_usl_cnt,
               t_more_norm_last.date_time last_send_more_norm_dt,
               t_more_norm_last.cnt_people last_send_more_norm_cnt,
               t_usl_less_last.date_time last_send_less_usl_dt,
               t_usl_less_last.cnt_people last_send_less_usl_cnt,
               t_less_norm_last.date_time last_send_less_norm_dt,
               t_less_norm_last.cnt_people last_send_less_norm_cnt,
               C.id_class_yolo_coco
        from calc_people_camera C

        left join (select tc.id_calc_people_camera, max(id) id_max
                     from calc_people_camera_cnt_people tc
                     where tc.b_add_mess_usl_more = True
                     group by tc.id_calc_people_camera) t_usl_more_last_id on
                                t_usl_more_last_id.id_calc_people_camera = C.id
          left join calc_people_camera_cnt_people t_usl_more_last on
                                t_usl_more_last.id = t_usl_more_last_id.id_max

        left join (select tc.id_calc_people_camera, max(id) id_max
                     from calc_people_camera_cnt_people tc
                     where tc.b_add_mess_usl_more_norm = True
                     group by tc.id_calc_people_camera) t_usl_more_norm_id on
                                t_usl_more_norm_id.id_calc_people_camera = C.id
        left join calc_people_camera_cnt_people t_more_norm_last on
                                t_more_norm_last.id = t_usl_more_norm_id.id_max

              left join (select tc.id_calc_people_camera, max(id) id_max
                     from calc_people_camera_cnt_people tc
                     where tc.b_add_mess_usl_less = True
                     group by tc.id_calc_people_camera) t_usl_less_last_id on
                                t_usl_less_last_id.id_calc_people_camera = C.id
          left join calc_people_camera_cnt_people t_usl_less_last on
                                t_usl_less_last.id = t_usl_less_last_id.id_max

        left join (select tc.id_calc_people_camera, max(id) id_max
                     from calc_people_camera_cnt_people tc
                     where tc.b_add_mess_usl_less_norm = True
                     group by tc.id_calc_people_camera) t_usl_less_norm_id on
                                t_usl_less_norm_id.id_calc_people_camera = C.id
          left join calc_people_camera_cnt_people t_less_norm_last on
                                t_less_norm_last.id = t_usl_less_norm_id.id_max
        """

    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        kamera = Kamera_class()
        kamera.id = row[0]
        kamera.name = row[1]
        kamera.url = row[2]
        if row[3] is not None:
            kamera.active_st = row[3]
        else:
            kamera.active_st = dt_time(hour=0, minute=0, second=0)
        if row[4] is not None:
            kamera.active_end = row[4]
        else:
            kamera.active_end = dt_time(hour=0, minute=0, second=0)
        if row[5] is not None:
            kamera.usl_send_less = row[5]
        else:
            kamera.usl_send_less = -1
        if row[6] is not None:
            kamera.usl_send_more = row[6]
        else:
            kamera.usl_send_more = -1
        if row[7] is not None:
            kamera.usl_change_min = row[7]
        else:
            kamera.usl_change_min = -1

        if row[8] is not None:
            kamera.usl_norm_less = row[8]
        else:
            kamera.usl_norm_less = -1

        if row[9] is not None:
            kamera.usl_norm_more = row[9]
        else:
            kamera.usl_norm_more = -1

        if row[10] is not None:
            kamera.last_send_more_usl_dt = row[10]
        else:
            kamera.last_send_more_usl_dt = datetime(1900, 1, 1)

        if row[11] is not None:
            kamera.last_send_more_usl_cnt = row[11]
        else:
            kamera.last_send_more_usl_cnt = -1

        if row[12] is not None:
            kamera.last_send_more_norm_dt = row[12]
        else:
            kamera.last_send_more_norm_dt = datetime(1900, 1, 1)

        if row[13] is not None:
            kamera.last_send_more_norm_cnt = row[13]
        else:
            kamera.last_send_more_norm_cnt = -1

        if row[14] is not None:
            kamera.last_send_less_usl_dt = row[14]
        else:
            kamera.last_send_less_usl_dt = datetime(1900, 1, 1)

        if row[15] is not None:
            kamera.last_send_less_usl_cnt = row[15]
        else:
            kamera.last_send_less_usl_cnt = -1

        if row[16] is not None:
            kamera.last_send_less_norm_dt = row[16]
        else:
            kamera.last_send_less_norm_dt = datetime(1900, 1, 1)

        if row[17] is not None:
            kamera.last_send_less_norm_cnt = row[17]
        else:
            kamera.last_send_less_norm_cnt = -1

        if row[18] is not None:
            kamera.id_class_yolo_coco = row[18]
        else:
            kamera.id_class_yolo_coco = 0
        
        
        kameras.append(kamera)
    cur.close()

    return kameras


def add_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# копируем картинки с камер на диск
def get_pic_from_camera(spisok_kamer):
    for i, kamera in enumerate(spisok_kamer):
        try:
            headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    }
            response = requests.get(kamera.url, stream=True, headers=headers)
            #print(kamera.url)
            #print (response.status_code)
            # Проверка успешности запроса
            if response.status_code == 200:
                # Открытие файла для сохранения картинки
                folder_name = 'photo_camera/' + "{:04d}".format(kamera.id)
                add_folder(folder_name)
                file_name = '{:04d}'.format(kamera.id) + '_' + \
                            datetime.now().strftime('%Y-%m-%d__%H_%M_%S') + '.jpg'
                # папка куда загружен файл
                spisok_kamer[i].folder_name = folder_name
                spisok_kamer[i].file_name = file_name
                with open(folder_name + '/' + file_name, 'wb') as file:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, file)
            else:
                spisok_kamer[i].folder_name = 'error'
                spisok_kamer[i].file_name = 'error'    
        except:
            spisok_kamer[i].folder_name = 'error'
            spisok_kamer[i].file_name = 'error'

# удаляем файлы картинок при превышении заданного числа
def clear_photo_folder(spisok_kamer, cnt_file):

    for kamera in spisok_kamer:
        if kamera.folder_name != 'error':
            files = []
            for item in sorted(os.listdir(kamera.folder_name)):
                if os.path.isfile(kamera.folder_name + '/' + item):
                    files.append(kamera.folder_name + '/' + item)
            if len(files) > cnt_file:
                for file in files[:len(files) - cnt_file]:
                    #print(file)
                    os.remove(file)


# проверка условий на необходимость направления сообщений
def usl_send_mess(spisok_kamer, konstants):
    min_inrerval_sec = get_konstant('Мин интервал оповещения - сек',
                                    konstants, 2)

    for i, kam in enumerate(spisok_kamer):
        if kam.cnt_people >= 0:
            # сообщение на выполнение условия превышения
            if kam.usl_send_more > 0 and kam.cnt_people > kam.usl_send_more and \
               (
                (
                    kam.active_st == dt_time(hour=0, minute=0, second=0) and
                    kam.active_end == dt_time(hour=0, minute=0, second=0)
                ) or
                (
                    kam.active_st <= datetime.now().time() and
                    kam.active_end >= datetime.now().time()
                )
               ) and \
               (
                 kam.last_send_more_usl_dt == datetime(1900, 1, 1) or
                 (datetime.now() - kam.last_send_more_usl_dt).total_seconds() >
                 min_inrerval_sec or
                 (kam.last_send_more_norm_dt != datetime(1900, 1, 1) and
                  kam.last_send_more_norm_dt > kam.last_send_more_usl_dt) or
                 kam.cnt_people > kam.last_send_more_usl_cnt + kam.usl_change_min
               ):
                spisok_kamer[i].b_add_mess_usl_more = 1
            # сообщение на выполнения условия занижения
            if kam.usl_send_less > 0 and kam.cnt_people < kam.usl_send_less and \
               (
                (
                    kam.active_st == dt_time(hour=0, minute=0, second=0) and
                    kam.active_end == dt_time(hour=0, minute=0, second=0)
                ) or
                (
                    kam.active_st <= datetime.now().time() and
                    kam.active_end >= datetime.now().time()
                )
               ) and \
               (
                 kam.last_send_less_usl_dt == datetime(1900, 1, 1) or
                 (datetime.now() - kam.last_send_less_usl_dt).total_seconds() >=
                 min_inrerval_sec or
                 (kam.last_send_less_norm_dt != datetime(1900, 1, 1) and
                  kam.last_send_less_norm_dt > kam.last_send_less_usl_dt) or
                 kam.cnt_people < kam.last_send_less_usl_cnt - kam.usl_change_min
               ):
                spisok_kamer[i].b_add_mess_usl_less = 1
            # сообщение на выполнение условия нормализации при превышении
            if kam.usl_send_more > 0 and kam.cnt_people <= kam.usl_norm_more and \
               kam.last_send_more_usl_dt > datetime(1900, 1, 1) and \
               (
                (
                    kam.active_st == dt_time(hour=0, minute=0, second=0) and
                    kam.active_end == dt_time(hour=0, minute=0, second=0)
                ) or
                (
                    kam.active_st <= datetime.now().time() and
                    kam.active_end >= datetime.now().time()
                )
               ) and \
               (
                 kam.last_send_more_norm_dt == datetime(1900, 1, 1) or
                 kam.last_send_more_norm_dt < kam.last_send_more_usl_dt
               ):
                spisok_kamer[i].b_add_mess_usl_more_norm = 1
            # сообщение на выполнение условия нормализации при снижении
            if kam.usl_send_less > 0 and kam.cnt_people >= kam.usl_norm_less and \
               kam.last_send_less_usl_dt > datetime(1900, 1, 1) and \
               (
                (
                    kam.active_st == dt_time(hour=0, minute=0, second=0) and
                    kam.active_end == dt_time(hour=0, minute=0, second=0)
                ) or
                (
                    kam.active_st <= datetime.now().time() and
                    kam.active_end >= datetime.now().time()
                )
               ) and \
               (
                   kam.last_send_less_norm_dt == datetime(1900, 1, 1) or
                   kam.last_send_less_norm_dt < kam.last_send_less_usl_dt
               ):
                spisok_kamer[i].b_add_mess_usl_less_norm = 1


# запись результата анализачисленности в базу
def result_write_base(spisok_kamer, conn):
    for kam in spisok_kamer:
        b_add_mess_usl_more, b_add_mess_usl_more_norm = False, False
        b_add_mess_usl_less, b_add_mess_usl_less_norm = False, False
        if kam.b_add_mess_usl_more == 1:
            b_add_mess_usl_more = True
        if kam.b_add_mess_usl_more_norm == 1:
            b_add_mess_usl_more_norm = True
        if kam.b_add_mess_usl_less == 1:
            b_add_mess_usl_less = True
        if kam.b_add_mess_usl_less_norm == 1:
            b_add_mess_usl_less_norm = True

        sql = f""" INSERT INTO calc_people_camera_cnt_people
                      (id_calc_people_camera, cnt_people, date_time,
                       b_add_mess_usl_more, b_add_mess_usl_more_norm,
                       b_add_mess_usl_less, b_add_mess_usl_less_norm,
                       folder_name, file_name)
                   VALUES
                      ({kam.id}, {kam.cnt_people}, current_timestamp,
                       {b_add_mess_usl_more}, {b_add_mess_usl_more_norm},
                       {b_add_mess_usl_less}, {b_add_mess_usl_less_norm},
                       '{kam.folder_name}', '{kam.file_name_obr}'
                       )   """
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()


# направляем сообщения
def send_message(spisok_kamer, konstants, conn):

    # ищем шв кому отправлять
    tg_user_name = get_konstant('Адресат - Telegramm', konstants, 1)

    cur = conn.cursor()
    sql = f""" select tg_id
                 from public.tg_users
                 where tg_username = '{tg_user_name}'
                 """

    cur.execute(sql)
    conn.commit()
    rows = cur.fetchall()
    if len(rows) > 0:
        tg_id = rows[0][0]
        for kam in spisok_kamer:
            if kam.b_add_mess_usl_more == 1 or \
               kam.b_add_mess_usl_more_norm == 1 or\
               kam.b_add_mess_usl_less == 1 or \
               kam.b_add_mess_usl_less_norm == 1:
                sOut = str(kam.id) + ' - ' + kam.name
                sOut += ' кол-во лиц ' + str(kam.cnt_people)
                if kam.b_add_mess_usl_more == 1:
                    sOut += ' условие превышения'
                if kam.b_add_mess_usl_less == 1:
                    sOut += ' условие снижения'
                if kam.b_add_mess_usl_more_norm == 1:
                    sOut += ' возврат в норму после превышения'
                if kam.b_add_mess_usl_less_norm == 1:
                    sOut += ' возврат в норму после снижения'
                bot.send_message(tg_id, sOut, parse_mode='html')

                file_name = kam.folder_name + '/' + kam.file_name_obr
                if os.path.exists(file_name):
                    photo = open(file_name, 'rb')
                    bot.send_photo(tg_id, photo)
                else:
                    sOut = 'фото отсутствует'
                    bot.send_message(tg_id, sOut, parse_mode='html')

    cur.close()

    return konstants


def run_proccess():
    # параметры соединения с базой
    conn = psycopg2.connect(
        host=host_,
        database=database_,
        user=user_,
        password=password_,
        port=port_
        )
    # получаем константы
    konstants = get_all_konstant(conn)

    # получаем список камер (с которых будем брать картинки)
    spisok_kamer = get_spisok_kamer(conn)
    get_pic_from_camera(spisok_kamer)

    # считаем кол-во людей на кадом фото и сохраняем новые картинки
    for i in range(len(spisok_kamer)):
        find_people(spisok_kamer[i])

    # удаляем последние файлы при превышении заданного числа
    cnt_file = get_konstant('Макс число картинок по камере', konstants, 2)
    clear_photo_folder(spisok_kamer, cnt_file)
    # проверка условий на необходимость направления сообщений
    usl_send_mess(spisok_kamer, konstants)
    # собственно направляем сообщения
    send_message(spisok_kamer, konstants, conn)
    # записываем результата в базу
    result_write_base(spisok_kamer, conn)

    conn.close()

    # спать - и снова в работу (чтобы не замучить камеры и базу)
    sec_sleep = get_konstant('Периодичность опроса - сек', konstants, 2)
    time.sleep(sec_sleep)


# точка входа
def main():
    while 1 == 1:
        run_proccess()


main()
