INSERT INTO public.calc_people_camera ("name",url,usl_send_less,usl_send_more,usl_change_min,usl_norm_less,usl_norm_more,opisanie,active_st,active_end,id_class_yolo_coco,b_video_potok) VALUES
	 ('Пермь - общепит','http://188.17.153.103:91/webcapture.jpg?command=snap&amp;channel=1',NULL,8,3,NULL,2,'Permskiy Kray Kondratovo 57.981390, 56.108330','11:20:00','23:59:59',NULL,NULL),
	 ('Копейск - автстанция','https://intercom-video-1.insit.ru/camera67-centr-vokzal/preview.jpg',NULL,8,2,2,3,'',NULL,NULL,NULL,NULL),
	 ('Копейск - автстанция','https://intercom-video-1.insit.ru/camera67-centr-vokzal/preview.jpg',NULL,3,2,2,NULL,'',NULL,NULL,2,NULL),
	 ('Копейск - пр. Победы, пр. Славы','https://intercom-video-1.insit.ru/camera53-centr-pobedy_slavy/preview.jpg',NULL,5,2,2,NULL,'',NULL,NULL,2,NULL);
INSERT INTO public.calc_people_camera_param ("name",value) VALUES
	 ('Периодичность опроса - сек','60'),
	 ('Мин интервал оповещения - сек','3600'),
	 ('Макс число картинок по камере','100'),
	 ('Адресат - Telegramm','Alexandr_4352');
INSERT INTO public.tg_users (tg_id,tg_username) VALUES
	 (680901518,'Alexandr_4352');
