
CREATE TABLE public.calc_people_camera (
	id serial4 NOT NULL,
	"name" varchar(100) NULL,
	url varchar(200) NULL,
	usl_send_less int4 NULL,
	usl_send_more int4 NULL,
	usl_change_min int4 NULL,
	usl_norm_less int4 NULL,
	usl_norm_more int4 NULL,
	opisanie varchar(1024) NULL,
	active_st time NULL,
	active_end time NULL,
	id_class_yolo_coco int4 NULL,
	b_video_potok bool NULL,
	CONSTRAINT calc_people_camera_pkey PRIMARY KEY (id)
);


-- public.calc_people_camera_param определение

-- Drop table

-- DROP TABLE public.calc_people_camera_param;

CREATE TABLE public.calc_people_camera_param (
	id serial4 NOT NULL,
	"name" varchar(100) NULL,
	value varchar(100) NULL,
	CONSTRAINT calc_people_camera_param_pkey PRIMARY KEY (id)
);


-- public.tg_users определение

-- Drop table

-- DROP TABLE public.tg_users;

CREATE TABLE public.tg_users (
	id serial4 NOT NULL,
	tg_id int8 NULL,
	tg_username varchar(100) NULL,
	CONSTRAINT tg_users_pkey PRIMARY KEY (id)
);


-- public.calc_people_camera_cnt_people определение

-- Drop table

-- DROP TABLE public.calc_people_camera_cnt_people;

CREATE TABLE public.calc_people_camera_cnt_people (
	id serial4 NOT NULL,
	id_calc_people_camera int4 NULL,
	cnt_people int4 NOT NULL,
	date_time timestamp NULL,
	folder_name varchar(200) NULL,
	file_name varchar(100) NULL,
	b_add_mess_usl_more bool NULL,
	b_add_mess_usl_more_norm bool NULL,
	b_add_mess_usl_less bool NULL,
	b_add_mess_usl_less_norm bool NULL,
	CONSTRAINT calc_people_camera_cnt_people_pkey PRIMARY KEY (id),
	CONSTRAINT calc_people_camera_cnt_people__id_calc_people_camera_fkey FOREIGN KEY (id_calc_people_camera) REFERENCES public.calc_people_camera(id) ON DELETE CASCADE
);
