# Author:   lsysue
# GitHub:   https://github.com/lsysue/Video-Face-Recognition
# Mail:     lsyxll268@163.com

# 进行人脸录入 / Face register
import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
from config import cfg

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = cfg.rgst_face_dir
        self.path_frames = cfg.rgst_frame_dir

        self.existing_faces_cnt = 0         # 已录入的人脸计数器 / cnt for counting saved faces
        self.ss_cnt = 0                     # 录入 personX 人脸时图片计数器 / cnt for screen shots
        self.current_frame_faces_cnt = 0    # 录入人脸计数器 / cnt for counting faces in current frame

        self.last_frame_faces_cnt = 0

        self.new_register = 1               # TODO:假设每一段视频只录入一个人脸
        self.save_flag = 1                  # 之后用来控制是否保存图像的 flag / The flag to control if save

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # frame id
        self.frame_id = 0

    # 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # 新建文件夹 / Create folders to save faces images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

        if os.path.isdir(self.path_frames):
            pass
        else:
            os.mkdir(self.path_frames)

    # 删除之前存的人脸数据文件夹 / Delete old face folders
    def pre_work_del_old_face_folders(self):
        # 删除之前存的人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile(cfg.root_dir + "data/known_features.csv"):
            os.remove(cfg.root_dir + "data/known_features.csv")

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir(cfg.rgst_face_dir):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir(cfg.rgst_face_dir)
            if len(person_list) == 0:
                self.existing_faces_cnt = 0
                return
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.existing_faces_cnt = 0

    # 获取处理之后 stream 的帧数 / Update FPS of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 获取人脸 / Main process of face detection and saving
    def process(self, stream):
        # 1. 新建储存人脸图像文件目录 / Create folders to save photos
        self.pre_work_mkdir()

        # 2. 删除 "/data/data_faces_from_camera" 中已有人脸图像文件 / Uncomment if want to delete the saved faces and start from person_1
        if os.path.isdir(self.path_frames):
            self.pre_work_del_old_frame_folders()

        # 3. 检查 "/data/data_faces_from_camera" 中已有人脸文件
        self.check_existing_faces_cnt()
        logging.info("系统中已录入人脸数 / Registered face num: %d",self.existing_faces_cnt)
        while stream.isOpened() and self.frame_id <= cfg.register_frame_num:
            flag, img_rd = stream.read()        # Get camera video stream
            # kk = cv2.waitKey(1)
            faces = detector(img_rd, 0)         # Use Dlib face detector

            self.last_frame_faces_cnt = self.current_frame_faces_cnt
            self.current_frame_faces_cnt = len(faces)
            
            if not self.frame_id % 10:
                cv2.imwrite(os.path.join(self.path_frames, '{}.jpg'.format(self.frame_id)),img_rd)
            self.frame_id += 1

            if len(faces) == 0:
                logging.info("No face detected")
                # cfg.register_frame_num += 1
            # 5. 检测到人脸 / Face detected
            if len(faces) != 0:
                if self.current_frame_faces_cnt > self.last_frame_faces_cnt:
                    logging.warning("出现新的人脸，新建人脸文件夹保存 / New faces detected")
                elif self.current_frame_faces_cnt < self.last_frame_faces_cnt:
                    logging.warning("人脸数减少")
                    
                # 矩形框 / Show the ROI of faces
                for d in faces:
                    # TODO: 默认运行程序时出现的人脸没有在数据库中
                    if self.new_register:
                        self.existing_faces_cnt += 1
                        current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                        os.makedirs(current_face_dir)
                        logging.info("\n%-40s %s", "新建的人脸文件夹 / Create folders:", current_face_dir)
                        self.ss_cnt = 0                 # 将人脸计数器清零 / Clear the cnt of screen shots

                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)

                    color_rectangle = (255, 255, 255)
                    save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    # 根据人脸大小生成空的图像 / Create blank image according to the size of face detected
                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)
                    for ii in range(height*2):
                        for jj in range(width*2):
                            img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]

                    if save_flag:
                        self.ss_cnt += 1
                        cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                        logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：", str(current_face_dir), str(self.ss_cnt))

                self.new_register = 0

            self.current_frame_faces_cnt = len(faces)

            # 11. Update FPS
            self.update_fps()

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")   # Get video stream from video file
        # cap = cv2.VideoCapture(0)               # Get video stream from camera
        cap = cv2.VideoCapture(cfg.camera)               # Get video stream from camera
        self.process(cap)

        cap.release()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        filename='face_register.log',
                        filemode='w',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()