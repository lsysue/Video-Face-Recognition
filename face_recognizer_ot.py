# Author:   lsysue
# GitHub:   https://github.com/lsysue/Video-Face-Recognition
# Mail:     lsyxll268@163.com

# 利用 OT 人脸追踪, 进行人脸实时识别 / Real-time face detection and recognition via OT for multi faces
# 检测 -> 识别人脸, 新人脸出现 -> 不需要识别, 而是利用质心追踪来判断识别结果 / Do detection -> recognize face, new face -> not do re-recognition
# 人脸进行再识别需要花费大量时间, 这里用 OT 做跟踪 / Do re-recognition for multi faces will cost much time, OT will be used to instead it

import dlib
import numpy as np
import cv2
import os
import re
import csv
import shutil
import pandas as pd
import time
import logging
from config import cfg

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

infinity = 999999999

class Face_Recognizer:
    def __init__(self):
        # face, frames path
        self.path_detect_frames = cfg.dtc_frame_dir
        # For FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # cnt for frame
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.face_features_known_list = []
        # 存储录入人脸名字 / Save the name of faces in the database
        self.face_name_known_list = []
        # 用来存放所有未知人脸特征的数组 / Save the features of faces in the database
        self.face_features_unknown_list = []
        # 存储未知人脸名字 / Save the name of faces in the database
        self.face_name_unknown_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数 / Reclassify after 'reclassify_interval' frames
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 5

    # 从 "known_features.csv" 读取录入人脸特征 / Get known faces from "known_features.csv"
    def get_face_database(self):
        if os.path.exists("data/known_features.csv"):
            path_features_known_csv = "data/known_features.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0.0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
                self.face_name_known_list.append("Person_" + str(i + 1))
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'known_features.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def pre_work_folders(self):
        # 删除之前存的视频帧和人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        if os.path.isdir(self.path_detect_frames):
            folders_rd = os.listdir(self.path_detect_frames)
            for i in range(len(folders_rd)):
                os.remove(self.path_detect_frames+folders_rd[i])
        else:
            os.mkdir(self.path_detect_frames)
        if os.path.exists(cfg.data_dir + "unknown_features.csv"):
            os.remove(cfg.data_dir + "unknown_features.csv")

    # 获取处理之后 stream 的帧数 / Get the fps of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 使用质心追踪来识别人脸 / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算 / For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 处理获取的视频流, 进行人脸识别 / Face detection and recognition wit OT from input video stream
    def process(self, instream, outstream):
        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        self.pre_work_folders()
        if not self.get_face_database():
            logging.error("请录入人脸信息！")
 
        while instream.isOpened() and self.frame_cnt < cfg.detect_frame_num:
            self.frame_cnt += 1
            logging.debug("Frame " + str(self.frame_cnt) + " starts")
            flag, img_rd = instream.read()

            # 2. 检测人脸 / Detect faces for frame X
            faces = detector(img_rd, 0)

            # 3. 更新人脸计数器 / Update cnt for faces in frames
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)

            if not self.current_frame_face_cnt:
                logging.debug("当前帧没有检测到人脸 / No faces detected in this frame")

            # 4. 更新上一帧中的人脸列表 / Update the face name list in last frame
            self.last_frame_face_name_list = self.current_frame_face_name_list[:]

            # 5. 更新上一帧和当前帧的质心列表 / update frame centroid list
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []
            self.current_frame_face_position_list = []
            self.current_frame_face_name_list = ['unknown' for _ in range(self.current_frame_face_cnt)]
            self.current_frame_face_X_e_distance_list = []

            for i in range(self.current_frame_face_cnt):
                self.current_frame_face_position_list.append(
                            [faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom()])
                self.current_frame_face_centroid_list.append(
                            [int(faces[i].left() + faces[i].right()) / 2, 
                             int(faces[i].top() + faces[i].bottom()) / 2])

            # 6.1 如果当前帧和上一帧人脸数没有变化 / if cnt not changes
            if self.current_frame_face_cnt == self.last_frame_face_cnt:
                logging.debug("scene 1: 当前帧和上一帧相比没有发生人脸数变化 / No face cnt changes in this frame!!!")

                # 如果当前帧中有人脸, 使用质心追踪 / if there're faces in current frame, use centroid-tracker to track
                if self.last_frame_face_cnt != 0:
                    self.centroid_tracker()

                # if  "unknown" in self.last_frame_face_name_list:
                if list(filter(lambda x: re.match('unknown.*', x) != None, self.last_frame_face_name_list)):
                    logging.debug("  有未知人脸, 进行 reclassify_interval_cnt 计数: %d", self.reclassify_interval_cnt)
                    # known_indexes = [i for i, name in enumerate(self.current_frame_face_name_list) if name != 'unknown']
                    # unknown_indexes = [i for i, name in enumerate(self.current_frame_face_name_list) if name == 'unknown']
                    known_indexes = [i for i, name in enumerate(self.current_frame_face_name_list) if not re.match('unknown.*', name)]
                    unknown_indexes = [i for i, name in enumerate(self.current_frame_face_name_list) if re.match('unknown.*', name)]
                    logging.debug("known indexes | {}".format(known_indexes))
                    logging.debug("unknown indexes | {}".format(unknown_indexes))

                    for k in known_indexes:
                        img_rd = cv2.rectangle(img_rd,
                                                tuple([faces[k].left(), faces[k].top()]),
                                                tuple([faces[k].right(), faces[k].bottom()]),
                                                (0, 255, 0), 2)

                    if self.reclassify_interval_cnt != self.reclassify_interval:
                        self.reclassify_interval_cnt += 1
                        for k in unknown_indexes:
                            img_rd = cv2.rectangle(img_rd,
                                                tuple([faces[k].left(), faces[k].top()]),
                                                tuple([faces[k].right(), faces[k].bottom()]),
                                                (0, 0, 255), 2)

                    else:
                        logging.debug("  未知人脸出现 %d 帧，对未知人脸进行识别", self.reclassify_interval_cnt)
                        for k in unknown_indexes:
                            for i in range(len(self.face_features_unknown_list)):
                                if str(self.face_features_unknown_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_unknown_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(infinity)
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))
                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = 'New_' + str(similar_person_num + 1)
                            self.face_name_unknown_list.append('New_' + str(similar_person_num + 1))
                            with open('data/unknown_features.csv', 'a') as unknown_feature_file:
                                    writer = csv.writer(unknown_feature_file)
                                    writer.writerow(self.current_frame_face_feature_list[k])
                            img_rd = cv2.rectangle(img_rd,
                                                    tuple([faces[k].left(), faces[k].top()]),
                                                    tuple([faces[k].right(), faces[k].bottom()]),
                                                    (0, 255, 255), 2)
                
                elif self.current_frame_face_cnt != 0:
                    logging.debug("  检测到已知人脸")
                    for i in range(self.current_frame_face_cnt):
                        if "Person" in self.current_frame_face_name_list[i].split('_'):
                            img_rd = cv2.rectangle(img_rd,
                                                    tuple([faces[i].left(), faces[i].top()]),
                                                    tuple([faces[i].right(), faces[i].bottom()]),
                                                    (0, 255, 0), 2)
                        else:
                            img_rd = cv2.rectangle(img_rd,
                                                    tuple([faces[i].left(), faces[i].top()]),
                                                    tuple([faces[i].right(), faces[i].bottom()]),
                                                    (0, 255, 255), 2)
                outstream.write(img_rd)
                cv2.imwrite(self.path_detect_frames + "frame_{}.png".format(self.frame_cnt), img_rd)
                logging.info("Frame " + str(self.frame_cnt) + " face ids: {}".format(self.current_frame_face_name_list))

            # 6.2 如果当前帧和上一帧人脸数发生变化 / If cnt of faces changes, 0->1 or 1->0 or ...
            else:
                logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list = []
                self.reclassify_interval_cnt = 0

                # 6.2.1 人脸数减少 / Face cnt decreases: 1->0, 2->1, ...
                if self.current_frame_face_cnt  < self.last_frame_face_cnt:
                    logging.debug("  scene 2.1 人脸减少, 获取缺失人脸ID / Lost faces in this frame")
                    # clear list of names and features
                    self.centroid_tracker()
                    lost_frame_face_name_list = list(set(self.last_frame_face_name_list).difference(set(self.current_frame_face_name_list)))
                    logging.info("上一帧人脸ID列表 / last frame face ids: {}".format(self.last_frame_face_name_list))
                    logging.info("缺失人脸ID列表 / Lost face ids: {}".format(lost_frame_face_name_list))
                    for unknown_name in list(filter(lambda x: re.match('unknown.*', x) != None, lost_frame_face_name_list)):
                        unknown_id = int(list(unknown_name)[-1])
                        logging.debug(unknown_id)
                        self.face_features_unknown_list.pop(unknown_id - 1)
                    # if "unknown" in lost_frame_face_name_list:
                    #     self.face_features_unknown_list.pop(-1)
                    #     logging.debug("face features unknown list len: %d", len(self.face_features_unknown_list))
                    for i in range(self.current_frame_face_cnt):
                        img_rd = cv2.rectangle(img_rd,
                                                    tuple([faces[i].left(), faces[i].top()]),
                                                    tuple([faces[i].right(), faces[i].bottom()]),
                                                    (0, 255, 0), 2)
                # 6.2.2 人脸数增加 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                else:
                    logging.debug("  scene 2.2 出现新人脸, 进行人脸识别 / Get new faces in this frame and do face recognition")
                    for i in range(self.current_frame_face_cnt):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape))

                    # 6.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    for k in range(self.current_frame_face_cnt):
                        logging.debug("  For face %d in current frame:", k + 1)

                        self.current_frame_face_X_e_distance_list = []

                        # 6.2.2.2 对于某张人脸, 遍历所有存储的人脸特征
                        # For every faces detected, compare the faces in the database
                        for i in range(len(self.face_name_known_list)):
                            # 如果 q 数据不为空
                            
                            if str(self.face_features_known_list[i][0]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                    self.current_frame_face_feature_list[k],
                                    self.face_features_known_list[i])
                                logging.debug("      with known person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_frame_face_X_e_distance_list.append(infinity)

                        # 6.2.2.3 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                        similar_person_num = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list))

                        if min(self.current_frame_face_X_e_distance_list) < cfg.similarity_threshold:
                            self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                            logging.debug("  Face recognition result: %s",
                                            self.face_name_known_list[similar_person_num])
                            img_rd = cv2.rectangle(img_rd,
                                                tuple([faces[k].left(), faces[k].top()]),
                                                tuple([faces[k].right(), faces[k].bottom()]),
                                                (0, 255, 0), 2)
                            continue
                        
                        # 6.2.2.2 对于某张人脸, 遍历所有存储的未知人脸特征
                        # For every faces detected, compare the faces in the database
                        logging.debug("face name unknown list len: %d", len(self.face_name_unknown_list))
                        logging.debug("face features unknown list len: %d", len(self.face_features_unknown_list))
                        for i in range(len(self.face_name_unknown_list)):
                            # 如果 q 数据不为空
                            
                            if str(self.face_features_unknown_list[i][0]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                    self.current_frame_face_feature_list[k],
                                    self.face_features_unknown_list[i])
                                logging.debug("      with unknown person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_frame_face_X_e_distance_list.append(infinity)

                        # 6.2.2.3 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                        similar_person_num = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list))

                        logging.debug("similar person num | %d", similar_person_num + 1)
                        logging.debug("current frame face_X_e_distance_list len | %d", len(self.current_frame_face_X_e_distance_list))
                        logging.debug("similar unknown person id | %d", similar_person_num - len(self.face_name_known_list))
                        if min(self.current_frame_face_X_e_distance_list) < cfg.similarity_threshold and similar_person_num >= len(self.face_name_known_list):
                            self.current_frame_face_name_list[k] = self.face_name_unknown_list[similar_person_num - len(self.face_name_known_list)]
                            logging.debug("  Face recognition result: %s",
                                            self.face_name_unknown_list[similar_person_num - len(self.face_name_known_list)])
                            img_rd = cv2.rectangle(img_rd,
                                                tuple([faces[k].left(), faces[k].top()]),
                                                tuple([faces[k].right(), faces[k].bottom()]),
                                                (0, 255, 255), 2)
                            continue

                        else:
                            logging.debug("  Face recognition result: Unknown person")
                            self.current_frame_face_name_list[k] = 'unknown_' + str(len(self.face_features_unknown_list) + 1)
                            self.face_features_unknown_list.append(self.current_frame_face_feature_list[k])
                            self.reclassify_interval_cnt += 1
                            img_rd = cv2.rectangle(img_rd,
                                                tuple([faces[k].left(), faces[k].top()]),
                                                tuple([faces[k].right(), faces[k].bottom()]),
                                                (0, 0, 255), 2)

                outstream.write(img_rd)
                cv2.imwrite(self.path_detect_frames + "frame_{}.png".format(self.frame_cnt), img_rd)
                logging.info("Frame " + str(self.frame_cnt) + " face ids: {}".format(self.current_frame_face_name_list))

            # self.update_fps()

            logging.debug("Frame ends\n")

    def run(self):
        # cap = cv2.VideoCapture("data/input/video.avi")  # Get video stream from video file
        # cap = cv2.VideoCapture(0)              # Get video stream from camera
        cap = cv2.VideoCapture(cfg.camera)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video = cv2.VideoWriter("data/output/video.avi", cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, size)
        self.process(cap, video)

        cap.release()
        video.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        filename='face_recog.log',
                        filemode='w',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        ) # Set log level to 'logging.DEBUG' to print debug info of every frame
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
