from logging import root
import os
class Config():
    root_dir = os.getcwd() + '/'
    camera = "camera video path"
    data_dir = root_dir + 'data/'
    rgst_frame_dir = data_dir + 'rgst_frames/'
    rgst_face_dir = data_dir + 'rgst_faces/'
    dtc_frame_dir = data_dir + 'dtc_frames/'
    register_frame_num = 50
    detect_frame_num = 300
    similarity_threshold = 0.45

    # output video

cfg = Config()