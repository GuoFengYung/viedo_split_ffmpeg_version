from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
from main import VideoSplit
import time
import os
video_suffix_list = ['webm', 'mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng'
                     , 'avi', 'MTS', 'M2Ts', 'TS', 'mov', 'qt', 'wmv', 'yuv', 'rm', 'rmvb', 'viv'
                     , 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm4v'
                     , 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 'f4b']

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super().__init__()
        self.log = ''
        self.mask_dir_path = False
        self.video_path = False

        self.ui = Ui_MainWindow()
        self.label_style = """
                        background-color: white;
                        border-radius: 5px;
                        border-radius:
                            30px;
                        border-style:
                            solid;
                        border-width:
                            2px;
                        border-color: white;
                        font:12pt;
                        color:black;
                    """
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.VideoPathButton.clicked.connect(self.open_video_file)
        self.ui.MaskDirPathButton.clicked.connect(self.open_mask_folder)
        self.ui.StartButton.clicked.connect(self.start_split_video)
        self.ui.ClearButton.clicked.connect(self.clear)
        self.ui.MaskDirPath.setStyleSheet(self.label_style)
        self.ui.VideoPath.setStyleSheet(self.label_style)
        self.ui.LogLabel.setStyleSheet(self.label_style)

    def open_video_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        self.ui.VideoPath.setText(filename)
        self.video_path = filename

    def open_mask_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        self.ui.MaskDirPath.setText(folder_path)
        self.mask_dir_path = folder_path


    def start_split_video(self):
        print(self.mask_dir_path and self.video_path)
        if not (self.mask_dir_path and self.video_path):
            self.log += '沒有正確選擇檔案路徑 \n'
            self.ui.LogLabel.setText(self.log)
            return

        img_ext_list = ['bmp', 'jpeg', 'png', 'tiff', 'ppm', 'tif']
        not_mask_imgs_path_list = [path for path in os.listdir(self.mask_dir_path) if path.split('.')[-1].lower() not in img_ext_list]
        mask_imgs_path_list = [path for path in os.listdir(self.mask_dir_path) if path.split('.')[-1].lower() in img_ext_list]
        for not_mask_imgs_path in not_mask_imgs_path_list:
            self.log += '{} 背景圖檔格式不支援... \n'.format(not_mask_imgs_path)
        self.ui.LogLabel.setText(self.log)

        # try:
        video_suffix = self.video_path.split('.')[-1]
        if video_suffix in video_suffix_list:
            if os.listdir(self.mask_dir_path) and len(mask_imgs_path_list) != 0:

                self.log += '處理中... \n'
                self.ui.LogLabel.setText(self.log)
                folder_name = VideoSplit(self.video_path, self.mask_dir_path).start_recording()
                self.log += '影片分割結束，存取在./' + folder_name + '\n'
            else:
                self.log += '背景圖資料夾為空 \n'
        else:
            self.log += '影像檔檔案格式不支援 \n'

        # except AttributeError:
        #     self.log += '沒有正確選擇檔案路徑 \n'
        self.ui.LogLabel.setText(self.log)

    def clear(self):
        self.ui.VideoPath.setText('')
        self.ui.MaskDirPath.setText('')
        self.ui.LogLabel.setText('')
        self.log = ''
        self.video_path = False
        self.mask_dir_path = False

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())