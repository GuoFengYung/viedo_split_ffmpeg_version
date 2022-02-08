import cv2
import argparse
import os
from collections import deque
from skimage.metrics import structural_similarity as ssim
import numpy as np
import imutils
import subprocess
from datetime import date, datetime

threshold = 0.9
width_rate = 1
coordinates = [[(1254, 633), (1920, 633), (1920, 1080), (1254, 1080)],
              [(0, 836), (847, 836), (847, 1080), (0, 1080)],
              [(0, 1030), (1920, 1030), (1920, 1080), (0, 1080)]]


video_save_path = './video/'
# image_save_path = '/home/Mailbox/PycharmProjects/viedo/image/'

class VideoSplit():

    def __init__(self, path, mask_folder_path):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not

        self.video_path = path
        self.stream = cv2.VideoCapture(path)
        self.mask_images = []
        img_ext_list = ['bmp', 'jpeg', 'png', 'tiff', 'ppm', 'tif']
        self.mask_imgs_path_list = [os.path.join(mask_folder_path, path) for path in os.listdir(mask_folder_path) if path.split('.')[-1].lower() in img_ext_list]
        print(self.mask_imgs_path_list)

        self.start_flag, self.end_flag = False, False
        self.start_time, self.end_time = 0, 0
        self.interval = 1000
        self.previous_ssim = deque([])
        for i in range(3000 * 30 // self.interval):
            self.previous_ssim.append(1)
        self.threshold = 0.9
        self.count = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.stream.get(3))
        self.frame_height = int(self.stream.get(4))
        self.length = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fi = -1

        self.video_file_name = video_save_path
        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')

    def show_frame(self, frame):
        # Display frames in main program
        if self.status:
            f = imutils.resize(frame, width=450)
            cv2.imshow('', f)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def time_format(self, fi):
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        ft = fi / fps
        sec = int(ft)
        mnt = int(sec / 60)
        hr = int(mnt / 60)
        return "{:0>2d}".format(hr) + ":" + "{:0>2d}".format(mnt % 60) + ":" + "{:0>2d}".format(sec % 60)

    def time_interval(self, start_time, end_time):
        time_1 = datetime.strptime(start_time, "%H:%M:%S")
        time_2 = datetime.strptime(end_time, "%H:%M:%S")
        time_interval = time_2 - time_1
        return time_interval

    def cal_ssim(self, frame):

        ssim_value_per_mask = []

        for mask_image in self.mask_images:

            img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

            v1 = np.mean(img1)
            v2 = np.mean(img2)
            avg = (v1 + v2) / 2
            lvalue = int(avg)
            _, img1 = cv2.threshold(img1, lvalue, 255, cv2.THRESH_BINARY)
            _, img2 = cv2.threshold(img2, lvalue, 255, cv2.THRESH_BINARY)

            img1 = imutils.resize(img1, width=1920, height=1080)
            img2 = imutils.resize(img2, width=1920, height=1080)

            for coordinate in coordinates:

                (x1, y1) = coordinate[0]

                (x3, y3) = coordinate[2]

                img1_crop = img1[y1:y3, x1:x3]
                img2_crop = img2[y1:y3, x1:x3]
                img_ssim = ssim(img1_crop, img2_crop)

                if img_ssim > threshold:
                    print(self.fi, img_ssim, 'yes')
                    # meaning finish a segment point
                    if self.start_flag and self.previous_ssim_mean < threshold:
                        # close viedo writer
                        # segment count ++
                        # creat a new video writer
                        self.end_flag = True
                        # print('end frame:', fi)
                        #
                        # print('new start point:', fi)
                    # meaning start a new segment point
                    self.start_flag = True

                    return img_ssim

                else:

                    ssim_value_per_mask.append(img_ssim)

        return sum(ssim_value_per_mask) / len(ssim_value_per_mask)

    def start_recording(self):
        # Create another thread to show/save frames

        folder_name = self.video_path.split('.')[0].split(os.path.sep)[-1]
        today = date.today().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H-%M-%S")
        folder_name = folder_name + '_' + str(today) + '_' + str(now)
        str_path = ''.join(folder_name)
        last_dir = str_path.split('/')[-1]
        pat = r"C:\Users\kuo\桌面\viedo_split_ffmpeg_version\News1210-1200_2022-01-28_09:45:43"
        new_path = os.path.join(os.getcwd() + '\\' + last_dir)
        os.makedirs(folder_name)
        for path in self.mask_imgs_path_list:
            self.mask_images.append(cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1))

        time = dict()

        while True:

            self.fi += 1

            if self.fi % int(1 * self.interval) == 0:
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.fi)
                self.status, frame = self.stream.read()
                self.show_frame(frame)
            else:
                continue

            if not self.status:
                break

            img_ssim = self.cal_ssim(frame)
            self.previous_ssim.append(img_ssim)
            self.previous_ssim.popleft()
            self.previous_ssim_mean = sum(self.previous_ssim) / len(self.previous_ssim)

            if self.start_flag:

                if self.start_time == 0:
                    self.start_time = self.time_format(self.fi)

                if self.end_flag:
                    self.end_time = self.time_format(self.fi)
                    self.time_elapse = self.time_interval(self.start_time, self.end_time)
                    cmd = 'ffmpeg -ss {start_time} -i {video_path} -to {time_elapse} -c copy {folder_name}/clip_{count}.mp4'.format(
                        start_time=self.start_time, video_path=self.video_path, time_elapse=self.time_elapse, folder_name=folder_name, count=str(self.count)
                    )
                    print(cmd)
                    subprocess.run(cmd)
                    time[self.count] = dict(start=self.start_time, end=self.end_time)
                    self.start_time = self.end_time
                    self.count += 1
                    self.end_flag = False
                    self.previous_ssim = deque([])
                    for i in range(3000 * 30 // self.interval):
                        self.previous_ssim.append(1)

        least_star_time = time[self.count - 1]['end']
        video_end_time = self.time_format(int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)))
        time[self.count] = dict(start=least_star_time, end=video_end_time)
        self.time_elapse = self.time_interval(least_star_time, video_end_time)
        cmd = 'ffmpeg -ss {start_time} -i {video_path} -to {time_elapse} -c copy ./{folder_name}/clip_{count}.mp4'.format(
            start_time=self.start_time, video_path=self.video_path, time_elapse=self.time_elapse,
            folder_name=folder_name, count=str(self.count)
        ).split()
        subprocess.run(cmd)
        print(time)
        return folder_name

if __name__ == '__main__':


    video_path = '/home/aibox/PycharmProjects/viedo/video/clip_13.avi'
    # print(os.getcwd())
    # parser = argparse.ArgumentParser()
    #
    # # region ===== Common arguments =====
    # parser.add_argument('-r', '--reference_dir', type=str, required=True, help='path to reference directory')
    #
    # args = parser.parse_args()
    #
    #
    # # video_path = 'D:/test/News1210-1200.mp4'
    # s = VideoSplit(args.reference_dir, './20210122/', '')
    # time = VideoWriterWidget(video_path, queue_size=223).time_format(10000)
    # print(time)
    # count =1
    # start_time = '00:05:10'
    # end_time = '00:17:44'
    # cmd = 'ffmpeg -ss {start_time} -i {video_path} -to {end_time} -c copy ./video/clip_{count}.mp4'.format(
    #     start_time=start_time, video_path=args.reference_dir, end_time=end_time, count=str(1)
    # ).split()
    #
    # # print(cmd)
    # import subprocess
    # subprocess.run(cmd)
    # import subprocess
    # t = 'ffmpeg -ss 00:01:00 -i D:/test/News1207-1900.mp4 -to 00:20:00 -c copy output.avi'
    # t = t.split()
    # print(t)

    # subprocess.run(t)