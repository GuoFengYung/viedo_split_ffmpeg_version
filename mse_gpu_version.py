import numpy as np
import cv2
from progressbar import *
import argparse
from torch.nn.functional import mse_loss as mse
import pandas as pd
import os
import statistics
from collections import deque
import imutils
import torch
from torch.autograd import Variable
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse

width_rate = 1.0
coordinate = np.asarray([[0, 781, 1090, 1080], [1059, 592, 1920, 1080], [794, 899, 1920, 1080],
                       [0, 842, 947, 1080], [[0, 1035, 1920, 1080], [1820, 819, 1915, 987], [3, 829, 120, 986]]], dtype=object)

path = '/home/aibox/Downloads/20210122/'

def img_crop(image, coordinate):
    x1 = int(coordinate[0] * width_rate)
    y1 = int(coordinate[1] * width_rate)
    x2 = int(coordinate[2] * width_rate)
    y2 = int(coordinate[3] * width_rate)
    crop_image = image[y1:y2, x1:x2]
    return crop_image

def reversed_formatTime(ft: str):

    time = ft.split(':')

    return int(time[0])*60*60 + int(time[1])*60 + int(time[2])


def formatTime(ft):

    sec = int(ft/1000)
    mnt = int(sec/60)
    hr = int(mnt/60)
    return "{:0>2d}".format(hr)+":"+"{:0>2d}".format(mnt % 60)+":"+"{:0>2d}".format(sec % 60)

def compareVideo(dir_reference, dir_save, mask_images, threshold):

    image_save_path = dir_save + '/image/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    video_save_path = dir_save + '/video/'
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    fvs = FileVideoStream(path=dir_reference, queue_size=1280).start()
    time.sleep(1.0)

    input_movie1 = cv2.VideoCapture(dir_reference)
    width = int(input_movie1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_movie1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length1 = int(input_movie1.get(cv2.CAP_PROP_FRAME_COUNT))

    times = time.perf_counter()
    fi = -1
    l = []
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=length1).start()
    start_flag, end_flag = False, False
    previous_mse = deque([])
    for i in range(3000*30):
        previous_mse.append(0.1)
    count = 1
    while True:
        fi += 1
        # if fi % int(30) == 0:
        #     input_movie1.set(cv2.CAP_PROP_POS_FRAMES, fi)
            # input_movie2.set(cv2.CAP_PROP_POS_FRAMES, fi+int(r1))
        ret1, frame = input_movie1.read()
        frame1 = imutils.resize(frame, width=int(width_rate*width))

        if not ret1:
            break
            # ret2, frame2 = input_movie2.read()
        # else:
        #     continue

        for index, mask_image in enumerate(mask_images):

            img1 = frame1
            img2 = imutils.resize(mask_image, width=int(width_rate*width))

            try:

                len((coordinate[index][0]))

            except TypeError:

                img1_crop = img_crop(img1, coordinate[index])
                img1_crop = torch.from_numpy(np.rollaxis(img1_crop, 2)).float().unsqueeze(0) / 255.0
                img2_crop = img_crop(img2, coordinate[index])
                img2_crop = torch.from_numpy(np.rollaxis(img2_crop, 2)).float().unsqueeze(0) / 255.0

                if torch.cuda.is_available():
                    img1_crop = img1_crop.cuda()
                    img2_crop = img2_crop.cuda()

                img1_crop = Variable(img1_crop, requires_grad=False)
                img2_crop = Variable(img2_crop, requires_grad=False)

                img_mse = mse(img1_crop, img2_crop).item()
                # if img_mse > threshold:
                #
                #     cv2.rectangle(frame1, (coordinate[index][0], coordinate[index][1]),
                #                   (coordinate[index][2], coordinate[index][3]), (0, 0, 255), 8)
                #     cv2.imwrite(image_save_path + str(fi) + ".jpg", frame1)
                #     l.append(
                #         [fi, str(image_save_path + str(fi) + ".jpg"),
                #          str(img_mse)])

            else:

                mse_avg = []
                for j in coordinate[index]:
                    img1_crop = img_crop(img1, j)
                    img1_crop = torch.from_numpy(np.rollaxis(img1_crop, 2)).float().unsqueeze(0) / 255.0
                    img2_crop = img_crop(img2, j)
                    img2_crop = torch.from_numpy(np.rollaxis(img2_crop, 2)).float().unsqueeze(0) / 255.0

                    if torch.cuda.is_available():
                        img1_crop = img1_crop.cuda()
                        img2_crop = img2_crop.cuda()

                    img1_crop = Variable(img1_crop, requires_grad=False)
                    img2_crop = Variable(img2_crop, requires_grad=False)
                    mse1 = mse(img1_crop, img2_crop).item()
                    mse_avg.append(mse1)
                img_mse = statistics.mean(mse_avg)

                # if img_mse > threshold:
                #     for j in coordinate[index]:
                #         cv2.rectangle(frame1, (j[0], j[1]),
                #                       (j[2], j[3]), (0, 0, 255), 8)
                #
                #     cv2.imwrite(image_save_path + str(fi) + ".jpg", frame1)
                #     l.append(
                #         [fi, str(image_save_path + str(fi) + ".jpg"),
                #          str(img_mse)])

            if img_mse < threshold:

                # meaning finish a segment point
                if start_flag and statistics.mean(previous_mse) < threshold:
                    # close viedo writer
                    # segment count ++
                    # creat a new video writer
                    end_flag = True
                    # print('end frame:', fi)
                    #
                    # print('new start point:', fi)
                # meaning start a new segment point
                start_flag = True
            previous_mse.append(img_mse)
            previous_mse.popleft()
            pbar.update(fi)
            dosomework()
        if start_flag:

            try:

                writer.isOpened()

            except NameError:

                writer = cv2.VideoWriter(video_save_path + 'clip_' + str(count)+'.avi',
                                         cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                         30,  # fps
                                         (width, height))  # resolution
                print('-----------------create clip: ', str(count))

            writer.write(frame)

            if end_flag:
                print('-----------------clip: ', str(count) + ' done!!!')
                print()
                count += 1
                writer.release()
                del writer
                end_flag = False
                previous_mse = deque([])
                for i in range(3000*30):
                    previous_mse.append(0.1)

        cv2.waitKey(1)

    writer.release()
    input_movie1.release()
    pbar.finish()

    cv2.destroyAllWindows()

    df = pd.DataFrame(l, columns=['Time', 'image', 'mse Value'])

    timee = time.perf_counter()  # A few seconds later
    print("Check", fi, "frames.")
    print("Total Time:", (timee-times))

    pd.set_option('display.max_colwidth', None)
    df.to_html(dir_save + '/result_html.html', escape=False, formatters=dict(image=path_to_image_html))
    pdfkit.from_file(dir_save + '/result_html.html', dir_save + '/result.pdf')

def path_to_image_html(path):

    return '<img src="'+ path + '" width="640" >'

def dosomework():

    time.sleep(0.00001)

def mergeFrames(fm1,fm2):

    leftimg = cv2.resize(fm1, (640, 480), interpolation=cv2.INTER_CUBIC)
    rightimg = cv2.resize(fm2, (640, 480), interpolation=cv2.INTER_CUBIC)
    mergeimg = np.concatenate((leftimg, rightimg), axis=1)
    return mergeimg

def formatTime(ft):

    sec = int(ft / 1000)
    mnt = int(sec / 60)
    hr = int(mnt / 60)
    return "{:0>2d}".format(hr) + ":" + "{:0>2d}".format(mnt % 60) + ":" + "{:0>2d}".format(sec % 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # region ===== Common arguments =====
    parser.add_argument('-r', '--reference_dir', type=str, required=True, help='path to reference directory')
    parser.add_argument('-s', '--save_dir', type=str, required=True, help='path save directory')
    parser.add_argument('-t', '--threshold', type=float, help='mse threshold', default=0.8)

    args = parser.parse_args()
    # region ===== Common arguments =====

    img_path = [path+'B2 空間感.tif', path+'B1左(夜).tif', path+'S1左(夜).tif', path+'S2右(夜).png', path+'T三框預告.tif']
    mask_images = []

    for path in img_path:
        mask_images.append(cv2.imread(path))

    compareVideo(args.reference_dir, args.save_dir, mask_images, args.threshold)
