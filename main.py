#!/usr/bin/python
# -*- coding:UTF-8 -*-
##===============================================================================
# | file       :   main.py
# | author     :   Wang-wonk(CSDN); ZhengHao; Waveshare .Ltd
# | date       :   2020-04-27
# | function   :   执行车牌识别的主代码，使用一块1.54英寸的墨水屏显示指示，工作后
#                  自动将处理画面和识别结果显示HDMI屏和墨水屏上，运行前请检查本代
#                  码基于opencv 3.4.1和python 2.7。
##===============================================================================
#
#=============================== PART-1 模块导入 ================================
import sys
import os
import logging
from waveshare_epd import epd1in54_V2
import time

import cv2
from PIL import Image,ImageDraw,ImageFont

import commands
import predict

#=============================== PART-2 调用函数 ================================
#--------------------------------- get_cpu_temp ---------------------------------
# Param:None
# Return:float型温度值
# Note:返回CPU温度值
#--------------------------------------------------------------------------------
def get_cpu_temp():
    tempFile = open("/sys/class/thermal/thermal_zone0/temp")
    cpu_temp = tempFile.read()
    tempFile.close()
    return float(cpu_temp) / 1000  # 摄氏温度

#--------------------------------- get_gpu_temp ---------------------------------
# Param:None
# Return:float型温度值
# Note:返回GPU温度值
#--------------------------------------------------------------------------------
def get_gpu_temp():
    gpu_temp = commands.getoutput('/opt/vc/bin/vcgencmd measure_temp').replace('temp=', '').replace('\'C', '')
    return float(gpu_temp)


#=============================== PART-3 主函数 ==================================
def main():
    try:
        # 1-设备与环境初始化
        picdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib_E-ink/pic')  # 设置路径
        libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib_E-ink/lib')
        if os.path.exists(libdir):
            sys.path.append(libdir)
        logging.basicConfig(level=logging.DEBUG)

        # 摄像头
        camera = cv2.VideoCapture(0)     # 定义摄像头对象，参数0表示第一个摄像头，默认640x480
        # camera.set(cv2.CAP_PROP_FRAME_WIDTH,  960)  # 重设获取图像分辨率(predict.py设定了最大1000)
        # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # 图像越大处理速度越慢，树莓派不开超频时会卡
        if camera is None:               # 摄像头打开失败
            print(u'检测摄像头是否连接正常!')
            exit()
        fps = 24                         # 帧率

        # 水墨屏
        epd = epd1in54_V2.EPD()
        logging.info("E-ink Init & Clear")
        epd.init()
        epd.Clear(0xFF)
        image = Image.open(os.path.join(picdir, 'USST.bmp'))  # read bmp file
        epd.display(epd.getbuffer(image))                     # 欢迎界面
        time.sleep(2)

        # 绘制的图像界面
        image = Image.open(os.path.join(picdir, '1in54.bmp'))
        draw = ImageDraw.Draw(image)                            # 相当于清屏，得到当前图像对象句柄

        font20 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 20)  # 字体，高20个像素
        font24 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 24)  # 字体，高24个像素
        font30 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 30)

        draw.text((0, 89), u'   界面初始化   ', font=font30, fill=0)       # PIL库还提供很多绘制函数，十分方便
        draw.text((5, 165),u'CPU:     °C GPU:     °C', font=font20, fill=0)
        draw.rectangle((70, 4, 180, 30), fill=255)
        draw.text((70, 4), time.strftime('%m/%d/%Y'), font=font24, fill=0)
        epd.display(epd.getbuffer(image))

        # 向量机图像学习分类训练
        predictor = predict.CardPredictor()
        predictor.train_svm()
        logging.info(u'SVM学习训练完毕')

        refershsign = 0       # 刷新墨水屏的标记；0不刷新，1刷新
        lastrefreshtime = 0   # 记录墨水屏上次刷新的时间
        lastcarchar = None
        while True:
            start = time.time()
            res, cur_frame = camera.read()             # 读取视频流
            if res != True:
                break
            end = time.time()
            seconds = end - start
            if seconds < 1.0 / fps:                   # 按帧速处理，这里其实可以可以除掉了，因为这里视频输出帧率已经无法保证
                time.sleep(1.0 / fps - seconds)

            # 若检测按下ESC键，则退出程序
            key = cv2.waitKey(10) & 0xff
            if key == 27:
                break

            # 车牌识别
            carchar, roi, color = predictor.predict(cur_frame)              # 返回识别到的字符，定位的车牌图像，车牌颜色
            if carchar is not None:
                print(u'识别结果:{0:s}\n'.format(carchar.decode('utf-8')))  # Python2对中文字符不直接支持的问题
                if carchar != lastcarchar:   # 说明与上次识别的结果不一致，需要刷新下墨水屏
                    refershsign = 1
                    lastcarchar = carchar
            else:
                lastcarchar = None
                continue

            # 当有新的识别结果，且满足刷新频率时（2s/次），更新墨水屏显示
            if (time.time()-lastrefreshtime)>2 and refershsign == 1:
                draw.rectangle((10, 75, 190, 150), fill=255)
                draw.text((10, 75), u'      识别成功！   ', font=font24, fill=0)
                draw.text((35, 110), carchar.decode('utf-8'), font=font30, fill=0)

                draw.rectangle((70, 38, 180, 65), fill=255)
                draw.text((70, 38), time.strftime('   %H:%M:%S'), font=font24, fill=0)

                draw.rectangle((50, 165, 70, 198), fill=255)
                draw.text((50, 165), str(int(get_cpu_temp())), font=font20, fill=0)

                draw.rectangle((145, 165, 168, 198), fill=255)
                draw.text((145, 165), str(int(get_gpu_temp())), font=font20, fill=0)

                epd.displayPart(epd.getbuffer(image))      # 局部刷新显示
                lastrefreshtime = time.time()
                refershsign = 0

        camera.release()                # 释放摄像头
        cv2.destroyAllWindows()         # 关闭所有图像窗口
        logging.info("E-Ink Clear...")  # 墨水屏清屏
        epd.init()
        epd.Clear(0xFF)
        epd.sleep()

    # 异常处理
    except IOError as e:
        logging.info(e)

    except KeyboardInterrupt:
        logging.info("ctrl + c:")
        epd1in54_V2.epdconfig.module_exit()
        exit()

if __name__ == '__main__':
    main()

##================================== FILE END ===================================

