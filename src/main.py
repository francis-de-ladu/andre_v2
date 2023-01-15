import cv2 as cv

if __name__ == '__main__':
    net = cv.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    print(net)
