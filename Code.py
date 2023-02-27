import cv2
import numpy as np

confThreshold = 0.5
nmsThreshold = 0.2

model_cfg = "yolov3-320.cfg"
model_weights = "yolov3.weights"

cap = cv2.VideoCapture(0)

object_names = "coco.names"
with open(object_names,"r") as f:
    classNames = f.read().rstrip("\n").split("\n")
print(classNames)

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

bbox = []
classIds = []
confs = []




while True:
    success,img = cap.read()
    blob = cv2.dnn.blobFromImage(img,1/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob)
    layers_name = net.getLayerNames()
    output = net.getUnconnectedOutLayers()
    output_layer_names = [layers_name[i[0]-1] for i in output]
    outputs = net.forward(output_layer_names)
    #print(len(outputs))


    hT, wT, cT = img.shape
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        # print(indices)
        box = bbox[i[0]]
        x, y, w, h = box[0], box[1], box[2], box[3]
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i[0]]].upper()} {int(confs[i[0]] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    cv2.imshow("webcam", img)
    cv2.waitKey(1)
