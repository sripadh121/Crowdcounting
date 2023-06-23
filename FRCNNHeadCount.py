from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import os

main = tkinter.Tk()
main.title("Crowd Counting Method Based on Convolutional Neural Network With Global Density Feature")
main.geometry("1200x1200")

#loading FASTER RCNN model to count human head from images and videos
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_path, threshold):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_class = []
  for i in list(pred[0]['labels'].numpy()):
      pred_class.append(i)
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  head_count = 0
  for i in range(len(pred_class)):
      if pred_class[i] == 1:
          head_count += 1
  return head_count
  

def countFromImages():
    global filename
    count = 0
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.insert(END,str(filename)+" loaded\n")
    pathlabel.config(text=str(filename)+" loaded")
    head_count = get_prediction(filename, 0.8)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img,"Total Head: "+str(head_count), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),thickness=2)
    cv2.imshow("output",img)
    cv2.waitKey(0)



def countFromVideo():
    global filename
    global frcnn
    filename = filedialog.askopenfilename(initialdir="testVideos")
    text.insert(END,str(filename)+" loaded\n")
    pathlabel.config(text=str(filename)+" loaded")
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        print(ret)
        if ret == True:
            cv2.imwrite("test.jpg",frame)
            head_count = get_prediction("test.jpg", 0.8)
            cv2.putText(frame,"Total Head: "+str(head_count), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),thickness=2)
            cv2.imshow("output",frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break    
        else:
            break
    video.release()
    cv2.destroyAllWindows()


                

font = ('times', 14, 'bold')
title = Label(main, text='Crowd Counting Method Based on Convolutional Neural Network With Global Density Feature')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
imageButton = Button(main, text="People Counting from Images", command=countFromImages)
imageButton.place(x=50,y=100)
imageButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=100)

videoButton = Button(main, text="People Counting from Video", command=countFromVideo)
videoButton.place(x=50,y=150)
videoButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
