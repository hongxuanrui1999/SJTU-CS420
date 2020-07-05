from PIL import Image
import numpy as np
import os

#用于计算预测准确度，衡量模型的效果
def compute_acc(path_img, path_label):
    img = Image.open(path_img)
    #img.show()
    label = Image.open(path_label)
    img = np.array(img)
    #print(img[250])
    label = np.array(label)
    #print(label[250])
    print(img[0][1])
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if(img[i][j].all() == label[i][j].all()):
                if(label[i][j].all() == 0):
                    TN = TN + 1
                else:
                    TP = TP + 1
            else:
                if(label[i][j].all() == 0):
                    FP = FP + 1
                else:
                    FN = FN + 1
    return TN, TP, FP, FN

acc = 0
all_TN = 0
all_TP = 0
all_FP = 0
all_FN = 0


for name in range(len(os.listdir("img"))):
    img_file = os.path.join("img/%s" % (str(name)+".jpg"))
    label_file = os.path.join("label/%s" % (str(name)+".png"))
    TN, TP, FP, FN = compute_acc(img_file, label_file)
    all_TN = all_TN + TN
    all_TP = all_TP + TP
    all_FP = all_FP + FP
    all_FN = all_FN + FN
acc = (all_TP + all_TN)/ (all_TP + all_FN + all_FP + all_TN)

print(all_TP)
print(all_FP)
print(all_FN)
print(all_TN)

print(acc)










