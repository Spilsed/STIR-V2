import torch
import torch.nn as nn
import clip
import torchvision.transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
import numpy as np

voc_2012_classes = ['background', 'Aeroplane', "Bicycle", 'Bird', "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", 'Cow',
                    "Dining table", "Dog", "Horse", "Motorbike", 'Person', "Potted plant", 'Sheep', "Sofa", "Train",
                    "TV/monitor"]


class SVM(nn.Module):
    def __init__(self, classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(x)


def train_svm(svm, train, val, optimizer):
    running_loss = 0.0
    last_loss = 0.0

    for i, (image, data) in enumerate(train):
        ss_results = selective_search(np.array(image.convert('RGB'))[:, :, ::-1])
        for bbox in data["annotation"]["object"]:
            bbox = bbox["bndbox"]
            cv2_image = np.array(image.convert('RGB'))[:, :, ::-1]
            cv2_image = cv2.UMat(cv2_image)
            cv2.rectangle(cv2_image, (int(bbox["xmin"]), int(bbox["ymax"])), (int(bbox["xmax"]), int(bbox["ymin"])),
                          [200, 20, 250], 2)
            bbox = {"x1": int(bbox["xmin"]), "y1": int(bbox["ymax"]), "x2": int(bbox["xmax"]), "y2": int(bbox["ymin"])}
            for ss_bbox in ss_results:
                # ss_box = [x, y, w, h]
                x = ss_bbox[0]
                y = ss_bbox[1]
                w = ss_bbox[2]
                h = ss_bbox[3]
                iou = calculate_IoU(bbox, {"x1": ss_bbox[0], "y1": ss_bbox[1]+ss_bbox[3], "x2": ss_bbox[0]+ss_bbox[2], "y2": ss_bbox[1]})
                cv2.rectangle(cv2_image, (x, y), (x + w, y + h), [150, 0, 25], 2)
                if iou > 0.0:
                    print(iou)
                cv2.imshow("HJ", cv2_image)
                cv2.waitKey(0)


def calculate_IoU(bb1, bb2):
    # calculate IoU(Intersection over Union) of 2 boxes
    # **IoU = Area of Overlap / Area of Union
    # https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # calculate Overlapping area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area


def selective_search(image):
    # return region proposals of selective searh over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()


class RCNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 224, force_remake: bool = False):
        self.train_labels = []
        self.train_images = []
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if not self.dataset_exists() and not force_remake:
            pass

    def dataset_exists(self):
        pass

    def generate_dataset(self):
        obj_counter = 0
        bg_counter = 0
        print("-= GENERATING DATASET =-")

def main():
    device = torch.device("cuda:0")

    voc_train = VOCDetection(root="./VOC2012", year="2012", image_set="train", download=False)
    voc_val = VOCDetection(root="./VOC2012", year="2012", image_set="val", download=False)

    dataloader_train = DataLoader(voc_train, batch_size=32, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(voc_val, batch_size=32, shuffle=True, pin_memory=True)

    svm = SVM(21)
    print(svm.forward(torch.rand(512)))

    optimizer = torch.optim.AdamW(svm.parameters(), lr=0.001)

    train_svm(svm, voc_train, voc_val, optimizer)

if __name__ == "__main__":
    main()
