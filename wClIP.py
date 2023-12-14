import os.path
import pickle

import torch
import torch.nn as nn
import clip
import torchvision.transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from torchvision import ops
import numpy as np
from tqdm import tqdm
from PIL import Image

voc_2012_classes = ['background', 'aeroplane', "bicycle", 'bird', "boat", "bottle", "bus", "car", "cat", "chair", 'cow',
                    "diningtable", "dog", "horse", "motorbike", 'person', "pottedplant", 'sheep', "sofa", "train",
                    "tvmonitor"]


def label_to_index(label: str):
    # Remove any formatting
    label = label.lower().replace(" ", "")

    return voc_2012_classes.index(label)


def index_to_label(index: int):
    return voc_2012_classes[index]


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
        # Get selective search results
        ss_results = selective_search(np.array(image.convert('RGB'))[:, :, ::-1])




def selective_search(image):
    # return region proposals of selective search over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()


def transform_to_dataset(image, annotations):
    pass


class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataloader: DataLoader, data_type: str = "train", size: int = 224, force_remake: bool = False, iou_threshold: float = 0.5,
                 image_ratio: tuple[int, int] = (32, 96), data_path: str = "./"):
        self.dataloader = dataloader
        self.data_path = data_path
        self.data_type = data_type
        self.train_labels = []
        self.train_images = []

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        # If the dataset doesn't exist, or it is being force remade generate a new dataset
        if not self.dataset_exists() or force_remake:
            self.generate_dataset(dataloader, iou_threshold, image_ratio)
        else:
            print("-= LOADING DATASET FROM", self.data_path, "=-")
            with open(self.data_path + 'train_images.pkl', 'rb') as f:
                self.train_images = pickle.load(f)
                print(self.train_images[:10])
            with open(self.data_path + 'train_labels.pkl', 'rb') as f:
                self.train_labels = pickle.load(f)

    def dataset_exists(self):
        # If both don't exist the other is kinda worthless
        if os.path.exists(self.data_path + self.data_type + "_images.pkl") and os.path.exists(self.data_path + self.data_type + "train_labels.pkl"):
            return True
        else:
            return False

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"image": self.transform(Image.fromarray(cv2.cvtColor(self.train_images[idx], cv2.COLOR_BGR2RGB))), "label": self.train_labels[idx][0], "bbox": self.train_labels[idx][1]}

    def generate_dataset(self, dataloader: DataLoader, iou_threshold: float, image_ratio: tuple[int, int]):
        obj_counter = 0
        bg_counter = 0
        total_obj_counter = 0
        total_bg_counter = 0

        print("-= GENERATING DATASET =-")
        for index, (image, data) in enumerate(tqdm(dataloader.dataset)):
            # TODO: Remove 500 image limit after testing (I'm not waiting 3:30:00 every time I test obv)--
            if index >= 100:
                break

            # Get selective search for whole image
            ss_results = selective_search(np.array(image.convert('RGB'))[:, :, ::-1])

            # IDEA: Multi-threading? There are never more then ~24 objects. So each could run in its own thread.
            # For object in data get the bounding box
            for obj in data["annotation"]["object"]:
                bbox = obj["bndbox"]

                # Iterate through every predicted bbox
                for ss_bbox in ss_results:
                    gt_bbox_t = torch.tensor(
                        [[int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])]],
                        dtype=torch.float)
                    ss_bbox_t = torch.tensor(
                        [[ss_bbox[0], ss_bbox[1], ss_bbox[0] + ss_bbox[2], ss_bbox[1] + ss_bbox[3]]],
                        dtype=torch.float)

                    # Calculate the IoU
                    iou = ops.box_iou(gt_bbox_t, ss_bbox_t).numpy()[0][0]

                    # Mark ss_bbox as positive if the IoU is above threshold
                    if iou >= iou_threshold:
                        obj_counter += 1
                        total_obj_counter += 1

                        # Crop image to ss_box
                        cropped = np.asarray(image)[ss_bbox[1]:ss_bbox[1] + ss_bbox[3], ss_bbox[0]:ss_bbox[0] + ss_bbox[2], ::-1]

                        # Add cropped to images and label to labels
                        self.train_images.append(cropped)
                        self.train_labels.append([label_to_index(obj["name"]), ss_bbox])

                    elif bg_counter < image_ratio[1]:
                        bg_counter += 1
                        total_bg_counter += 1

                        # Crop image to ss_box (", ::-1]" to change from BGR to RGB)
                        cropped = np.asarray(image)[ss_bbox[1]:ss_bbox[1] + ss_bbox[3], ss_bbox[0]:ss_bbox[0] + ss_bbox[2], ::-1]

                        self.train_images.append(cropped)
                        self.train_labels.append([0, ss_bbox])

                    # Maintain ratio between the types
                    if obj_counter >= image_ratio[0] and bg_counter == image_ratio[1]:
                        obj_counter -= image_ratio[0]
                        bg_counter = 0

        print("-= DATASET FINALIZED WITH ", len(self.train_images), "DATA POINTS =-", "\n-= SAVING DATA TO",
              self.data_path, "=-")

        # Dump all that data as a pickle so that it can just be reloaded later
        with open(self.data_path + self.data_type + "_labels.pkl", "wb") as f:
            pickle.dump(self.train_labels, f)
        with open(self.data_path + self.data_type + "_images.pkl", "wb") as f:
            pickle.dump(self.train_images, f)

def main():
    device = torch.device("cuda:0")

    voc_train = VOCDetection(root="./VOC2012", year="2012", image_set="train", download=False)
    voc_val = VOCDetection(root="./VOC2012", year="2012", image_set="val", download=False)

    dataloader_train = DataLoader(voc_train, batch_size=32, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(voc_val, batch_size=32, shuffle=True, pin_memory=True)

    # Create and save train dataset
    train_dataset = RCNNDataset(dataloader_train, "train")
    with open("./" + "train_FullDataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    # Create and save val dataset
    val_dataset = RCNNDataset(dataloader_train, "val")
    with open("./" + "val_FullDataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    print(train_dataset[0])
    toPIL = transforms.ToPILImage()
    for i in range(100, 120):
        toPIL(train_dataset.__getitem__(i)["image"]).show()

    svm = SVM(21)
    print(svm.forward(torch.rand(512)))

    optimizer = torch.optim.AdamW(svm.parameters(), lr=1e-3)

    train_svm(svm, train_dataset, voc_val, optimizer)


if __name__ == "__main__":
    main()
