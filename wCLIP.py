import os.path
import pickle
import random as r
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
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


def train_svm(svm, train, val, optimizer, loss_fn, epochs: int = 1):
    # Initialize loss variables
    running_loss = 0.0
    last_loss = 0.0

    # Load clip model
    model, preprocess = clip.load("ViT-B/16", device="cuda")

    toPIL = transforms.ToPILImage()

    # Shuffle the datasets
    train.shuffle()
    val.shuffle()

    for epoch in tqdm(range(epochs)):
        images = []
        labels = []

        vimages = []
        vlabels = []

        for i, data in enumerate(tqdm(train)):
            images.append(data["image"])
            labels.append(data["label"])

            vimages.append(val[i]["image"])
            vlabels.append(val[i]["label"])

            # TODO: This cannot be the best way
            if (i + 1) % 32 != 0:
                continue

            optimizer.zero_grad()

            # Get the outputs
            correct_outputs = []
            outputs = []
            for x, image in enumerate(images):
                feature = model.encode_image(preprocess(toPIL(image)).unsqueeze(0).cuda()).to(torch.float32)
                correct_outputs.append(
                    F.one_hot(torch.tensor(labels[x]), num_classes=len(voc_2012_classes)).unsqueeze(0))
                outputs.append(svm.forward(feature))
            images = []

            # Calculate loss with loss_fn
            loss = loss_fn(torch.cat(outputs).to(torch.float32).squeeze(0).cuda(),
                           torch.cat(correct_outputs).to(torch.float32).cuda())
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            # Every 10 batches print loss and preform validation
            if (i + 1) % (32 * 10) == 0:
                # Preform validation
                with torch.no_grad():
                    correct_outputs = []
                    outputs = []

                    for x, image in enumerate(vimages):
                        feature = model.encode_image(preprocess(toPIL(image)).unsqueeze(0).cuda()).to(torch.float32)
                        correct_outputs.append(
                            F.one_hot(torch.tensor(vlabels[x]), num_classes=len(voc_2012_classes)).unsqueeze(0))
                        outputs.append(svm.forward(feature))
                    images = []

                    vloss = loss_fn(torch.cat(outputs).to(torch.float32).squeeze(0).cuda(),
                                    torch.cat(correct_outputs).to(torch.float32).cuda())

                last_loss = running_loss / (32 * 10)  # loss per batch
                print('Batch {} loss: {} vloss: {}'.format(i + 1, last_loss, vloss))
                running_loss = 0.

    return last_loss


def selective_search(image):
    # return region proposals of selective search over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()


def transform_to_dataset(image, annotations):
    pass


class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataloader: DataLoader, data_type: str = "train", size: int = 224, force_remake: bool = False,
                 iou_threshold: float = 0.5,
                 image_ratio: tuple[int, int] = (32, 96), data_path: str = "./cache", section_size: int = 50):
        self.dataloader = dataloader
        self.data_path = data_path + "/" if data_path[-1] != "/" else data_path
        self.section_size = section_size
        self.data_type = data_type
        self.train_labels = []
        self.train_images = []

        self.total_obj_counter = 0
        self.total_bg_counter = 0

        self.iou_threshold = iou_threshold

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        # If the dataset doesn't exist, or it is being force remade generate a new dataset
        if not self.dataset_exists() or force_remake:
            self.generate_dataset(dataloader, image_ratio)
        else:
            print("-= LOADING DATASET FROM", self.data_path, "=-")
            with open(self.data_path + 'train_images.pkl', 'rb') as f:
                self.train_images = pickle.load(f)
            with open(self.data_path + 'train_labels.pkl', 'rb') as f:
                self.train_labels = pickle.load(f)

    def dataset_exists(self):
        # If both don't exist the other is kinda worthless
        if os.path.exists(self.data_path + self.data_type + "_images.pkl") and os.path.exists(
                self.data_path + self.data_type + "_labels.pkl"):
            return True
        else:
            return False

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {"image": self.transform(Image.fromarray(cv2.cvtColor(self.train_images[idx], cv2.COLOR_BGR2RGB))),
                "label": self.train_labels[idx][0], "bbox": self.train_labels[idx][1]}

    def shuffle(self):
        r.shuffle(self.train_images)
        r.shuffle(self.train_labels)

    def generate_dataset(self, dataloader: DataLoader, image_ratio: tuple[int, int]):

        print("-= GENERATING DATASET =-")
        for index, (image, data) in enumerate(tqdm(dataloader.dataset)):
            # Get selective search for whole image
            ss_results = selective_search(np.array(image.convert('RGB'))[:, :, ::-1])

            # Process each object
            for obj in data["annotation"]["object"]:
                self.process_object(obj, ss_results, image, image_ratio)

            del ss_results

            print("IMAGES: ", sys.getsizeof(self.train_images) / 1e9, "GB")
            print("LABELS: ", sys.getsizeof(self.train_labels) / 1e9, "GB")

            if index % self.section_size == 0 and index != 0:
                self.create_section(index % self.section_size)

        self.shuffle()

        print("-= DATASET FINALIZED WITH ", len(self.train_images), "DATA POINTS =-", "\n-= SAVING DATA TO", self.data_path, "=-")

    def process_object(self, obj, ss_results, image, image_ratio):
        obj_counter = 0
        bg_counter = 0

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
            if iou >= self.iou_threshold:
                obj_counter += 1

                # Crop image to ss_box
                cropped = np.asarray(image)[ss_bbox[1]:ss_bbox[1] + ss_bbox[3], ss_bbox[0]:ss_bbox[0] + ss_bbox[2], ::-1]

                # Add cropped to images and label to labels
                self.train_images.append(cropped)
                self.train_labels.append([label_to_index(obj["name"]), ss_bbox])

            elif bg_counter < image_ratio[1]:
                bg_counter += 1

                # Crop image to ss_box (", ::-1]" to change from BGR to RGB)
                cropped = np.asarray(image)[ss_bbox[1]:ss_bbox[1] + ss_bbox[3], ss_bbox[0]:ss_bbox[0] + ss_bbox[2], ::-1]

                self.train_images.append(cropped)
                self.train_labels.append([0, ss_bbox])

            # Maintain ratio between the types
            if obj_counter >= image_ratio[0] and bg_counter == image_ratio[1]:
                obj_counter -= image_ratio[0]
                bg_counter = 0
        return

    # Dump all that data as a pickle so that it can just be reloaded later
    def create_section(self, section_id: int):
        with open(self.data_path + self.data_type + "_labels_" + str(section_id) +".pkl", "wb") as f:
            pickle.dump(self.train_labels, f)
        with open(self.data_path + self.data_type + "_images_" + str(section_id) +".pkl", "wb") as f:
            pickle.dump(self.train_images, f)

        self.train_labels = []
        self.train_images = []


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
    val_dataset = RCNNDataset(dataloader_val, "val")
    with open("./" + "val_FullDataset.pkl", "wb") as f:
        pickle.dump(val_dataset, f)

    svm = SVM(21).cuda(device=0)

    print(len(train_dataset))

    optimizer = torch.optim.AdamW(svm.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss().cuda(device=0)

    train_svm(svm, train_dataset, val_dataset, optimizer, loss_fn, 10)

    torch.save(svm.state_dict(), "./Model")


if __name__ == "__main__":
    main()
