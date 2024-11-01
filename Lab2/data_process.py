import numpy as np
import pandas as pd
import os
import shutil
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder

class DataProcess():
    def __init__(self) -> None:
        image_data = []
        annotations = []
        
        self.annotations_dir = 'data/annotations/'
        self.images_dir = 'data/images/'
        self.output_dir = 'data/face-mask-detection-yolo'
        self._create_yolo_dirs()

        for xml_file in os.listdir(self.annotations_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(self.annotations_dir, xml_file)
                self._parse_xml(xml_path, annotations)

                image_file = xml_file.replace('.xml', '.png')
                image_path = os.path.join(self.images_dir, image_file)

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (128,128))

                    image_flatten = image.flatten()

                    image_data.append(image_flatten)

        self.X = np.array(image_data)
        self.y = annotations

    def get_data(self) -> tuple:
        return (self.X, self.y)

    def _parse_xml(self, xml_file, annotations: list) -> list:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        file_name = root.find('filename').text
        objects = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append((class_name, xmin, ymin, xmax, ymax))
        annotations.append((file_name, objects))
        return annotations
    
    def _create_yolo_dirs(self) -> None:
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'val'), exist_ok=True)

    def _convert_to_yolo_format(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh

        return (x,y,w,h)

    def save_yolo_files(self, annotations, image_dir, label_dir) -> None:
        labels = [obj[0] for ann in self.y for obj in ann[1]]
        label_enc = LabelEncoder()
        label_enc.fit(labels)
        for file_name, objects in annotations:
            img_path = os.path.join(self.images_dir, file_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            shutil.copy(img_path, image_dir)
            label_path = os.path.join(label_dir, file_name.replace('.png', '.txt'))

            with open(label_path, 'w') as f:
                for obj in objects:
                    class_name, xmin, ymin, xmax, ymax = obj
                    class_id = label_enc.transform([class_name])[0]
                    bbx = self._convert_to_yolo_format((w,h), (xmin, ymin, xmax, ymax))
                    f.write(f"{class_id} {' '.join(map(str, bbx))}\n")

        data_yaml = f"""
        train: {os.path.abspath(os.path.join(self.output_dir, 'images', 'train'))}
        val: {os.path.abspath(os.path.join(self.output_dir, 'images', 'val'))}

        nc: {len(label_enc.classes_)}
        names: {label_enc.classes_.tolist()}
        """
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write(data_yaml)