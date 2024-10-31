import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2

#https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

class DataProcess():
    def __init__(self):
        image_data = []
        labels = []
        annotations_dir = 'data/annotations/'
        images_dir = 'data/images/'

        for xml_file in os.listdir(annotations_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotations_dir, xml_file)
                label = self._parse_xml(xml_path)

                image_file = xml_file.replace('.xml', '.png')
                image_path = os.path.join(images_dir, image_file)

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (128,128))

                    image_flatten = image.flatten()

                    image_data.append(image_flatten)
                    labels.append(label)

        self.X = np.array(image_data)
        self.y = np.array(labels)

    def get_data(self) -> tuple:
        return (self.X, self.y)

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        label = None

        for member in root.findall('object'):
            label = member.find('name').text

        return label
    