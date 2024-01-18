import argparse
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import random
import math
import yaml
from collections import OrderedDict

# label 데이터 파일 확장자 추출 함수
def get_file_extension(labels):
    extensions = {os.path.splitext(extentions)[1][1:] for extentions in labels}
    # extenstions에 값이 존재하면 pop()을 사용하여 고유한 확장자 반환, 그렇지 않으면 None 반환
    return extensions.pop() if extensions else None

# XML 파일에서 고유한 클래스 추출 함수
def extract_classes_from_xml(opt, labels):
    classes = []
    # label_name에 해당하는 XML 파일에서 classes를 추출하는 반복문
    for label_name in labels:
        try:
            # xml 파일 파싱
            tree = ET.parse(os.path.join(opt.labelpath, label_name))
            root = tree.getroot()
            
            # object 아래 name 태그에서 클래스 레이블 추출
            for obj in root.findall("object"):
                class_label = obj.find("name").text
                classes.append(class_label)

        except ET.ParseError as e:
            print(f"Error parsing XML file {label_name}: {e}")
            return None

    uni_classes = list(np.unique(classes))
    num_classes = len(uni_classes)
    #  고유한 클래스 항목 목록 반환
    return uni_classes, num_classes

# XML 파일 형식을 YOLO 파일 형식으로 변환하는 함수
def convert_xml_to_yolo(opt, labels, classes):
    for label_name in labels:
        # XML 파일 파싱
        tree = ET.parse(os.path.join(opt.labelpath, label_name))
        root = tree.getroot()

        # XML 파일에서 이미지의 크기 정보를 추출
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_data_format = []
        # XML 파싱 정보를 추출하여 YOLO 형식으로 변환
        for member in root.findall('object'):
            # 객체의 클래스 이름 가져오기
            class_name = member.find('name').text
            if class_name not in classes:
                continue
            # 클래스 이름에 해당하는 인덱스(클래스 ID) 저장
            class_id = classes.index(class_name)

            # 바운딩 박스 좌표 추출
            bbox = member.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            # YOLO 형식에 맞게 좌표를 변환
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # 변환된 데이터를 문자열로 저장
            yolo_data_format.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
        
        # opt.savepath에서 지정한 디렉토리가 없으면 생성
        if not os.path.exists(opt.savepath):
            print(f"{opt.savepath} 해당 폴더가 없습니다.")
            os.makedirs(opt.savepath)
        
        # 변환된 YOLO 형식 데이터를 텍스트 파일에 쓰기
        save_path = os.path.join(opt.savepath, os.path.basename(os.path.splitext(label_name)[0]) + '.txt')
        with open(save_path, "w") as file:
            file.write("\n".join(yolo_data_format))

def split_dataset(opt):
    # opt.savepath의 상위 폴더를 루트 폴더로 설정
    root_folder = os.path.dirname(opt.savepath)

    # 생성할 서브 폴더 목록
    subfolders = ["train/images", "train/labels", "val/images", "val/labels", "test/images", "test/labels"]

    # 각 서브 폴더 생성(존재하지 않을 경우)
    for subfolder in subfolders:
        folder_path = os.path.join(root_folder, subfolder)
        if not os.path.exists(folder_path):
            print(f"{subfolder} 해당 폴더가 없습니다.")
            os.makedirs(folder_path, exist_ok=True)

    # 이미지 파일 목록 불러오기
    files = os.listdir(opt.imagepath)
    random.shuffle(files)

    # 데이터 셋 분할 비율 설정(8:1:1)
    # train_cnt = int(len(files) * 0.8)
    # val_cnt = int(len(files) * 0.1)
    # test_cnt = len(files) - train_cnt - val_cnt

    # 데이터 셋 분할 비율 설정
    train_cnt = math.ceil(int(len(files) * opt.trratio))
    val_cnt = int(len(files) * opt.vratio)
    test_cnt = len(files) - train_cnt - val_cnt

    # 각 파일을 대상 디렉터리로 복사
    for file in files[:train_cnt]:
        original_image_path = os.path.join(opt.imagepath, file)
        split_train_image_path = os.path.join(root_folder, "train/images", file)
        shutil.copy(original_image_path, split_train_image_path)

        original_label_file = os.path.splitext(file)[0] + ".txt"
        original_label_path = os.path.join(opt.savepath, original_label_file)
        split_train_label_path = os.path.join(root_folder, "train/labels", original_label_file)
        shutil.copy(original_label_path, split_train_label_path)

    for file in files[train_cnt:train_cnt + val_cnt]:
        original_image_path = os.path.join(opt.imagepath, file)
        split_val_image_path = os.path.join(root_folder, "val/images", file)
        shutil.copy(original_image_path, split_val_image_path)

        original_label_file = os.path.splitext(file)[0] + ".txt"
        original_label_path = os.path.join(opt.savepath, original_label_file)
        split_val_label_path = os.path.join(root_folder, "val/labels", original_label_file)
        shutil.copy(original_label_path, split_val_label_path)

    for file in files[train_cnt + val_cnt:]:
        original_image_path = os.path.join(opt.imagepath, file)
        split_test_image_path = os.path.join(root_folder, "test/images", file)
        shutil.copy(original_image_path, split_test_image_path)

        original_label_file = os.path.splitext(file)[0] + ".txt"
        original_label_path = os.path.join(opt.savepath, original_label_file)
        split_test_label_path = os.path.join(root_folder, "test/labels", original_label_file)
        shutil.copy(original_label_path, split_test_label_path)

    # 분할 결과 출력
    print(f"train: {train_cnt}, val: {val_cnt}, test: {test_cnt}")

# yolo 형식의 data yaml 파일 생성 함수
def create_yolo_yaml(opt, uni_classes,num_classes):
    # yaml 파일에서는 numpy 객체를 표준적인 방식으로 저장하지 않음
    # numpy 객체를 python 문자열로 변환
    uni_classes = [str(cls) for cls in uni_classes]

    yaml_str = f"""train: {os.path.join(os.path.dirname(opt.savepath), 'train/images')}
val: {os.path.join(os.path.dirname(opt.savepath), 'val/images')}
test: {os.path.join(os.path.dirname(opt.savepath), 'test/images')}
nc: {num_classes}
names: {uni_classes}
"""

    with open(os.path.join(os.path.dirname(opt.savepath), 'yolo.yaml'), 'w') as file:
        file.write(yaml_str)

    # data = {
    #     'train': str(os.path.join(os.path.dirname(opt.savepath), 'train/images')),
    #     'val': str(os.path.join(os.path.dirname(opt.savepath), 'val/images')),
    #     'nc': num_classes,
    #     'names': uni_classes,
    # }
    #
    # with open('yolo.yaml', 'w') as file:
    #     yaml.dump(data, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-path", type=str, dest="labelpath", help='label dataset path')
    parser.add_argument("--image-path", type=str, dest="imagepath", help='image dataset path')
    parser.add_argument("--save-path", type=str, dest="savepath", help='convert label format to YOLO format and save the converted data')
    parser.add_argument("--train-ratio", type=int, default=0.8, dest="trratio", help='training ratio')
    parser.add_argument("--val-ratio", type=int, default=0.1, dest="vratio", help='validataion ratio')
    opt = parser.parse_args()

    # label 데이터 셋 이름 리스트(확장자 포함)
    labels = [label_name for label_name in os.listdir(opt.labelpath)]
    extention = get_file_extension(labels)

    # xml 파일 형식일 때 YOLO 변환
    if extention == "xml":
        print(f"xml 형식의 라벨 데이터입니다.")
        uni_classes, num_classes = extract_classes_from_xml(opt, labels)
        convert_xml_to_yolo(opt, labels, uni_classes)
        split_dataset(opt)
        create_yolo_yaml(opt, uni_classes=uni_classes, num_classes=num_classes)
        
        
        