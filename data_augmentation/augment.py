import numpy as np
import cv2
import json
import random
import copy
import matplotlib.pyplot as plt
import albumentations as A

#Please update the directory as you needed
image_dir = '<Update the directory>' #training image directory
json_dir = '<Update the directory' #training annotations directory

def draw_rect(img, bboxes, color=(255, 0, 0)):
    img = img.copy()
    for bbox in bboxes:
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img

def debug_bbox(x, h, w):
    x[0] = max(0., min(float(x[0]), w))
    x[1] = max(0., min(float(x[1]), h))
    x[2] = max(0., min(float(x[2]), w))
    x[3] = max(0., min(float(x[3]), h))
    return x

def getTransform(loop):
    print(loop)
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 2:
        transform = A.Compose([
            A.HueSaturationValue(p=1),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 3:
        transform = A.Compose([
            A.RandomSnow(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 4:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 5:
        transform = A.Compose([
            A.RandomRain(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 6:
        transform = A.Compose([
            A.FancyPCA(alpha=0.2, p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 7:
        transform = A.Compose([
            A.Blur(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 8:
        transform = A.Compose([
            A.GaussNoise(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 9:
        transform = A.Compose([
            A.RGBShift(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    elif loop == 10:
        transform = A.Compose([
            A.RandomShadow(p=1)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    return transform


# augmentation and saving json files and images to new directory (2x)

with open(f"{json_dir}", 'r') as f:
    data = json.load(f)
    im_file_name = data["images"]
    count = 0
    image_id_start = 532 #number of training image in the split
    an_id_start = 978 #number of annotations in training set
    danger = 0
    for i in range(len(im_file_name)):
        im_name = im_file_name[i]["file_name"]
        image = f"{image_dir}{im_name}"  # previous image dir (full)
        print(image)
        img = cv2.imread(image)[:, :, ::-1]

        n1 = image.split('.')[0]
        n2 = n1.split('/')[-1]  # image name without .jpg
        new_dir = '<Update the directory>'  # new root dir for augmented images
        filename_ = f"{new_dir}{n2}_augmented.jpg"  # new image dir (full)
        new_filename = filename_.split('/')[-1]  # image name to save in json
        h, w, channel = img.shape

        flag = False
        bbox = []
        index = []
        class_labels = []
        im_id = im_file_name[i]["id"]
        for an in data["annotations"]:
            if an["image_id"] == im_id:
                bbox.append(an["bbox"])
                index.append(an["id"])
                class_labels.append(an["category_id"])
        print(bbox)

        bbox_ = copy.deepcopy(bbox)

        print('w: ', w, " h: ", h)
        for box in bbox:
            box = debug_bbox(box, h, w)  # limiting box in the image shape range

        img_ = copy.deepcopy(img)
        transform = getTransform(random.randrange(11))  # we have 11 augmentation methods
        transformed = transform(image=img_, bboxes=bbox, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        print("Transformed bboxes")
        print(transformed_bboxes)
        transformed_class_labels = transformed['class_labels']

        if (len(transformed_bboxes) == 0):
            bboxes = bbox
            transformed_image = img
        else:
            bboxes = xywh_to_xyxy(np.array(transformed_bboxes))
            bboxes = bboxes.astype('int32')
        print(bboxes)
        image_with_box = draw_rect(transformed_image, bboxes)

        plt.imshow(image_with_box)
        plt.show()

        if (len(bboxes) == 0):
            bboxes = np.array(bbox_)
            image = img
            danger = danger + 1
            hi, wi, ch = img.shape

        else:
            bboxes = xyxy_to_xywh(bboxes)
            image = transformed_image
            hi, wi, ch = image.shape

        new_image_dict = {
            "id": image_id_start,
            "license": 1,
            "file_name": str(new_filename),
            "height": hi,
            "width": wi,
            "date_captured": "null"
        }
        im_file_name.append(new_image_dict)
        for i, j in zip(index, bboxes):
            an_dict = {
                "id": an_id_start,
                "image_id": image_id_start,
                "category_id": data["annotations"][i]["category_id"],
                "bbox": j.tolist(),
                "area": int(j[-2] * j[-1]),
                "segmentation": [],
                "iscrowd": 0}
            data["annotations"].append(an_dict)
            an_id_start += 1
        cv2.imwrite(filename_, image[:, :, ::-1])
        count += 1
        image_id_start += 1

print("Number of annotations problematic: ", danger)
# update image name, bbox, and area

save_dir = "<Update the directory>" #directory to save the annotations
with open(f"{save_dir}/_aug_annotations.coco.json", "w") as write_file:
    json.dump(data, write_file)

# augmentation and saving json files and images to new directory ____ 4x, 8x

with open(f"{json_dir}", 'r') as f:
    data = json.load(f)
    im_file_name = data["images"]
    count = 0
    image_id_start = 532
    an_id_start = 978
    danger = 0
    for iteration in range(1, 8):  # for creating 4x put range(1,4), for 8x put range(1,8)
        im_count = 0
        for i in range(len(im_file_name)):
            im_count += 1
            im_name = im_file_name[i]["file_name"]
            image = f"{image_dir}{im_name}"  # previous image dir (full)
            print(image)
            img = cv2.imread(image)[:, :, ::-1]

            n1 = image.split('.')[0]
            n2 = n1.split('/')[-1]  # image name without .jpg
            new_dir = '<Update the directory>'  # new root dir for augmented images
            filename_ = f"{new_dir}{n2}_augmented_{iteration}.jpg"  # new image dir (full)
            new_filename = filename_.split('/')[-1]  # image name to save in json
            h, w, channel = img.shape

            flag = False
            bbox = []
            index = []
            class_labels = []
            im_id = im_file_name[i]["id"]
            for an in data["annotations"]:
                if an["image_id"] == im_id:
                    bbox.append(an["bbox"])
                    index.append(an["id"])
                    class_labels.append(an["category_id"])
            print(bbox)
            bbox_ = copy.deepcopy(bbox)

            print('w: ', w, " h: ", h)
            for box in bbox:
                box = debug_bbox(box, h, w)  # limiting box in the image shape range

            img_ = copy.deepcopy(img)
            transform = getTransform(random.randrange(11))  # we have 11 augmentation methods
            transformed = transform(image=img_, bboxes=bbox, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            print("Transformed bboxes")
            print(transformed_bboxes)
            transformed_class_labels = transformed['class_labels']

            if (len(transformed_bboxes) == 0):
                bboxes = bbox
                transformed_image = img
            else:
                bboxes = xywh_to_xyxy(np.array(transformed_bboxes))
                bboxes = bboxes.astype('int32')
            print(bboxes)
            image_with_box = draw_rect(transformed_image, bboxes)

            plt.imshow(image_with_box)
            plt.show()


            if (len(bboxes) == 0):
                bboxes = np.array(bbox_)
                image = img
                danger = danger + 1
                hi, wi, ch = img.shape

            else:
                bboxes = xyxy_to_xywh(bboxes)
                image = transformed_image
                hi, wi, ch = image.shape

            new_image_dict = {
                "id": image_id_start,
                "license": 1,
                "file_name": str(new_filename),
                "height": hi,
                "width": wi,
                "date_captured": "null"
            }
            im_file_name.append(new_image_dict)

            for i, j in zip(index, bboxes):
                an_dict = {
                    "id": an_id_start,
                    "image_id": image_id_start,
                    "category_id": data["annotations"][i]["category_id"],
                    "bbox": j.tolist(),
                    "area": int(j[-2] * j[-1]),
                    "segmentation": [],
                    "iscrowd": 0}
                data["annotations"].append(an_dict)
                # print(an_dict)
                an_id_start += 1
            cv2.imwrite(filename_, image[:, :, ::-1])
            count += 1
            image_id_start += 1
            if im_count >= 532:
                break

print("Number of annotations problematic: ", danger)
# update image name, bbox, and area

save_dir = "<Update the directory>" #directory to save augmented annotation file
with open(f"{save_dir}/_aug_annotations.coco.json", "w") as write_file:
    json.dump(data, write_file)