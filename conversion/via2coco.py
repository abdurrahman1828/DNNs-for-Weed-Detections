import json
import os
import cv2

json_dir = "<Update the directory>" #VIA directory
img_dir = "<Update the directory>" #source image directory
json_files = os.listdir(json_dir)
names = []
for file in json_files:
    name = file.split(".")[0]
    names.append(name)

category = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2
}

c = 0  # image id counter
a = 0  # annotation id counter
images = []
annotations = []

for js, name in zip(json_files, names):
    # print(f"{json_dir}/{js}")

    with open(f"{json_dir}/{js}", 'r') as f:
        data = json.load(f)
        json_obj = f"via_{name}"
        file_name = data[json_obj]["filename"]
        print(file_name)
        im = cv2.imread(f"{img_dir}/{file_name}")
        width = im.shape[1]
        height = im.shape[0]
        dictionary = {
            "id": c,
            "license": 1,
            "file_name": str(file_name),
            "width": width,
            "height": height,
            "date_captured": "null"
        }
        images.append(dictionary)
        for i in range(len(data[json_obj]['regions'])):
            category_id = category.get(data[json_obj]['regions'][i]['region_attributes']["class"])
            x = int(data[json_obj]['regions'][i]['shape_attributes']["x"])
            y = int(data[json_obj]['regions'][i]['shape_attributes']["y"])
            w = int(data[json_obj]['regions'][i]['shape_attributes']["width"])
            h = int(data[json_obj]['regions'][i]['shape_attributes']["height"])
            bbox = [x, y, w, h]
            area = int(w * h)
            an_dict = {
                "id": a,
                "image_id": c,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0}
            annotations.append(an_dict)
            a = a + 1
        c = c + 1


info = {
    "year": "2021",
    "version": "1",
    "description": "Cotton Weed by Dr. Yuzhen Lu",
    "contributor": "",
    "url": "",
    "date_created": ""
}
licenses = [
    {
        "id": 1,
        "url": "",
        "name": ""
    }
]

categories = [
    {
        "id": 0,
        "name": "carpetweed",
        "supercategory": "cotton_weed"
    },
    {
        "id": 1,
        "name": "morningglory",
        "supercategory": "cotton_weed"
    },
    {
        "id": 2,
        "name": "palmer_amaranth",
        "supercategory": "cotton_weed"
    }
]

final_json = {"info": info, "licenses": licenses, "categories": categories, "images": images, "annotations": annotations}

save_dir = "<Update the directory>" #coco json save directory
with open(f"{save_dir}/_annotations.coco.json", "w") as write_file:
    json.dump(final_json, write_file)