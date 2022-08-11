import json
import cv2


save_folder_dir = '<Update the directory>' #directory to save the annotations
json_dir = '<Update the directory>' #coco json directory with file name
src_im_dir = '<Update the directory>' #image directory

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]



with open(json_dir, "r") as f:
    data = json.load(f)

im_info = data['images']
an_info = data['annotations']

for im in im_info:

    file_name = im['file_name']
    im_dir = src_im_dir + file_name
    print(im_dir)
    image = cv2.imread(im_dir)
    print(image.shape)
    image_h = image.shape[0]
    image_w = image.shape[1]
    file_name = file_name.split('.')[0]

    im_id = im["id"]
    bbox = []
    category = []
    for an in an_info:
        # print(an)
        if an["image_id"] == im_id:
            bbox.append(an["bbox"])
            category.append(an["category_id"])

    with open(save_folder_dir + '/' + file_name + '.txt', 'w') as f:
        check = 1
        for cat, box in zip(category, bbox):
            b = convert_bbox_coco2yolo(image_w, image_h, box)
            f.write(str(cat) + ' ' + "{:.6f}".format(b[0]) + ' ' + "{:.6f}".format(b[1]) + ' ' + "{:.6f}".format(
                b[2]) + ' ' + "{:.6f}".format(b[3]))
            if check <= len(category) - 1:
                f.write('\n')
            check = check + 1

