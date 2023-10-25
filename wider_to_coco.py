import os
import json
import argparse
from PIL import Image


def parse_wider_annots(annots_file):
    """
          Input: WIDER-format annotation file:
            - first line is image file name
            - second line is an integer, for `n` faces in that image
            - next `n` lines are bounding box(bbox) coordinates
            - again, next line is image file name
            - bbox are [top_left_x top_left_y w h blur expression illumination invalid occlusion pose]
          Returns a dict: {'img_filename': [[x,y,w,h], [x,y,w,h], ....]}
        """

    # read annotation file
    ann_file = open(annots_file, 'r')

    is_img = True          # a line has image name or not
    isNum_annots = False   # a line has number of annotations or not
    start_annots_count = False
    annots_count = 0
    num_annots = -1

    # initialize dictionary to add image paths and their corresponding annotations
    ann_dict = {}
    img_name = ''     # initialize empty image name

    for line in ann_file:
        line = line.strip()
        
        if line == '0 0 0 0 0 0 0 0 0 0':
            if annots_count == num_annots + 1:
                start_annots_count = False
                annots_count = 0
                is_img = True  # next line is image file
                isNum_annots = False
                num_annots = -1
                ann_dict.pop(img_name)
            continue

        if is_img:
            # current line has image name
            is_img = False     # false until next line comes with image name
            isNum_annots = True
            img_name = line
            ann_dict[img_name] = []  # creating a key in dictionary for current image name
            continue

        if isNum_annots:
            # get number of annotations
            num_annots = int(line)
            isNum_annots = False     # false until reach to next image annotations
            if num_annots > 0:
                start_annots_count = True  # start counting annotations
                annots_count = 0
            else:
                # no annotations for this image
                is_img = True     # next line will be the next image name
                num_annots = -1   # will count again
            continue

        if start_annots_count:
            # After the line with # annotations, annotations: [x,y,w,h,......] start
            annots = [float(x) for x in line.split()]  # split on whitespace
            ann_dict[img_name].append(annots[:4])      # add annotations:[x,y,w,h] for current image
            annots_count += 1

        if annots_count == num_annots:
            # gone through all the annotations for current image
            start_annots_count = False
            annots_count = 0
            is_img = True        # next line will be image name
            isNum_annots = False
            num_annots = -1

    return ann_dict


def wider_to_coco_json(args):
    """"----****---- Convert WIDERFace annotation format to COCO bounding box format ----****----"""

    # whether convert train, val or both
    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    # check for directory to save the outputs
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.out_dir = args.data_dir

    categories = [{"id": 1, "name": 'face', "supercategory": 'face'}]   # WIDER has only one category (human face)
    for split in splits:
        print(f'Processing {split}....')
        out_json_name = os.path.join(args.out_dir, f'instances_{split}.json')
        img_dir = os.path.join(args.data_dir, f'WIDER_{split}', 'images')

        # initialize image ids, annotations ids, and category ids
        img_id = 0
        ann_id = 0
        cat_id = 1

        ann_dict = {}      # dict to store all the annotations
        images = []        # to store image info
        annotations = []   # to store annotations for an image

        # read WIDER annotations text file
        ann_file = os.path.join(args.data_dir, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')
        wider_annot_dict = parse_wider_annots(ann_file)    # [im_name] = [[x,y,w,h], ...]

        for img_name in wider_annot_dict.keys():
            if len(images) % 100 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))

            # add image info
            image = {'id': img_id}
            img_id += 1
            im = Image.open(os.path.join(img_dir, img_name))
            image['width'] = im.width
            image['height'] = im.height
            image['file_name'] = img_name
            images.append(image)

            # add annotations
            for bbox in wider_annot_dict[img_name]:
                ann = {'id': ann_id}
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id      # 1:"face" for WIDER (fixed for all)
                ann['iscrowd'] = 0
                ann['area'] = bbox[2] * bbox[3]
                ann['boxes'] = bbox
                ann['bbox'] = bbox[:4]
                annotations.append(ann)

        # add all the details to dictionary
        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Number of categories: %s" % len(categories))
        print("Number of images: %s" % len(images))
        print("Number of annotations: %s" % len(annotations))
        with open(out_json_name, 'w', encoding='utf8') as outfile:
            json.dump(ann_dict, outfile, indent=4, sort_keys=True)


# create argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Convert WIDERFace annotation format to COCO Bbox format')
    parser.add_argument(
        '-d', '--data_dir', help="path to WIDERFace dataset dir", default='data/WIDER', type=str)

    parser.add_argument(
        '-s', '--split', help="which split to convert", default='all', choices=['all', 'train', 'val'], type=str)

    parser.add_argument(
        '-o', '--out_dir', help="where to output the annotation file, default same as data dir", default='data/WIDER/annotations')
    return parser.parse_args()


# main function
if __name__ == '__main__':
    args = parse_args()
    wider_to_coco_json(args)
