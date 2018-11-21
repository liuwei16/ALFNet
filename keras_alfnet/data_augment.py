from __future__ import division
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
import numpy.random as npr
# da functions
def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min,max)
    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def augment(img_data, c, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])
    img_height, img_width = img.shape[:2]

    if augment:
        # random brightness
        if c.brightness and np.random.randint(0, 2) == 0:
            img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
        # random horizontal flip
        if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]
        # random crop a patch
        ratio = np.random.uniform(c.scale[0], c.scale[1])
        crop_h, crop_w = np.asarray(ratio * np.asarray(img.shape[:2]), dtype=np.int)
        gts = np.copy(img_data_aug['bboxes'])
        igs = np.copy(img_data_aug['ignoreareas'])
        if len(gts) > 0:
            sel_id = np.random.randint(0, len(gts))
            sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
            sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, img_width - crop_w) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, img_height - crop_h) + crop_h * 0.5)
        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - img_width, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - img_height, int(0))
        crop_y1 -= diff_y
        patch_X = np.copy(img[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
        img = patch_X
        if len(igs) > 0:
            igs[:, [0, 2]] -= crop_x1
            igs[:, [1, 3]] -= crop_y1
            y_coords = igs[:, [1, 3]]
            y_coords[y_coords < 0] = 0
            y_coords[y_coords >= crop_h] = crop_h - 1
            igs[:, [1, 3]] = y_coords
            x_coords = igs[:, [0, 2]]
            x_coords[x_coords < 0] = 0
            x_coords[x_coords >= crop_w] = crop_w - 1
            igs[:, [0, 2]] = x_coords
            after_area = (igs[:, 2] - igs[:, 0]) * (igs[:, 3] - igs[:, 1])
            igs = igs[after_area > 100]
        if len(gts) > 0:
            before_limiting = copy.deepcopy(gts)
            gts[:, [0, 2]] -= crop_x1
            gts[:, [1, 3]] -= crop_y1
            y_coords = gts[:, [1, 3]]
            y_coords[y_coords < 0] = 0
            y_coords[y_coords >= crop_h] = crop_h - 1
            gts[:, [1, 3]] = y_coords
            x_coords = gts[:, [0, 2]]
            x_coords[x_coords < 0] = 0
            x_coords[x_coords >= crop_w] = crop_w - 1
            gts[:, [0, 2]] = x_coords
            before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                before_limiting[:, 3] - before_limiting[:, 1])
            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            gts = gts[after_area >= c.in_thre * before_area]

        # resize to original image size
        img = cv2.resize(img, dsize=(c.random_crop[1], c.random_crop[0]))
        reratio = crop_h/c.random_crop[0]
        if len(gts) > 0:
            gts = (gts/reratio).astype(np.int)
            w = gts[:,2]-gts[:,0]
            gts = gts[w>=16,:]
        if len(igs) > 0:
            igs = (igs / reratio).astype(np.int)
            w, h = igs[:,2]-igs[:,0], igs[:,3]-igs[:,1]
            igs = igs[np.logical_and(w>=8, h>=8),:]
        img_data_aug['bboxes'] = gts
        img_data_aug['ignoreareas'] = igs
    else:
        crop_h, crop_w = c.random_crop[0], c.random_crop[1]
        crop_ymin = int((img_height - crop_h) / 2)
        crop_xmin = int((img_width - crop_w) / 2)
        patch_X = np.copy(img[crop_ymin:crop_ymin + crop_h, crop_xmin:crop_xmin + crop_w])
        img = patch_X
        if len(img_data_aug['ignoreareas']) > 0:
            boxes = copy.deepcopy(img_data_aug['ignoreareas'])
            boxes[:, [0, 2]] -= crop_xmin
            boxes[:, [1, 3]] -= crop_ymin
            y_coords = boxes[:, [1, 3]]
            y_coords[y_coords < 0] = 0
            y_coords[y_coords >= crop_h] = crop_h - 1
            boxes[:, [1, 3]] = y_coords
            x_coords = boxes[:, [0, 2]]
            x_coords[x_coords < 0] = 0
            x_coords[x_coords >= crop_w] = crop_w - 1
            boxes[:, [0, 2]] = x_coords
            after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[after_area > 100]
            img_data_aug['ignoreareas'] = boxes
        if len(img_data_aug['bboxes']) > 0:
            boxes = copy.deepcopy(img_data_aug['bboxes'])
            boxes[:, [0, 2]] -= crop_xmin
            boxes[:, [1, 3]] -= crop_ymin
            before_limiting = copy.deepcopy(boxes)

            y_coords = boxes[:, [1, 3]]
            y_coords[y_coords < 0] = 0
            y_coords[y_coords >= crop_h] = crop_h - 1
            boxes[:, [1, 3]] = y_coords

            x_coords = boxes[:, [0, 2]]
            x_coords[x_coords < 0] = 0
            x_coords[x_coords >= crop_w] = crop_w - 1
            boxes[:, [0, 2]] = x_coords

            before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                before_limiting[:, 3] - before_limiting[:, 1])
            after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[after_area >= c.in_thre * before_area]
            img_data_aug['bboxes'] = boxes

    # gt = img_data_aug['bboxes']
    # ig = img_data_aug['ignoreareas']
    # imgsh = np.copy(img)
    # for i in range(len(gt)):
    #     (x1, y1, x2, y2) = gt[i, :]
    #     cv2.rectangle(imgsh, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # for i in range(len(ig)):
    #     (x1, y1, x2, y2) = ig[i, :]
    #     cv2.rectangle(imgsh, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # plt.imshow(imgsh, interpolation='bicubic')
    # plt.close()
    img_data_aug['width'] = c.random_crop[1]
    img_data_aug['height'] = c.random_crop[0]
    return img_data_aug, img

def random_crop(image, gts, igs, crop_size, limit=8):
    img_height, img_width = image.shape[0:2]
    crop_h, crop_w = crop_size

    if len(gts)>0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w+1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h+1) + crop_h * 0.5)

    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    # crop detections
    if len(igs)>0:
        igs[:, 0:4:2] -= crop_x1
        igs[:, 1:4:2] -= crop_y1
        igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
        igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
        keep_inds = ((igs[:, 2] - igs[:, 0]) >=8) & \
                    ((igs[:, 3] - igs[:, 1]) >=8)
        igs = igs[keep_inds]
    if len(gts)>0:
        ori_gts = np.copy(gts)
        gts[:, 0:4:2] -= crop_x1
        gts[:, 1:4:2] -= crop_y1
        gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
        gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

        before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        keep_inds = ((gts[:, 2] - gts[:, 0]) >=limit) & \
                    (after_area >= 0.5 * before_area)
        gts = gts[keep_inds]

    return cropped_image, gts, igs
def random_pave(image, gts, igs, pave_size, limit=8):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    pave_x = int(np.random.randint(0, pave_w-img_width+1))
    pave_y = int(np.random.randint(0, pave_h-img_height+1))
    paved_image[pave_y:pave_y+img_height, pave_x:pave_x+img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >=8) & \
                    ((igs[:, 3] - igs[:, 1]) >=8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
        gts = gts[keep_inds]

    return paved_image, gts, igs
def resize_image(image, gts,igs, scale=[0.6,1.4]):
    height, width = image.shape[0:2]
    ratio = np.random.uniform(scale[0], scale[1])
    if len(gts)>0 and np.max(gts[:,3]-gts[:,1])>=300:
        ratio = np.random.uniform(scale[0], 1.0)
    new_height, new_width = int(ratio*height), int(ratio*width)
    image = cv2.resize(image, (new_width, new_height))
    if len(gts)>0:
        gts = np.asarray(gts,dtype=float)
        gts[:, 0:4:2] *= ratio
        gts[:, 1:4:2] *= ratio

    if len(igs)>0:
        igs = np.asarray(igs, dtype=float)
        igs[:, 0:4:2] *= ratio
        igs[:, 1:4:2] *= ratio

    return image, gts, igs

def augment_resizecrop(img_data, c):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])
    img_height, img_width = img.shape[:2]

    # random brightness
    if c.brightness and np.random.randint(0, 2) == 0:
        img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
    # random horizontal flip
    if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        # plt.imshow(img,interpolation='bicubic')
        if len(img_data_aug['bboxes']) > 0:
            img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
        if len(img_data_aug['ignoreareas']) > 0:
            img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]

    gts = np.copy(img_data_aug['bboxes'])
    igs = np.copy(img_data_aug['ignoreareas'])

    # img, gts, igs = resize_image(img, gts, igs, scale=[0.5,1.4])
    img, gts, igs = resize_image(img, gts, igs, scale=[0.4,1.5])
    if img.shape[0]>=c.random_crop[0]:
        img, gts, igs = random_crop(img, gts, igs, c.random_crop,limit=16)
    else:
        img, gts, igs = random_pave(img, gts, igs, c.random_crop,limit=16)

    img_data_aug['bboxes'] = gts
    img_data_aug['ignoreareas'] = igs

    # gt = img_data_aug['bboxes']
    # ig = img_data_aug['ignoreareas']
    # imgsh = np.copy(img)
    # if len(gt) > 0 and len(gt[(gt[:, 3] - gt[:, 1]) > 300, :]) > 0:
    #     for i in range(len(gt)):
    #         (x1, y1, x2, y2) = gt[i, :]
    #         cv2.rectangle(imgsh, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    #     for i in range(len(ig)):
    #         (x1, y1, x2, y2) = ig[i, :]
    #         cv2.rectangle(imgsh, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #     plt.imshow(imgsh, interpolation='bicubic')
    #     plt.close()

    img_data_aug['width'] = c.random_crop[1]
    img_data_aug['height'] = c.random_crop[0]

    return img_data_aug, img
