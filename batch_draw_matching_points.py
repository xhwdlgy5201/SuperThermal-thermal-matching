import cv2
import torch
import random
import argparse
import numpy as np

from utils.common_utils import gct, nearest_neighbor_distance_ratio_match
from model.thermal_des import HardNetwork   
from model.thermal_det_so import ThermalDetSO 
from model.thermal_net_so import ThermalNetSO 
from skimage.measure import ransac as _ransac


def My_drawMatches(img1, kp1, img2, kp2, matches):
    """
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])


    for match in matches:
        # Get the matching keypoints for each of the images
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (255, 0, 0), -1)     
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (0, 255, 0), -1)
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), color, 1, lineType=cv2.LINE_AA, shift=0)

    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference multiple images.")
    parser.add_argument("--resume", default=None, type=str, help="Path to model checkpoint.")
    parser.add_argument("--folder", default=None, type=str, help="Path to input folder.")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    # Initialize model
    det = ThermalDetSO(100., 100., 0, 5, 512, 15, 0.5, 3, 1, 1, [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0])
    des = HardNetwork(1, 5)
    model = ThermalNetSO(det, des, 1000, 1, 32, 512)

    # Load model checkpoint
    device = torch.device("cuda")
    model = model.to(device)
    resume = args.resume
    input_folder = args.folder
    print(f"Loading model from {resume}")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])


    # Detect and compute keypoints for each image in input folder
    import glob
    path = input_folder + "/*.jpg"
    print(glob.glob(path))
    for i in range(len(glob.glob(path))-1):
        img1_path = sorted(glob.glob(path))[i]
        img2_path = sorted(glob.glob(path))[i+1]

        print('Image 1 path:', img1_path)
        print('Image 2 path:', img2_path)

        kp1, des1, img1 = model.detectAndCompute(img1_path, device, (240, 320))
        kp2, des2, img2 = model.detectAndCompute(img2_path, device, (240, 320))

        # Perform keypoint matching
        predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.8)
        idx = predict_label.nonzero().view(-1)
        mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
        mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2


        def to_cv2_kp(kp):
            # kp is like [batch_idx, y, x, channel]
            return cv2.KeyPoint(kp[2], kp[1], 0)

        def to_cv2_dmatch(m):
            return cv2.DMatch(m, m, m, m)

        def reverse_img(img):
            """
            reverse image from tensor to cv2 format
            :param img: tensor
            :return: RBG image
            """
            img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
            img = (img * 255).astype(np.uint8)  # change to opencv format
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
            return img

        img1, img2 = reverse_img(img1), reverse_img(img2)
        keypoints1 = list(map(to_cv2_kp, mkp1))
        keypoints2 = list(map(to_cv2_kp, mkp2))


        DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))
        print(mkp1.data.cpu().numpy()[:,1:3])
        H, mask = cv2.findHomography(mkp1.data.cpu().numpy()[:,1:3], mkp2.data.cpu().numpy()[:,1:3], cv2.RANSAC, 10)

        print('Total match no. is: ', list(mask).count(1))
        ransac_match = My_drawMatches(img1, keypoints1, img2, keypoints2, [m for i,m in enumerate(DMatch) if mask[i]])


        outImg = My_drawMatches(img1, keypoints1, img2, keypoints2, DMatch)


        cv2.imwrite("output_img_%04d.png"%i, outImg)
