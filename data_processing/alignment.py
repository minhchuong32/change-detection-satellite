import cv2
import numpy as np
from PIL import Image

def align_images_orb(img1, img2, max_features=2000):
    """
    Căn chỉnh ảnh vệ tinh T2 theo T1 sử dụng ORB + RANSAC.
    """
    # Chuyển đổi sang numpy array nếu đầu vào là PIL
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Chuyển sang grayscale
    gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

    # Khởi tạo ORB detector
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return img2 

    # Matching features
    bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = bf.match(des1, des2)
    
    # Sắp xếp và lấy 20% matches tốt nhất
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.2)]

    if len(good_matches) < 4: # Cần ít nhất 4 điểm để tìm Homography
        return img2

    # Lấy tọa độ các điểm khớp
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Tìm ma trận Homography bằng RANSAC
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    # Thực hiện Warp (Căn chỉnh)
    height, width, channels = img1_np.shape
    aligned_img = cv2.warpPerspective(img2_np, H, (width, height))
    
    return Image.fromarray(aligned_img)