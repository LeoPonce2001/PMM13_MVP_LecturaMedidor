from PIL import Image, ImageEnhance
import numpy as np
import cv2

def warpQuad(image, points):
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required")
    
    src = np.array(points, dtype=np.float32)
    
    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    max_width = max(width_top, width_bottom)
    
    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])
    max_height = max(height_left, height_right)
    
    target_width = int(round(max_width))
    target_height = int(round(max_height))
    
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    warped_cv = cv2.warpPerspective(img_cv, M, (target_width, target_height))
    warped_pil = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
    
    return warped_pil.convert("RGB")

def prepareImage(img, points):
    img = warpQuad(img, points)
    img = img.convert("L").convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(3)
    return img.resize((20, 32))
