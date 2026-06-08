import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LassoCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
# from skimage.transform import ThinPlateSpline, warp

def apply_clahe(image):
    # Convert to LAB color space (L channel is lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    # --- First Pass: Apply CLAHE to the L channel ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    cl = clahe.apply(l_channel)
    # --- Second Pass: Apply standard equalizeHist to the CLAHE result ---
    # Note: This might make the image look overly bright or unnatural.
    cl_eq = cv2.equalizeHist(cl)
    # Merge the enhanced L channel back with A and B channels
    merged_channels = cv2.merge([cl_eq, a_channel, b_channel])
    # Convert back to BGR
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

def align_images_polynomial_warp(img_base, img_edit, edit_switcheroo=None, degree=2):
    # Convert images to grayscale
    try:
        color1 = apply_clahe(img_base)
        gray1 = cv2.equalizeHist(cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY))
        resized_edit = cv2.resize(img_edit, gray1.shape[::-1])
        color2 = apply_clahe(resized_edit)
        gray2 = cv2.equalizeHist(cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY))

        # --- DETECTOR ---
        # Use STAR or FAST detector (LUCID is descriptor-only)
        detector = cv2.KAZE_create(False, False, 0.001, 4, 4)# cv2.xfeatures2d.StarDetector_create()
        kp1 = detector.detect(color1, None)
        kp2 = detector.detect(color2, None)

        # --- DESCRIPTOR ---
        boostdesc = cv2.xfeatures2d.BoostDesc_create(
            desc=302,
            use_scale_orientation=False,
            scale_factor=16.0
        )

        # blur_kernel: pre-blur size (reduces noise)
        # lucid_kernel: patch size for comparison (higher = larger context)

        kp1, des1 = boostdesc.compute(color1, kp1)
        kp2, des2 = boostdesc.compute(color2, kp2)

        # --- MATCHER ---
        # LUCID descriptors are float-based → use L2 norm
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        if des1 is None or des2 is None:
            raise ValueError("No descriptors found — check detector or image content")

        # Use Brute-Force Matcher (Hamming distance works with BRIEF)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # FILTER START
        max_offset = 32  # pixels
        filtered_matches = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            dx = abs(pt1[0] - pt2[0])
            dy = abs(pt1[1] - pt2[1])

            if dx <= max_offset and dy <= max_offset:
                filtered_matches.append(m)
        matches = filtered_matches
        # FILTER END

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched point locations
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        print(f"FOUND {len(matches)} POINTS")
        # assert len(matches) > 60, 

        # Draw matches
        # img3 = cv2.drawMatches(img_base, kp1, resized_edit, kp2, filtered_matches, None,
        #                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3)
        # plt.show()


        model_i = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), PolynomialFeatures(degree=degree), RANSACRegressor(estimator=LassoCV(), min_samples=0.7))
        model_i.fit(points1, points2[:, 0])
        model_j = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), PolynomialFeatures(degree=degree), RANSACRegressor(estimator=LassoCV(), min_samples=0.7))
        model_j.fit(points1, points2[:, 1])

        # Create coordinate grid for remapping
        # height, width = img1_enhanced.shape[:2]

        XX, YY = np.meshgrid(np.arange(color1.shape[1]), np.arange(color1.shape[0]))
        coords = np.column_stack([XX.ravel(), YY.ravel()])

        map_i = np.clip(model_i.predict(coords), 0, color1.shape[1]).reshape(XX.shape).astype(np.float32)
        map_j = np.clip(model_j.predict(coords), 0, color1.shape[0]).reshape(YY.shape).astype(np.float32)

        # Edit switcheroo
        if edit_switcheroo is not None:
            resized_edit = cv2.resize(edit_switcheroo, gray1.shape[::-1])

        # Remap using the fitted mapping
        img2_aligned = cv2.remap(
            resized_edit, map_i, map_j,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return img2_aligned[:img_base.shape[0], :img_base.shape[1]]
    except:
        raise
        return img_edit

if __name__ == "__main__":

    img_1_path = "pic1.png"
    img_2_path = "pic2.png"

    # Load images
    img1 = cv2.imread(img_1_path) # Base Image
    img2 = cv2.imread(img_2_path) # Image with less illumination and shift
    # Perform alignment

    if False:
        img2 = cv2.copyMakeBorder(
                 img2, 
                 32, 
                 32, 
                 32, 
                 32, 
                 cv2.BORDER_CONSTANT, 
                 value=(0, 0, 0)
              )
    else:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    img2_aligned = align_images_polynomial_warp(np.array(img1), np.array(img2), np.array(img2))

    cv2.imwrite("cv2_outputs/1.png", img1)
    cv2.imwrite("cv2_outputs/2_old.png", img2) 
    cv2.imwrite("cv2_outputs/1_fix.png", np.array(remove_borders(img1)))
    cv2.imwrite("cv2_outputs/2_fix.png", img2_aligned)# img2_aligned)
