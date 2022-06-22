import mrcnn.model as modellib
import cv2
from mrcnn.visualize import random_colors, get_mask_contours, InferenceConfig, draw_mask

# Load Model
num_classes = 1
config = InferenceConfig(num_classes=num_classes, image_size=1024)
model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
model.load_weights("dnn_model/mask_rcnn_object_0007.h5", by_name=True)

# Generate random colors
colors = random_colors(num_classes)

# Load Camera
cap = cv2.VideoCapture("videos/005-2018-from106to168_shadowremoved.mpg")

while True:
    # Get frame
    ret, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect results
    results = model.detect([image])
    r = results[0]
    object_count = len(r["class_ids"])

    for i in range(object_count):
        # 1. Class ID
        class_id = r["class_ids"][i]
        print("Class id", class_id)

        # 2. Rectangle
        y1, x1, y2, x2 = r["rois"][i]
        #cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)

        # 3. Mask
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)
        for cnt in contours:
            cv2.polylines(img, [cnt], True, colors[class_id], 2)
            img = draw_mask(img, [cnt], colors[class_id])

        # 4 Score
        score = r["scores"][i]

    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
