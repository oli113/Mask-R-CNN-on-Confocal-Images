import cv2
import mrcnn.model as modellib
from mrcnn.visualize import InferenceConfig, get_mask_contours, random_colors, draw_mask

# Load Mask-RCNN
config = InferenceConfig(num_classes=1, image_size=1024)
model = modellib.MaskRCNN(mode="inference",config=config, model_dir="")
model.load_weights(filepath="dnn_model/mask_rcnn_object_0007.h5", by_name=True)

# Load image
img = cv2.imread("images/MH_01_05._orig.png")

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




# Detect objects
result = model.detect([image])[0]
class_ids = result["class_ids"]
object_count = len(class_ids)

# Random colors
colors = random_colors(object_count)

for i in range(object_count):
    # 1. Class ID
    class_id = result["class_ids"][i]


    # 2. Box
    box = result["rois"][i]
    y1, x1, y2, x2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)


    # 3. Score
    score = result["scores"][i]
    print(score)

    # 4.Mask
    mask = result["masks"][:,:,i]
    contours = get_mask_contours(mask)
    for cnt in contours:
        cv2.polylines(img, [cnt], True, colors[i], 2)
        img = draw_mask(img, [cnt], colors[i], alpha=0.5)

# Display image
cv2.imshow("Img", img)
cv2.waitKey(0)