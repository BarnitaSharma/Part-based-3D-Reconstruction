import numpy as np

def sam_point(predictor, image, x, y, pos_label=1):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([pos_label], dtype=np.int32),
        multimask_output=True
    )
    return masks[scores.argmax()]

def sam_box(predictor, image, box_xyxy):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=np.array(box_xyxy, dtype=np.int32)[None, :],
        multimask_output=True
    )
    return masks[scores.argmax()]
