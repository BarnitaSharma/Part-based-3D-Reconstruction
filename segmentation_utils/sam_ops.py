import numpy as np

def sam_point(predictor, image, x, y, label=1):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([label]),
        multimask_output=True
    )
    return masks[np.argmax(scores)]

def sam_box(predictor, image, box):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=np.array(box)[None, :],
        multimask_output=True
    )
    return masks[np.argmax(scores)]
