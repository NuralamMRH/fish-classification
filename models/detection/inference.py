import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
        
     
class YOLOInference:
    def __init__(self, model_path, imsz = (640, 640), conf_threshold = 0.05, nms_threshold = 0.3, yolo_ver = 'v10'):
        """
        Initializing a class with loading a model from TorchScript.
        Args:
        imsz: Size of input image to YOLO required
        conf_thresh: Confidence threshold to filter out low-confidence boxes.
        iou_thresh: IoU threshold for Non-Maximum Suppression.

        """
        self.device = torch.device("cpu")
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        
        self.yolo_ver = yolo_ver

        self.fp_16 = False
        self.imsz = imsz
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.letterbox = Letterbox(self.imsz)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (List(np.ndarray)): [(HWC) x B] for list.
        """

        im, params  = zip(*(self.letterbox(img) for img in im))
        
        im = np.stack(im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        im = im.to('cpu')
        im = im.half() if self.fp_16 else im.float()  # uint8 to fp16/32

        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im, params
    
    def v10postprocess(self, predictions):
        boxes, scores, labels = predictions.split([4, 1, 1], dim=-1)

        selected_boxes = []
        selected_scores = []
        for box_id in range(len(boxes)):
            if scores[box_id] > self.conf_threshold:
                new_box = [
                    max(0, boxes[box_id][0].item()), 
                    max(0, boxes[box_id][1].item()), 
                    min(self.imsz[0], boxes[box_id][2].item()), 
                    min(self.imsz[0], boxes[box_id][3].item())]
                
                selected_boxes.append(new_box)
                selected_scores.append(scores[box_id].item())
                
        if len(selected_boxes) != 0:
            
            selected_boxes = np.array(selected_boxes)
            selected_scores = np.array(selected_scores).reshape(-1, 1)
            boxes_scores = np.hstack([selected_boxes, selected_scores])

            indices = self.nms(boxes_scores)
            selected_boxes = boxes_scores[indices]

        return selected_boxes

    
    def v8postprocess(self, predictions):
        """
        Post-processing for the YOLO V8 object detection model.

        Arguments:
        predictions (np.array): Prediction tensor of size (1, 5, 8400).

        Returns:
        np.array:Array of filtered boxes after NMS.
        """
        # Extracting Predictions from a Tensor
        x_center, y_center, width, height, confidence = predictions

        # Convert coordinates from center to angular
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes = np.stack((x1, y1, x2, y2, confidence), axis=1)

        # Filtering boxes by confidence threshold
        boxes = boxes[boxes[:, 4] > self.conf_threshold]

        # Application of non-maximum suppression
        indices = self.nms(boxes)
        boxes = boxes[indices]

        return boxes

    def nms(self, boxes):
        """
        Non-Maximum Suppression (NMS) to remove overlapping boxes.

        Arguments:
        boxes (np.array): An array of boxes of size (N, 5), where N is the number of boxes.

        Returns:
        list: List of indexes of selected boxes.
        """

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            inter = w * h
            overlap = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(overlap <= self.nms_threshold)[0] + 1]

        return keep

    def scale_coords_back(self, img_shape, coords, params):
        # Rescale coords (xyxy) from target image shape to original image shape
        ratio, dh, dw = params
        gain = ratio
        
        coords[:, [0, 2]] -= dw  # x padding
        coords[:, [1, 3]] -= dh  # y padding
        
        coords[:, :4] /= gain
        
        coords = np.clip(coords, 0, [np.max(img_shape), np.max(img_shape), img_shape[1], img_shape[0], 1])
       
        condition = (coords[:, 2] - coords[:, 0] > 10) & (coords[:, 3] - coords[:, 1] > 10)
        coords = coords[condition]

        return coords
    
    def predict(self, im_bgr):
        
        # Checking the type of the input argument and casting to a list
        if isinstance(im_bgr, np.ndarray):
            im_bgr = [im_bgr]
            
        input_imgs, params = self.preprocess(im_bgr)
        
        with torch.no_grad():
            predictions = self.model(input_imgs)
        
        final_pred = []
        for bbox_id in range(len(predictions)):
            if self.yolo_ver == 'v8':
                filtered_boxes = self.v8postprocess(predictions[bbox_id])
            elif self.yolo_ver == 'v10':
                filtered_boxes = self.v10postprocess(predictions[bbox_id])
                
            if len(filtered_boxes) == 0:
                final_pred.append([])
            else:
                boxes = self.scale_coords_back(im_bgr[bbox_id].shape[:2], filtered_boxes, params[bbox_id])
                final_pred.append([YOLOResult(box, im_bgr[bbox_id]) for box in boxes])
        return final_pred
    

class YOLOResult:
    def __init__(self, box, image):
        """
        Initializes the YOLOResult.
        
        Args:
            box (list): List containing bounding box coordinates and confidence score.
            image (numpy.ndarray): Image from which the mask will be cropped.
        """
        self.box = box[:4].astype(int)
        self.score = box[4]
        self.x1, self.y1, self.x2, self.y2 = map(int, self.box)
        
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        
        self.mask = image[self.y1:self.y2, self.x1:self.x2]
        
        # Additional attributes for convenience
        self.center_x = self.x1 + self.width / 2
        self.center_y = self.y1 + self.height / 2
        
    def get_box(self):
        """
        Returns the bounding box coordinates.
        
        Returns:
            tuple: Bounding box coordinates (x1, y1, x2, y2).
        """
        return self.x1, self.y1, self.x2, self.y2
    
    def get_score(self):
        """
        Returns the confidence score.
        
        Returns:
            float: Confidence score.
        """
        return self.score
    
    def get_area(self):
        """
        Calculates the area of the bounding box.
        
        Returns:
            int: Area of the bounding box.
        """
        return self.width * self.height
    
    def draw_box(self, image, color=(0, 255, 0), thickness=2):
        """
        Draws the bounding box on the image.
        
        Args:
            image (numpy.ndarray): Image on which the box will be drawn.
            color (tuple): Color of the box in (B, G, R) format.
            thickness (int): Thickness of the box lines.
        """
        cv2.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
    
    def draw_label(self, image, label, color=(0, 255, 0), font_scale=0.5, thickness=1):
        """
        Draws a label next to the bounding box.
        
        Args:
            image (numpy.ndarray): Image on which the label will be drawn.
            label (str): Text of the label.
            color (tuple): Color of the text in (B, G, R) format.
            font_scale (float): Font scale of the text.
            thickness (int): Thickness of the text lines.
        """
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top_left = (self.x1, self.y1 - label_size[1])
        bottom_right = (self.x1 + label_size[0], self.y1)
        
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, label, (self.x1, self.y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    def get_mask_BGR(self):
        """
        Returns the mask in BGR format.
        
        Returns:
            numpy.ndarray: Mask in BGR format.
        """
        return self.mask

    def get_mask_RGB(self):
        """
        Returns the mask in RGB format.
        
        Returns:
            numpy.ndarray: Mask in RGB format.
        """
        return cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)

    def __repr__(self):
        return f"YOLOResult(box_xyxy=({self.x1}, {self.y1}, {self.x2}, {self.y2}), score={self.score})"

    def to_dict(self):
        """
        Converts the object to a dictionary.
        
        Returns:
            dict: Dictionary with keys 'box' and 'score'.
        """
        return {
            'box': [self.x1, self.y1, self.x2, self.y2],
            'score': self.score,
            'area': self.get_area(),
            'center': (self.center_x, self.center_y)
        }
    
class Letterbox:
    def __init__(self, target_size, color=(0, 0, 0)):
        self.target_size = target_size
        self.color = color

    def __call__(self, image):
        return self.letterbox(image)

    def letterbox(self, image):
        shape = image.shape[:2]  # current shape [height, width]
        new_shape = self.target_size

        # Scale ratio (new / old)
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[0] * ratio)), int(round(shape[1] * ratio))
        
        # Compute padding
        dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        return image, [ratio, dh, dw]