import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from torchvision import transforms

        
class Inference:
    def __init__(self, model_path, image_size=416, threshold=0.5, poly_dict = True, max_points = 250):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.cpu()
        
        
        self.max_points = max_points
        self.IMAGE_SIZE = image_size
        self.THRESHOLD = threshold
        
        self.loader = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: np.ndarray):
        # Converting an image to a tensor and normalizing
        pil_image = Image.fromarray(image)
        input_tensor = self.loader(pil_image)
        
        return input_tensor
    
    def postprocess(self, logit, src_size):
        height, width = src_size
        #(1, img_size, img_size) -> (img_size, img_size)
        pr_mask = logit[0].numpy()
        pr_mask = resize_logits_mask_pil(pr_mask, width, height)
        pr_mask = pr_mask > self.THRESHOLD
        contours = bitmap_to_polygon(pr_mask)
        poly, valid_state = full_fix_contour(contours)
        poly = poly.astype(int)

        return poly, valid_state

    def predict(self, images):
        # Checking the type of the input argument and casting to a list
        if isinstance(images, np.ndarray):
            images = [images]
           
        #insurance in case you somehow end up with an empty list
        if len(images) == 0: return []
        
        # Preprocessing images and saving their sizes
        _input = [self.preprocess(image) for image in images]
        src_sizes = [image.shape[:2] for image in images] # HEIGHT - WIDTH
        
        _input = torch.stack(_input)
        
        # Processing a batch of images
        return self.predict_batch(_input, src_sizes)
    

    def predict_batch(self, _input, src_sizes):
        results = []
        start_time = time.time()
        
        with torch.no_grad():
            logits = self.model(_input).sigmoid()

        for idx, src_size in enumerate(src_sizes):
            logit = logits[idx]
            poly, valid_state = self.postprocess(logit, src_size)
            
            if len(poly) != 0:
                poly = approximate_to_max_point_cnt(poly, max_points=self.max_points)
            else:
                poly = [(0,0), (src_size[1],0), (src_size[1], src_size[0]), (0, src_size[0]), (0,0)]
                
            results.append(FishialPolygon(poly))
            
            duration = time.time() - start_time

        return results
    
    
def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    contours = [c.reshape(-1, 2) for c in contours]
    return sorted(contours, key=len, reverse = True)

def poly_array_to_dict(polygon):
    """
    Converts an array of polygon points into a dictionary with labeled coordinates.

    Args:
        polygon (ndarray): An array of points representing the polygon. Each point is an array [x, y].

    Returns:
        dict: A dictionary where keys are labeled coordinates ('x1', 'y1', 'x2', 'y2', etc.) 
              and values are the corresponding x and y coordinates from the input array.
    """
    polygons_dict = {}
    
    for i, point in enumerate(polygon):
        # Add x coordinate with label 'x{i+1}'
        polygons_dict[f"x{i + 1}"] = int(point[0])
        
        # Add y coordinate with label 'y{i+1}'
        polygons_dict[f"y{i + 1}"] = int(point[1])
    
    return polygons_dict

def is_contour_valid(contour):
    """
    Checks if a contour is valid (i.e., its lines do not intersect).

    Args:
        contour (ndarray): The contour represented as an array of points.

    Returns:
        bool: True if the contour is valid, False otherwise.
    """
    if len(contour) < 3:
        # A contour must contain at least three points to be a polygon
        return False
    
    polygon = Polygon(contour)
    
    # Check for self-intersection
    if not polygon.is_valid:
        return False
    
    # Check for intersection between the start and end points (to close the contour)
    ring = LinearRing(contour)
    if not ring.is_simple:
        return False
    
    return True

def fix_contour(contour):
    """
    Fixes a damaged contour (removes self-intersections).

    Args:
        contour (ndarray): The contour represented as an array of points.

    Returns:
        ndarray: The fixed contour.
    """
    polygon = Polygon(contour)
    if polygon.is_valid:
        return contour
    
    # Fix the contour using buffer(0)
    fixed_polygon = polygon.buffer(0)
    
    if fixed_polygon.is_empty:
        return np.array([])  # Return an empty array if the contour cannot be fixed
    
    # Check the type of the returned object
    if isinstance(fixed_polygon, Polygon):
        fixed_contour = np.array(fixed_polygon.exterior.coords)
    elif isinstance(fixed_polygon, MultiPolygon):
        # If it's a MultiPolygon, choose the polygon with the largest area
        largest_polygon = max(fixed_polygon.geoms, key=lambda p: p.area)
        fixed_contour = np.array(largest_polygon.exterior.coords)

    return fixed_contour

def full_fix_contour(poly):
    """
    Attempts to validate and fix a polygon contour. If the contour is valid, it returns the contour.
    If the contour is invalid, it tries to fix it. If the fix is successful, it returns the fixed contour.

    Args:
        poly (ndarray): An array of polygons, where each polygon is represented as an array of points.

    Returns:
        tuple: A tuple containing the following:
            - ndarray: The valid or fixed contour. If the contour cannot be fixed, an empty array is returned.
            - str: A message indicating the status of the contour ("Empty Contour", "Fixed Contour", or "Can't fix").
    """
    if len(poly) == 0 or len(poly[0]) < 10:
        return [], "Empty Contour"
    
    contour = poly[0]
    
    # Check the validity of the contour
    if is_contour_valid(contour):
        return contour, None
    else:
        # Attempt to fix the contour
        fixed_contour = fix_contour(contour)
        if fixed_contour.size > 0 and is_contour_valid(fixed_contour):
            return fixed_contour, "Fixed Contour"
        else:
            return [], "Can't fix"

def resize_logits_mask_pil(logits_mask, width, height):
    """
    Resize a logits mask to the specified output shape using PIL.

    Parameters:
        logits_mask (np.array): Input logits mask.
        width (int): Desired width of the output shape.
        height (int): Desired height of the output shape.

    Returns:
        np.array: Resized logits mask.
    """
    # Convert logits mask to float32 for PIL compatibility
    mask_float32 = logits_mask.astype(np.float32)
    
    # Create PIL image from the numpy array
    pil_image = Image.fromarray(mask_float32)
    
    # Resize the image
    resized_pil_image = pil_image.resize((width, height), Image.BILINEAR)
    
    # Convert back to numpy array
    resized_mask = np.array(resized_pil_image)
    
    return resized_mask

def approximate_to_max_point_cnt(poly, epsilon=0.08, max_points = 400):
    while(True):
        approximations = cv2.approxPolyDP(poly, epsilon, False)
        
        if len(approximations) > max_points:
            epsilon += 0.05 
        else:
            break
    approximations = np.reshape(approximations, (-1, 2))
    return approximations

def convert_local_polygons_to_global(outputs, list_of_boxes):
    for box_id, box in enumerate(list_of_boxes):
        x, y = box[:2]
        outputs[box_id] = [(point[0] + x, point[1] + y) for point in outputs[box_id]]
        
        
class FishialPolygon:
    def __init__(self, points):
        """
        Initializes the Polygon.
        
        Args:
            points (list): List of tuples representing the polygon points.
        """
        self.points = points
        self.width, self.height = self.calculate_dimensions()
    
    def calculate_dimensions(self):
        """
        Calculates the width and height of the polygon's bounding box.
        
        Returns:
            tuple: Width and height of the polygon's bounding box.
        """
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width, height

    def get_area(self):
        """
        Calculates the area of the polygon using the Shoelace formula.
        
        Returns:
            float: Area of the polygon.
        """
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        return 0.5 * abs(sum(x[i] * y[i+1] - y[i] * x[i+1] for i in range(-1, len(self.points)-1)))
    
    def get_centroid(self):
        """
        Calculates the centroid of the polygon.
        
        Returns:
            tuple: Coordinates of the centroid (x, y).
        """
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        area = self.get_area()
        cx = sum((x[i] + x[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i]) for i in range(-1, len(self.points)-1)) / (6 * area)
        cy = sum((y[i] + y[i+1]) * (x[i] * y[i+1] - x[i+1] * y[i]) for i in range(-1, len(self.points)-1)) / (6 * area)
        return (cx, cy)
    
    def draw_polygon(self, image, color=(0, 255, 0), thickness=2):
        """
        Draws the polygon on the image.
        
        Args:
            image (numpy.ndarray): Image on which the polygon will be drawn.
            color (tuple): Color of the polygon in (B, G, R) format.
            thickness (int): Thickness of the polygon lines.
        """
        pts = np.array(self.points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    
    def get_mask(self):
        """
        Creates a mask for the polygon.
        
        Args:
            image_shape (tuple): Shape of the image (height, width).
        
        Returns:
            numpy.ndarray: Mask of the polygon.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array(self.points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def mask_polygon(self, image):
        """
        Applies a mask to the polygon on the image.
        
        Args:
            image (numpy.ndarray): Image on which the mask will be applied.
        
        Returns:
            numpy.ndarray: Image with the polygon area masked.
        """
        mask = np.zeros_like(image)
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def move_to(self, x, y):
        """
        Moves the polygon to a new point (x, y).
        
        Args:
            x (float): The x-coordinate of the new point.
            y (float): The y-coordinate of the new point.
        """
        self.points = [(px + x, py + y) for px, py in self.points]
        self.width, self.height = self.calculate_dimensions()
    
    def to_points_dict(self):
        """
        Converts the polygon points to a dictionary format.
        
        Returns:
            dict: Dictionary with keys as 'x1', 'y1', 'x2', 'y2', etc.
        """
        points_dict = {}
        for i, (x, y) in enumerate(self.points, start=1):
            points_dict[f'x{i}'] = x
            points_dict[f'y{i}'] = y
        return points_dict
    
    def __repr__(self):
        return f"Polygon(points={self.points}, width={self.width}, height={self.height})"
    
    def to_dict(self):
        """
        Converts the object to a dictionary.
        
        Returns:
            dict: Dictionary with the key 'points'.
        """
        return {
            'points': self.points,
            'width': self.width,
            'height': self.height,
            'area': self.get_area(),
            'centroid': self.get_centroid()
        }