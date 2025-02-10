#!/usr/bin/env python3
import math
import numpy as np
import cv2


class DetectorMultiFramework:
    
    def __init__(self, framework: str, path: str, minconf: float):
        """
        for convenience and for handling execution on different platforms, this class bundles the execution of yolo inference with the ultralytics api and the tensorrt framework.
        NOTE: there seems to be weird behaviour regarding the color channel order. when getting wrong results, try switching the input from rgb to bgr or vice versa! 

        Args 
        ----
        - `framework`: specify wheter ultralytics api or tensorrt framework should be used to run the yolo model
        - `path`: path to either the yolo weights.pt or the compiled model.engine (platform specific!)
        - `minconf`: minimum confidence threshold for output detection points. range [0, 1]
        """
        
        if framework not in ["ultralytics", "tensorrt"]:
            raise ValueError(f"please specify a valid framework from [ultralytics, tensorrt]! (got {framework=})")
        
        self.framework = framework
        self.minconf   = minconf
        self.Net       = None
        
        if self.framework == "ultralytics": # initialize the detector by using ultralytics api
            from ultralytics import YOLO
            self.Net = YOLO(path)
            
        if self.framework == "tensorrt":    # initialize the detector by using the tensorrt framework 
            from tensorrt_inference import TensorRTInference
            self.Net = TensorRTInference(path)
      
    def _preprocess(self, image: np.ndarray, input_size: tuple=(640, 640), flip_colors: bool=True):
        """
        this replaces the image pre-processing that is normally handled by the ultralytics api. This mimics the source code from the ultralytics repo as much as possible, but obtaining the exact same results is unlikely. YoloV8 normally accepts dynamic input sizes. the compiled engine is usually fixed to a certain input shape (usually [640, 640]). For this reason, the input image is letterboxed (although scaling should be chosen such that the actually image content sizes are still divisible by 32), normalized and transposed. 

        Args
        ----
        - `image`: the original input image (arbitrary size)
        - `input_size`: the desired input size for the yolo model [H, W] (usually [640, 640])
        - `flip_colors`: flag to control whether colors are flipped RGB <-> BGR

        Returns
        -------
        - `image`: the processed image
        - `prep_params`: [scale_h, scale_w, padd_h, padd_w] for reconstructing the bbox cordinates in post-processing
        """
         
        # flip color channels ------------------------------------------------------------------------------------------
        if flip_colors is True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        # resizing and padding -----------------------------------------------------------------------------------------
        h_inp, w_inp, _ = image.shape
        h_exp, w_exp    = input_size
        
        # initially rescale the image naively so that the larger dimension fits into input_size while keeping ratio
        scale_init   = min(h_exp/h_inp, w_exp/w_inp) # the more extreme one is determinant
        h_new, w_new = int(h_inp * scale_init), int(w_inp * scale_init)
        
        # now make sure all new dimensions are divisible by 32. round to nearest multiple of 32, but limit to at least 32 and maximum the largest multiple of 32 that fits in expected input size
        h_new = np.clip(round(h_new/32), a_min=1, a_max=math.floor(h_exp/32)) * 32
        w_new = np.clip(round(w_new/32), a_min=1, a_max=math.floor(w_exp/32)) * 32
        
        # resize image
        image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        
        # compute padding
        h_padd = (h_exp - h_new) // 2
        w_padd = (w_exp - w_new) // 2
        
        # apply padding 
        image = cv2.copyMakeBorder(image, h_padd, h_padd, w_padd, w_padd, cv2.BORDER_CONSTANT, value=(144, )*3)
        
        # normalize values to [0, 1] -----------------------------------------------------------------------------------
        image = image.astype(np.float32) / 255.0
        
        # transpose [H, C, W] -> [C, W, H] -----------------------------------------------------------------------------
        image = np.transpose(image, (2, 0, 1))
        
        # add dummy batch dimension -> [1, C, H, W] --------------------------------------------------------------------
        image = image[None, :, :, :]
        
        return image, [h_new/h_inp, w_new/w_inp, h_padd, w_padd]    
      
    def _postprocess(self, pred: list, prep_params: list, orig_size: tuple, minconf: float, maxiou: float=0.45):
        """
        this replaces the post-processing that is usually handled by the ultralytics api: filtering the bboxes by confidence, applying non-maximum suppression and finally rescaling and translating the bbox coordinates to match the original image size. 

        Args
        ----
        - `pred`: the raw model output from the tensorrt inference (usually [1, 5, 8400] for one class detection)
        - `prep_params`: [scale_h, scale_w, padd_h, padd_w] used in pre-processing
        - `orig_size`: the original image size [H, W]
        - `minconf`: minimum confidence threshold for output detection points. range [0, 1]
        - `maxiou`: for NMS, IOU threshold. use 0.45 to match the ultralytics implementation.

        Returns
        -------
        - `boxes`: all the filtered bounding boxes of detections
        """
        
        # remove dummy list and batch dimension of model output -> [8400, 5] -------------------------------------------
        boxes = pred[0][0, :, :].T 
        
        # directly kick out most of the low confidence boxes to speed up NMS filtering later on ------------------------
        mask  = boxes[:, 4] > minconf
        boxes = boxes[mask]
        
        if boxes.shape[0] == 0:
            return np.empty((0, 4))
        
        # conver boxes from xywh format to xyxy (mincorner, maxcorner) -------------------------------------------------
        X, Y, W, H, C  = boxes.T
        X1             = X - W/2
        Y1             = Y - H/2
        X2             = X + W/2
        Y2             = Y + H/2
        boxes          = np.stack([X1, Y1, X2, Y2], axis=1)

        # apply non-maximum suppression --------------------------------------------------------------------------------
        # NOTE: depending in the version of opencv, either a list of lists or an ndarray is needed as input to bboxes!
        indices = cv2.dnn.NMSBoxes(
            bboxes          = boxes.tolist(), 
            scores          = C.tolist(), 
            score_threshold = minconf, 
            nms_threshold   = maxiou
        )
        if len(indices) == 0:
            # NOTE: early exit because NMSBoxes will return an empty tuple when no boxes are valid, instead of ndarray!
            return np.empty((0, 4))
        
        boxes = [boxes[i, :] for i in indices.flatten()]
        boxes = np.array(boxes)
        
        # rescale boxes to match the original image shape --------------------------------------------------------------
        scale_h = prep_params[0]
        scale_w = prep_params[1]
        h_padd  = prep_params[2]
        w_padd  = prep_params[3]
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - w_padd) / scale_w # adjust x coordinates 
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - h_padd) / scale_h # adjust y coordinates 
        
        # remove invalid boxes from the padding region -----------------------------------------------------------------
        h_orig, w_orig = orig_size
        
        valid_mask = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= w_orig) & (boxes[:, 3] <= h_orig)
        boxes      = boxes[valid_mask]
        
        return boxes
   
    def detect(self, input: np.ndarray):
        """
        executes inference on the input image with the framework that was specified when initializing.

        Args
        ----
        - `input`: the input image [H, W, C]. NOTE: when getting weird results, try switching RGB <-> BGR!

        Returns
        -------
        - `points`: an ndarray of shape [n, 2] with the x and y coordinates of all detection points
        """
        
        if self.framework == "ultralytics":
            result = self.Net(input, conf=self.minconf, verbose=False)
            boxes  = result[0].boxes.xyxy.detach().cpu().numpy() # [n, 4]

            pointsx = (boxes[:, 0] + boxes[:, 2]) / 2
            pointsy = (boxes[:, 1] + boxes[:, 3]) / 2
            points  = np.stack([pointsx, pointsy], axis=1)
            
            return points
            
        if self.framework == "tensorrt":
            orig_size = input.shape[0:2] # original size (h, w)
            
            input_prep, prep_params = self._preprocess(input, input_size=(640, 640), flip_colors=True)
            result                  = self.Net.infer(input_prep)
            boxes                   = self._postprocess(result, prep_params, orig_size=orig_size, minconf=self.minconf)
            
            pointsx = (boxes[:, 0] + boxes[:, 2]) / 2
            pointsy = (boxes[:, 1] + boxes[:, 3]) / 2
            points  = np.stack([pointsx, pointsy], axis=1)
        
        return points
