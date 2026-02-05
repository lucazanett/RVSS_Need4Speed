import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from machinevisiontoolbox import Image


@dataclass
class DetectorOutput:
    seen: bool
    area: float
    bbox: Optional[Tuple[int, int, int, int]]
    centroid: Optional[Tuple[float, float]]


class StopSignDetector:

    def __init__(
        self,
       # I used HSV since the stop sign is Red.

        hsv_low1=(0, 35, 35),
        hsv_high1=(8, 255, 255),
        hsv_low2=(170, 35, 35),
        hsv_high2=(179, 255, 255),

        min_area=60,
        max_area=8000,

        roi: Optional[Tuple[float, float, float, float]] = (0.05, 0.30, 0.95, 0.98),
        #roi: Optional[Tuple[float, float, float, float]] = (0.10, 0.40, 0.90, 0.98),

        morph_kernel=3,

        aspect_ratio_range=(0.2, 4.0),
        min_fill_ratio=0.10,

        reject_orange_hue_range=(9, 25),

        min_red_pixels_in_bbox=250,
    ):
        self.hsv_low1 = np.array(hsv_low1, dtype=np.uint8)
        self.hsv_high1 = np.array(hsv_high1, dtype=np.uint8)
        self.hsv_low2 = np.array(hsv_low2, dtype=np.uint8)
        self.hsv_high2 = np.array(hsv_high2, dtype=np.uint8)

        self.min_area = float(min_area)
        self.max_area = float(max_area)

        self.roi = roi
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

        self.aspect_ratio_range = aspect_ratio_range
        self.min_fill_ratio = float(min_fill_ratio)

        self.reject_orange_hue_range = reject_orange_hue_range
        self.min_red_pixels_in_bbox = int(min_red_pixels_in_bbox)

        self.last_mask: Optional[np.ndarray] = None

    def _apply_roi(self, rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        if self.roi is None:
            return rgb, (0, 0)
        h, w = rgb.shape[:2]
        x0f, y0f, x1f, y1f = self.roi
        x0, y0 = int(x0f * w), int(y0f * h)
        x1, y1 = int(x1f * w), int(y1f * h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        return rgb[y0:y1, x0:x1], (x0, y0)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    @staticmethod
    def _hue_stats_in_bbox(hsv: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[float], int]:

        x, y, w, h = bbox
        roi_hsv = hsv[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w]
        idx = roi_mask > 0
        red_px = int(idx.sum())
        if red_px == 0:
            return None, 0
        hue_vals = roi_hsv[:, :, 0][idx]
        return float(np.mean(hue_vals)), red_px


    def image_show(self, frame_rgb, det: DetectorOutput):

        if frame_rgb is None:
            return
        img = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)

        if det.bbox is not None:
            x, y, w, h = det.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if det.centroid is not None:
            cx, cy = int(det.centroid[0]), int(det.centroid[1])
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        txt = f"seen={det.seen} area={det.area:.1f}"
        cv2.putText(img, txt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ROBOT VIEW", img)

        if self.last_mask is not None:
            cv2.imshow("MASK", self.last_mask)

        cv2.waitKey(1)


    def detect(self, frame_rgb: np.ndarray) -> DetectorOutput:
        rgb_roi, (ox, oy) = self._apply_roi(frame_rgb)

        hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_low1, self.hsv_high1)
        mask2 = cv2.inRange(hsv, self.hsv_low2, self.hsv_high2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        self.last_mask = mask

        try:
            blobs = Image(mask).blobs()
        except Exception:
            return DetectorOutput(seen=False, area=0.0, bbox=None, centroid=None)

        best_area = 0.0
        best_centroid = None
        best_bbox_local = None

        for b in blobs:
            a = float(getattr(b, "area", 0.0))
            if a < self.min_area or a > self.max_area:
                continue

            cxy = None
            try:
                c = b.centroid
                cxy = (float(c[0]), float(c[1]))
            except Exception:
                cxy = None

            bbox_local = None
            try:
                bb = b.bbox
                if bb is not None and len(bb) == 4:
                    u0, u1, v0, v1 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    x, y, w, h = u0, v0, (u1 - u0), (v1 - v0)
                    if w > 0 and h > 0:
                        bbox_local = (x, y, w, h)
            except Exception:
                bbox_local = None

            if bbox_local is None:
                continue

            mean_h, red_px = self._hue_stats_in_bbox(hsv, mask, bbox_local)

            if red_px < self.min_red_pixels_in_bbox:
                continue

            lo, hi = self.reject_orange_hue_range
            if mean_h is not None and lo <= mean_h <= hi:
                continue

            _, _, w, h = bbox_local
            ar = (w / h) if h > 0 else 999.0
            ar_min, ar_max = self.aspect_ratio_range
            if not (ar_min <= ar <= ar_max):
                continue

            bbox_area = float(w * h)
            if bbox_area > 0:
                fill = a / bbox_area
                if fill < self.min_fill_ratio:
                    continue

            if a > best_area:
                best_area = a
                best_centroid = cxy
                best_bbox_local = bbox_local

        if best_bbox_local is None:
            return DetectorOutput(seen=False, area=0.0, bbox=None, centroid=None)

        centroid = None
        if best_centroid is not None:
            centroid = (best_centroid[0] + ox, best_centroid[1] + oy)

        x, y, w, h = best_bbox_local
        bbox = (x + ox, y + oy, w, h)


        det_out = DetectorOutput(seen=True, area=best_area, bbox=bbox, centroid=centroid)
        self.image_show(frame_rgb, det_out)

        return det_out


class StopSignController:

    def __init__(
        self,
        area_stop_threshold: float,
        confirm_k: int = 3,
        window_n: int = 5,
        stop_time: float = 1.0,
        cooldown_time: float = 1.5,
    ):
        self.area_stop_threshold = float(area_stop_threshold)
        self.confirm_k = int(confirm_k)
        self.window_n = int(window_n)
        self.stop_time = float(stop_time)
        self.cooldown_time = float(cooldown_time)

        self.history = deque(maxlen=self.window_n)
        self.state = "DRIVING"
        self.state_t0 = time.time()

    def update(self, det: DetectorOutput) -> Dict[str, Any]:
        now = time.time()

        close = bool(det.seen and det.area >= self.area_stop_threshold)
        self.history.append(1 if close else 0)
        confirmed = (sum(self.history) >= self.confirm_k)

        if self.state == "DRIVING":
            if confirmed:
                self.state = "STOPPED"
                self.state_t0 = now
                return {"override": True, "speed_cmd": 0.0, "state": self.state}
            return {"override": False, "speed_cmd": None, "state": self.state}

        if self.state == "STOPPED":
            if (now - self.state_t0) >= self.stop_time:
                self.state = "COOLDOWN"
                self.state_t0 = now
                self.history.clear()
            return {"override": True, "speed_cmd": 0.0, "state": self.state}

        if self.state == "COOLDOWN":
            if (now - self.state_t0) >= self.cooldown_time:
                self.state = "DRIVING"
                self.state_t0 = now
            return {"override": False, "speed_cmd": None, "state": self.state}

        self.state = "DRIVING"
        return {"override": False, "speed_cmd": None, "state": self.state}
