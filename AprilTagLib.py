import cv2
import pyapriltags as pt
import numpy as np
import time
import multiprocessing
from typing import List
from trackerClasses import AprilTracker


class AprilTagDetector():
    def __init__(self, cam: cv2.VideoCapture, shared_list) -> None:

        self._cam: cv2.VideoCapture = cam
        self._at_detector = pt.Detector(
            families="tag16h5",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0)

        self._tracked_tags: List[pt.Detection] = []
        self._tracked_ids: List[int] = []
        self._active_tags: List[pt.Detection] = []

        self._shared_list = shared_list

    
    def start(self) -> None:
        self._process.start()

    def run(self) -> None:
        while True:
            now = time.perf_counter()
            ret, frame = self._cam.read()

            processed_image = self._preprocess_image(frame)
            unfiltered_tags = self._at_detector.detect(processed_image, estimate_tag_pose=True, camera_params=(
                [1.37882055e+03, 1.37985939e+03, 9.18678547e+02, 5.50086195e+02]), tag_size=(0.153))

            new_tags = list(filter(self._filter_tag, unfiltered_tags))

            for tracked_tag in self._tracked_tags:
                found = False
                for current_tag in new_tags:
                    if tracked_tag.compare(current_tag):
                        tracked_tag.update(True, current_tag)
                        found = True
                        break

                if not found:
                    tracked_tag.update(False)

            for new_tag in new_tags:
                if new_tag.tag_id not in self._tracked_ids:
                    self._tracked_tags.append(AprilTracker(new_tag))
                    self._tracked_ids.append(new_tag.tag_id)

            for index, tracked_tag in enumerate(self._tracked_tags):
                if tracked_tag.appearances < 1:
                    self._tracked_tags.pop(index)
                    self._tracked_ids.remove(tracked_tag.detection.tag_id)
                    print(f"TAG {tracked_tag.detection.tag_id} KILLED")

            # for tracked_tag in self.tracked_tags:
            #     if tracked_tag.active:
            #         print(time.perf_counter(), tracked_tag.detection.tag_id, tracked_tag.appearances)
            self._active_tags = list(filter(lambda tag: tag.active, self._tracked_tags))
            self._active_tags = list(map(lambda tag: tag.detection, self._active_tags))

            self._shared_list.append(self._active_tags) 

    def get_active_tags(self) -> List[pt.Detection]:
        return self._active_tags

    def _verify_affine_homography(self, tag):
        AFFINE_THRESHOLD = (0.1, 0.1, 0.1)

        # Get homography
        homography = tag.homography

        # Get degrees of freedom that should be 0, 0, 1 (Affine transformation)
        dof1, dof2, dof3 = homography[2]

        # Verify if the homography is affine
        return abs(dof1) < AFFINE_THRESHOLD[0] and abs(dof2) < AFFINE_THRESHOLD[1] and abs(dof3) - 1 < AFFINE_THRESHOLD[2]

    def _verify_area(self, tag):

        # Get tag corners
        corners = tag.corners

        # Get tag width and height
        # Calculate the distance between the first and second corner
        width = np.linalg.norm(corners[0] - corners[1])
        # Calculate the distance between the second and third corner
        height = np.linalg.norm(corners[1] - corners[2])

        # Get tag area
        area = width * height

        # Verify if the tag area is in the range
        return area > 50

    def _filter_tag(self, tag):
        # Add filters
        return (self._verify_affine_homography(tag) and
                self._verify_area(tag) and
                tag.tag_id <= 8)

    def _preprocess_image(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blured = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur
        equalized = cv2.equalizeHist(blured)  # Histogram equalization

        return equalized
