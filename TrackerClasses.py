import pupil_apriltags
from abc import ABC, abstractmethod


class Tracker(ABC):
    def __init__(self) -> None:
        self.tolerance
        self.active
        self.appearances
        self.threshold

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def compare():
        raise NotImplementedError


class AprilTracker(Tracker):
    MAX_APPEARANCES = 5

    def __init__(self, detection) -> None:

        self.detection: pupil_apriltags.Detection = detection
        self.tolerance = 50
        self.active = False
        self.appearances = 1
        self.threshold = 3

        super().__init__()

    def update(self, found, new_tag=None):

        if found:
            self.detection = new_tag
            if self.appearances < self.MAX_APPEARANCES:
                self.appearances += 1

            if self.appearances >= self.threshold:
                self.active = True

            if self.appearances < self.threshold:
                self.active = False

        if not found:
            self.appearances -= 1

    def compare(self, new_tag: pupil_apriltags.Detection):
        # Check if the tag has the same ID
        if self.detection.tag_id == new_tag.tag_id:
            return True

        else:
            return False
