from ultralytics import YOLO
import cv2
import math

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("yolov8l.pt")
names = model.model.names

cap = cv2.VideoCapture("Small.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h))

class DistanceCalculation:
    def __init__(self):
        super().__init__()

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False
        self.line_color = (255, 255, 0)
        self.centroid_color = (255, 0, 255)

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.line_thickness = 2
        self.trk_ids = None

        # Distance calculation information
        self.centroids = []
        self.pixel_per_meter = 10

        # Mouse event
        self.selected_point = None  # Store the selected point

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        names,
        pixels_per_meter=10,
        view_img=False,
        line_thickness=2,
        line_color=(255, 255, 0),
        centroid_color=(255, 0, 255),
    ):
        self.names = names
        self.pixel_per_meter = pixels_per_meter
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.centroid_color = centroid_color

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_and_annotate_distances(self, centroids):
        num_objects = len(centroids)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                distance_m, distance_mm = self.calculate_distance(centroids[i], centroids[j])
                midpoint = ((centroids[i][0] + centroids[j][0]) // 2, (centroids[i][1] + centroids[j][1]) // 2)
                self.annotator.box_label([midpoint[0], midpoint[1], midpoint[0], midpoint[1]], label=f'{distance_m:.2f}m', color=self.line_color)

    def mouse_event_for_distance(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)  # Update the selected point

    def extract_tracks(self, tracks):
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def calculate_centroid(self, box):
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        return pixel_distance / self.pixel_per_meter, (pixel_distance / self.pixel_per_meter) * 1000

    def start_process(self, im0, tracks):
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=self.line_thickness)

        for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])
            centroid = self.calculate_centroid(box)
            if self.selected_point:
                distance_m, distance_mm = self.calculate_distance(self.selected_point, centroid)
                self.annotator.text(centroid, f'{distance_m:.2f}m')

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        cv2.namedWindow("Ultralytics Distance Estimation")
        cv2.setMouseCallback("Ultralytics Distance Estimation", self.mouse_event_for_distance)
        cv2.imshow("Ultralytics Distance Estimation", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

# Init distance-calculation obj
dist_obj = DistanceCalculation()
dist_obj.set_args(names=names, 
                  view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = dist_obj.start_process(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
