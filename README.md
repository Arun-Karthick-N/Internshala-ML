**Internshala ML Assignment: Player Tracking in Sports Video**

This repository contains the source code and documentation for tracking players in a 15-second sports video using the Ultralytics YOLOv11 model and BOTSORT tracker. The script detects players, assigns unique IDs, and annotates the output video with green bounding boxes and red IDs above players' heads.

**Repository Structure**





player_tracking.py: Main script for player detection and tracking.



README.md: Instructions for setup and execution (this file).



REPORT.md: Detailed report on approach, methodology, and challenges (or REPORT.odt.tex for LaTeX version).



15sec_input_720p.mp4: Input video (not included; place in repository root).



player_model.pt: YOLO model file (not included; place in repository root).



output_tracked_video.mp4: Output video (generated after running the script).

**Setup Instructions**

Prerequisites





Python Version: Python 3.8 or higher (tested with Python 3.12).



Operating System: Windows, Linux, or macOS.



Input Files:





15sec_input_720p.mp4: 15-second sports video in 720p resolution.



player_model.pt: Pre-trained YOLOv11 model for player detection.



Place both files in the repository root directory.

Dependencies

Install the required Python packages using pip:

pip install -U ultralytics opencv-python numpy lap





ultralytics: For YOLOv11 model and BOTSORT tracker (version 8.3.x or higher).



opencv-python: For video processing and rendering.



numpy: For numerical operations.



lap: For linear assignment in tracking.

Update pip if needed:

python -m pip install --upgrade pip

**Environment Setup**




Clone or download this repository:

git clone https://github.com/Arun-Karthick-N/Internshala-ML.git
cd Internshala-ML



Place 15sec_input_720p.mp4 and player_model.pt in the repository root.



Install dependencies as shown above.

**Running the Code**





Navigate to the repository directory:

cd path/to/Internshala-ML



Execute the script:

python player_tracking.py



Output:





The script generates output_tracked_video.mp4 with annotated players (green bounding boxes, red IDs).



Console displays debug output for the first 10 frames, showing detected classes, confidences, and number of tracks.

**Expected Output**





Video: output_tracked_video.mp4 contains the input video with players annotated by green bounding boxes and red IDs above their heads.



Console:

Frame 0:
Detected classes: [2.0, 2.0, ..., 3.0]
Confidences: [0.92468, 0.92299, ..., 0.85927]
Number of player detections: 16
Number of tracks: 16
Output video saved as output_tracked_video.mp4

**Troubleshooting**




No Annotations in Output Video:





Check console output for Number of player detections and Number of tracks. If tracks are 0, edit player_tracking.py and adjust:

match_thresh=0.5, track_buffer=150



Add a test annotation to verify rendering:

cv2.putText(frame, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

Place before out.write(frame) and check if "Test" appears.



Ensure 15sec_input_720p.mp4 and player_model.pt are in the correct directory.



Dependency Errors:

pip install --force-reinstall ultralytics opencv-python numpy lap



Video Corruption:





Edit player_tracking.py:

fourcc = cv2.VideoWriter_fourcc(*'XVID')



Performance:





For faster processing, use a smaller YOLO model (e.g., yolo11n.pt) or reduce video resolution.

**Notes**





The script assumes players are class ID 2. Verify this with your player_model.pt debug output.



For enhanced accuracy, enable ReID by setting with_reid=True and model="yolo11n-cls.pt" after updating Ultralytics and downloading the ReID model from Ultralytics.



The report is provided as REPORT.md (or REPORT.odt.tex for LaTeX). To convert LaTeX to ODF:

pandoc -f latex -t odt -o REPORT.odt REPORT.odt.tex

**Contact**

For issues or questions, create a GitHub issue or contact the repository owner.
