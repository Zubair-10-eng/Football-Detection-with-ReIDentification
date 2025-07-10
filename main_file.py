

# Import Required Libraries
import os
from utils import read_video, save_video
from trackers import Tracker

def main():
   
    # Read video (use raw string to avoid path issues)
    video_frames = read_video(r"Input videos\15sec_input_720p.mp4")

    # Initialize Tracker with YOLO model
    tracker = Tracker("custom_model/last.pt")

    # Get Object Tracks (players, referees, ball)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path='tracker_stubs/player_detection.pkl'
    )

    # Draw Annotations on Video Frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Output Video
    save_video(output_video_frames, 'output_videos/output.avi')


if __name__ == "__main__":
    main()
