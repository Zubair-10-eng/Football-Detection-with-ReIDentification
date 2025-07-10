#Import libraries
import cv2

#Read a video function
def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

#save video function
def save_video(output_videos_frames, output_video_path, fps=24):
    if not output_videos_frames:
        print("No frames to save.")
        return

    height, width, _ = output_videos_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # ✅ Use 4 characters
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_videos_frames:
        out.write(frame)

    out.release()   