from ultralytics import YOLO
import cv2
import colors
import threading
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=150, n_init=2, embedder='mobilenet', max_iou_distance=0.6)

thread_results = {}
field_color = None
groups_color_filters = None
rect_color = None

#calculate IOU
def calculate_iou(boxA, boxB):
    # boxA and boxB are [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Added epsilon to avoid div with 0
    return iou

def get_color(i, cropped_image):
    global field_color
    result = colors.detect_color(cropped_image, field_color)
    thread_results[i] = result if result else []

def detect_colors(image, data, ratio, class_names):
    global thread_results, field_color, groups_color_filters
    objects = {}

    boxes = data.boxes.xyxy
    scores = data.boxes.conf
    categories = data.boxes.cls

    threads = []
    thread_results = {}

    FOOTBALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    all_tracked_ids = [FOOTBALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID]

    CONF_THRESH = 0.4
    MIN_AREA = 300
    MAX_AREA = 50000

    object_keys = []

    for i, (score, label, box) in enumerate(zip(scores, categories, boxes)):
        current_label_id = int(label.item())
        current_object_name = class_names[current_label_id]

        # Check if the current object is a ball
        is_ball = (current_label_id == FOOTBALL_ID)

        # Apply general filtering for non-ball objects, or if it's a ball, ensure it's a tracked ID
        if current_label_id in all_tracked_ids:
            # For non-ball objects, apply confidence and area filters
            if not is_ball:
                if score.item() < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = box.tolist()
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_AREA or area > MAX_AREA:
                    continue
            # For ball objects, no confidence or area filtering is applied here,
            # it just needs to be one of the all_tracked_ids (which it is)

            object_keys.append(i)
            objects[i] = {
                "object": current_object_name,
                "score": round(score.item(), 3),
                "location": box.tolist(), # relative to frame_for_prediction (640x360)
                "track_id": None
            }

            if current_object_name == 'ball':
                objects[i]["bottom_color"] = None
                objects[i]["detected_jersey_colors"] = []
                objects[i]["team"] = "None"
                continue # Skip color detection for ball

    for i, (score, label, box) in enumerate(zip(scores, categories, boxes)):
        current_label_id = int(label.item())
        if current_label_id in all_tracked_ids and score.item() >= CONF_THRESH:
            x1, y1, x2, y2 = box.tolist()
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            object_keys.append(i)
            current_object_name = class_names[current_label_id]
            objects[i] = {
                "object": current_object_name,
                "score": round(score.item(), 3),
                "location": [x1, y1, x2, y2], #relative to frame_for_prediction (640x360)
                "track_id": None
            }

            if current_object_name == 'ball':
                objects[i]["bottom_color"] = None
                objects[i]["detected_jersey_colors"] = []
                objects[i]["team"] = "None"
                continue

            #Top Half of Bounding Box Cropping (for Jersey)

            original_height = y2 - y1
            original_width = x2 - x1

            y2_jersey = y1 + int(original_height * 0.5)
            x1_jersey = x1 + int(original_width * 0.15)
            x2_jersey = x2 - int(original_width * 0.15)

            x1_jersey = np.clip(x1_jersey, 0, image.shape[1])
            x2_jersey = np.clip(x2_jersey, 0, image.shape[1])
            y2_jersey = np.clip(y2_jersey, 0, image.shape[0])

            cropped_jersey = image[int(y1):int(y2_jersey), int(x1_jersey):int(x2_jersey)]
            if cropped_jersey.size > 0:
                ch_j, cw_j, _ = cropped_jersey.shape
                if cw_j > 0 and ch_j > 0:
                    resized_jersey = cv2.resize(cropped_jersey, (max(1, int(cw_j / ratio)), max(1, int(ch_j / ratio))))
                    threads.append(threading.Thread(target=get_color, args=(i, resized_jersey)))

            #Bottom Half of Bounding Box Cropping (for Shorts)

            y1_b = y1 + int(original_height * 0.5)
            y2_b = min(y1 + int(original_height * 0.73), y2)
            x1_b = x1 + int(original_width * 0.1)
            x2_b = x2 - int(original_width * 0.1)

            x1_b = np.clip(x1_b, 0, image.shape[1])
            x2_b = np.clip(x2_b, 0, image.shape[1])
            y2_b = np.clip(y2_b, 0, image.shape[0])

            cropped_bottom = image[int(y1_b):int(y2_b), int(x1_b):int(x2_b)]
            objects[i]["bottom_color"] = None
            if cropped_bottom.size > 0:
                ch_b, cw_b, _ = cropped_bottom.shape
                if cw_b > 0 and ch_b > 0:
                    resized_bottom = cv2.resize(cropped_bottom, (max(1, int(cw_b / ratio)), max(1, int(ch_b / ratio))))
                    bottom_colors = colors.detect_color(resized_bottom, field_color)
                    if bottom_colors:
                        objects[i]["bottom_color"] = bottom_colors[0]

    [t.start() for t in threads]
    [t.join() for t in threads]

    for i in objects:
        objects[i]["detected_jersey_colors"] = thread_results.get(i, [])

    # Prepare Deep SORT input
    formatted_detections = []
    for obj_key in object_keys: # Iterate through keys of detected objects
        x1, y1, x2, y2 = objects[obj_key]["location"]
        w = x2 - x1
        h = y2 - y1
        score = objects[obj_key]["score"]
        label = objects[obj_key]["object"]
        # DeepSORT expects [x, y, w, h] in the coordinate system of the frame it's processing (frame_for_prediction here)
        formatted_detections.append(([x1, y1, w, h], score, label))


    tracks = tracker.update_tracks(formatted_detections, frame=image) # image here is frame_for_prediction (640x360)

    # Assign track_ids to objects based on IOU overlap with DeepSORT's tracked boxes
    assigned_track_ids = set() # To ensure each track_id is assigned only once

    for track in tracks:
        if not track.is_confirmed() or track.track_id in assigned_track_ids:
            continue

        current_track_id = track.track_id
        # Get the predicted bounding box for the track, in the coordinates of the input frame (640x360)
        tx1, ty1, tw, th = track.to_tlwh()
        track_bbox_ltrb = [tx1, ty1, tx1 + tw, ty1 + th] # Convert to x1,y1,x2,y2 for IOU

        best_match_obj_key = None
        max_iou = 0.0

        for obj_key in object_keys: # Iterate through current YOLO detections
            yolo_bbox = objects[obj_key]["location"] # YOLO output [x1, y1, x2, y2] is already in 640x360 frame coords

            iou = calculate_iou(track_bbox_ltrb, yolo_bbox)

            if iou > max_iou and iou > 0.1: # Threshold IOU for a valid match, e.g., 0.1
                max_iou = iou
                best_match_obj_key = obj_key

        if best_match_obj_key is not None:
            objects[best_match_obj_key]["track_id"] = current_track_id
            assigned_track_ids.add(current_track_id) # Mark as assigned

    all_primary_jersey_colors = [obj["detected_jersey_colors"][0] for obj in objects.values() if obj["detected_jersey_colors"]]
    groups = colors.get_ranged_groups(all_primary_jersey_colors, groups_color_filters)

    for key, value in objects.items():
        if value["object"] == 'ball':
            continue

        assigned_team = 'Unknown'
        primary = value["detected_jersey_colors"][0] if value["detected_jersey_colors"] else None

        if value["object"] == 'referee' and primary:
            for r in groups_color_filters.get('R', ()):
                if colors.is_color_in_range(primary, *r):
                    assigned_team = 'None'; break

        if assigned_team == 'Unknown' and value.get("bottom_color"):
            for b in ((0, 360, 0.0, 0.2, 0.0, 0.2),):
                if colors.is_color_in_range(value["bottom_color"], *b):
                    for t1 in groups_color_filters.get('T1', ()):
                        if primary and colors.is_color_in_range(primary, *t1):
                            assigned_team = 'T1'; break
            for w in ((0, 360, 0.0, 0.1, 0.8, 1.0),):
                if colors.is_color_in_range(value["bottom_color"], *w):
                    for t2 in groups_color_filters.get('T2', ()):
                        if primary and colors.is_color_in_range(primary, *t2):
                            assigned_team = 'T2'; break

        if assigned_team == 'Unknown' and primary:
            for k, v in groups.items():
                if primary in v:
                    assigned_team = k; break

        value["team"] = assigned_team

    return image, objects



def draw_boxes(image, objects, ratio):
    global rect_color # Declare global for access

    # List to store information about the drawn text boxes for collision detection
    # Format: [x1, y1, x2, y2, score]
    drawn_text_boxes = []

    # Sort objects by score in descending order to prioritize higher confidence detections
    sorted_objects = sorted(objects.items(), key=lambda item: item[1]['score'], reverse=True)


    for key, value in sorted_objects:
        x1, y1, x2, y2 = value["location"]
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = int(x1 * ratio), int(y1 * ratio), int(x2 * ratio), int(y2 * ratio)

        team_color = (128, 128, 128) # Default to grey

        if value["object"] == 'referee':
            team_color = (0, 255, 255) # Yellow (B, G, R)

        elif value["object"] == 'ball': #Specific color for football
            team_color = (255, 255, 255) # White

        else:
            team_color = rect_color.get(value["team"], (128, 128, 128)) # Use team color or grey

        cv2.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), team_color, 2)

        x_center = (x1_scaled + x2_scaled) / 2
        y_center = (y1_scaled + y2_scaled) / 2

        # Prepare text lines dynamically based on object type
        text_lines = []
        text_lines.append(f"id: {value.get('track_id', 'N/A')}")      #Add ID line first
        text_lines.append(f"object: {value['object']}")

        if value["object"] == 'ball': # No team, jersey, or bottom color for football
            pass
        else:
            text_lines.append(f"team: {value['team']}")

        # Calculate the bounding box for the entire text block
        max_text_width = 0
        total_text_height = 0
        line_height = 15 # Based on your drawing logic

        for line in text_lines:
            (text_width, text_height_single), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_text_width = max(max_text_width, text_width)
            total_text_height += line_height

        # Text box dimensions (approximate). Adjust y-offset for text origin.
        text_box_x1 = int(x_center)
        text_box_y1 = int(y_center) - 40 - text_height_single # Top of the text block
        text_box_x2 = int(x_center) + max_text_width
        text_box_y2 = int(y_center) - 40 + (len(text_lines) * line_height) # Bottom of the text block

        # Ensure text box coordinates are within image bounds
        text_box_x1 = max(0, text_box_x1)
        text_box_y1 = max(0, text_box_y1)
        text_box_x2 = min(image.shape[1], text_box_x2)
        text_box_y2 = min(image.shape[0], text_box_y2)

        # Re-using the IOU function for text collision detection
        current_text_bbox = [text_box_x1, text_box_y1, text_box_x2, text_box_y2]

        # Check for collisions with already drawn text boxes
        draw_this_text = True
        for prev_text_bbox in drawn_text_boxes:
            iou = calculate_iou(current_text_bbox, prev_text_bbox[:4]) # Use the new calculate_iou
            # If IOU is high, consider it a collision
            if iou > 0.1:
                # If current box score is lower or equal than colliding one, suppress current text
                if value['score'] <= prev_text_bbox[4]:
                    draw_this_text = False
                    break

        if draw_this_text:
            # Display all prepared text lines
            for i, line in enumerate(text_lines):
                text_origin_x = int(x_center)
                text_origin_y = int(y_center) - 40 + (i * 15)

                cv2.putText(image, line,
                            (text_origin_x, text_origin_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) # Black text

            # Add the text box information to the list of drawn text boxes
            drawn_text_boxes.append([text_box_x1, text_box_y1, text_box_x2, text_box_y2, value['score']])

    return image


if __name__ == '__main__':
    cap = cv2.VideoCapture('15sec_input_720p.mp4')

    # Get input video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

    groups_color_filters = {
    "T1": ((195, 225, 0.2, 0.6, 0.7, 1.0),),
    "T2": ((340, 360, 0.5, 1.0, 0.4, 1.0), (0, 20, 0.5, 1.0, 0.4, 1.0)),
    "R":  ((55, 70, 0.7, 1.0, 0.8, 1.0),),
    "G":  ((45, 65, 0.1, 0.3, 0.85, 1.0),)
    }


    field_color = (40, 150, 0.15, 1, 0.3, 0.8)

    rect_color = {
        'R': (0, 255, 255),
        'T1': (255, 0, 0),
        'T2': (0, 0, 255),
        'G': (255, 0, 255)
    }

    model = YOLO('best.pt')

    print("Model Class Names:", model.names)

    counter, target = 0, 0  # Process every frame

    while True:
        if counter == target:
            ret, full_size_frame = cap.read()
            counter = 0
            if not ret:
                print('Looping video...')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_for_prediction = cv2.resize(full_size_frame, (640, 360))

            h0, w0 = full_size_frame.shape[:2]
            h, w = frame_for_prediction.shape[:2]
            aspect_ratio = w0 / w

            results = model.predict(frame_for_prediction, verbose=False)
            result = results[0]

            # Pass `frame_for_prediction` to detect_colors as it expects the scaled frame
            image_with_detections, obj = detect_colors(frame_for_prediction, result, 1, model.names)

            out_image = draw_boxes(full_size_frame, obj, aspect_ratio)

            # Write the frame to the output video file
            out.write(out_image)

            cv2.imshow('YOLO', out_image)
        else:
            cap.grab()
            counter += 1

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release() # Release the VideoWriter object