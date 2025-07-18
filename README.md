# Soccer-Player-Re-Identification
implements a real-time player tracking system using the YOLOv11 object detection model and Deep SORT for consistent object tracking. The system also classifies players into teams and roles (like referee or goalkeeper) using color detection logic.

---

##Approach and Methodology

1. **Object Detection with YOLOv11**  
   We fine-tuned a YOLOv11 model (`best.pt`) to detect four classes:
   - Ball
   - Goalkeeper
   - Player
   - Referee

2. **Real-Time Multi-Object Tracking with Deep SORT**  
   For temporal tracking of players across frames, we used Deep SORT with a MobileNet-based appearance embedder. This ensured consistent track IDs even when players temporarily left the frame.

3. **Color-Based Role and Team Identification**  
   A custom color clustering and filtering logic was implemented:
   - Extract jersey (top) and shorts (bottom) regions of players.
   - Use HSV thresholds to filter out field/background colors.
   - Match dominant jersey color with predefined HSV ranges for team classification:
     - Team 1: #96cbfa (Blue)
     - Team 2: #ba2930 (Red)
     - Referee: #f3f51c (Yellow)
     - Goalkeeper: #fffee0 (Light Yellow)

4. **Overlay and Visual Display**  
   Bounding boxes, IDs, roles, and team labels are rendered onto the original frame, with collision-avoidance logic for label placement using IOU suppression.

---

## Techniques Tried and Outcomes

| Technique | Description | Outcome |
|----------|-------------|---------|
| YOLOv11 fine-tuning | Training custom object detection model on match data | Accurate detection of players, ball, referee |
| Deep SORT (with MobileNet) | Embedding-based tracking | Maintains consistent IDs even with occlusion or re-entry |
| KMeans clustering | Extracts dominant colors from jersey and shorts | Allows team role classification even in mixed lighting |
| HSV threshold filtering | Removes field and background influence | Improved accuracy of color-based team identification |
| Dynamic IOU-based label suppression | Prevents overlapping text boxes | Cleaner visual presentation |

---

## 🚧 Challenges Encountered

- **Field Color Interference:**  
  The green pitch often interfered with jersey color detection. We resolved this by applying strict HSV thresholds for field masking.

- **Player Occlusion & Re-ID:**  
  In crowded scenes, players overlapped. Deep SORT with MobileNet helped but occasionally failed when occlusion was long-lasting.

- **Color Clustering Instability:**  
  KMeans occasionally returned unstable clusters on low-resolution frames. We added area/shape checks before resizing for better consistency.

- **Dynamic Lighting:**  
  Varying lighting caused hue/saturation shifts in jersey detection. HSV-based tolerance ranges were fine-tuned to account for this.

---

## If Incomplete: Next Steps

While the pipeline is largely functional, improvements with more time/resources could include:

- **Better Appearance Embeddings:**  
  Replace MobileNet with a ResNet-50 or custom sports ReID model to improve identity persistence.

- **Post-Processing Analytics:**  
  Add player speed, team heatmaps, or ball possession statistics.

- **Web Interface:**  
  Deploy the project on a web dashboard (e.g., Flask + WebRTC) for live match visualization.

- **Model Robustness:**  
  Train with more diverse datasets (stadium sizes, weather, resolutions) for better generalization.
