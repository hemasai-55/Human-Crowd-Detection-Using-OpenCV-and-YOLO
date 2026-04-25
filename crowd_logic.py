import cv2
import numpy as np

class CrowdLogic:
    def __init__(self, frame_width=640, frame_height=480):
        self.fw = frame_width
        self.fh = frame_height
        
        # 3x3 grid sizes
        self.cell_w = self.fw // 3
        self.cell_h = self.fh // 3
        
        # Heatmap accumulator (float32 for smooth fading)
        self.heatmap_accum = np.zeros((self.fh, self.fw), dtype=np.float32)
        
    def process(self, frame, detections):
        # 1. Update heatmap
        self.heatmap_accum *= 0.95 # Fade over time
        
        # Grid to count people in each of the 9 zones
        # 1D array of 9 ints, mapped row by row
        zone_counts = [0]*9
        
        # 2. Assign persons to zones
        for (x1, y1, x2, y2) in detections:
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Center of person
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Add to heatmap (draw a circle on accumulator)
            cv2.circle(self.heatmap_accum, (cx, cy), 30, 25.0, -1)
            
            # Find zone index 0-8
            col = cx // self.cell_w
            row = cy // self.cell_h
            if col > 2: col = 2
            if row > 2: row = 2
            idx = int(row * 3 + col)
            zone_counts[idx] += 1
            
        # 3. Create Heatmap overlay
        # Normalize and apply colormap
        heatmap_norm = np.clip(self.heatmap_accum, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Only overlay where heatmap_norm > 0 to keep background visible
        mask = heatmap_norm > 5
        # Blend entire frame then apply mask
        blended = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0.0)
        frame[mask] = blended[mask]
        
        # 4. Determine zone levels and draw grid
        zone_levels = []
        highest_density = 0
        for i in range(9):
            count = zone_counts[i]
            if count > highest_density:
                highest_density = count
                
            col = i % 3
            row = i // 3
            x1 = col * self.cell_w
            y1 = row * self.cell_h
            x2 = x1 + self.cell_w
            y2 = y1 + self.cell_h
            
            if count <= 1:
                level = "LOW"
                color = (0, 255, 0) # Green
            elif count <= 3:
                level = "MEDIUM"
                color = (0, 255, 255) # Yellow
            else:
                level = "HIGH"
                color = (0, 0, 255) # Red
                
            zone_levels.append(level)
            
            # Draw grid cell rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Put text for density
            cv2.putText(frame, f"Zone {i+1}: {count}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        # 5. Overall logic
        total_people = sum(zone_counts)
        if total_people < 5:
            overall_level = "LOW"
            action = "Normal Entry"
        elif total_people <= 15:
            overall_level = "MEDIUM"
            action = "Slow Entry"
        else:
            overall_level = "HIGH"
            action = "STOP Entry"

        data = {
            "total_people": total_people,
            "highest_density": highest_density,
            "zones": zone_levels,
            "zone_counts": zone_counts,
            "overall_level": overall_level,
            "action": action
        }
        
        return frame, data
