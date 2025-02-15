"""
Defines the graph, the shortest-path algorithm, and simulates tray rotation.
"""

import cv2

graph = {
    "plastic": ["paper", "food waste"],
    "paper": ["plastic", "metal"],
    "food waste": ["plastic", "metal"],
    "metal": ["paper", "food waste"]
}

def find_shortest_path(graph, start, target):
    queue = [(start, [start])]
    visited = set()
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex == target:
            return path
        visited.add(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None

def simulate_tray_rotation(target_compartment, frame):
    default = "plastic"
    path_to_target = find_shortest_path(graph, default, target_compartment)
    path_to_default = find_shortest_path(graph, target_compartment, default)
    combined_path = path_to_target + path_to_default[1:]
    print(f"Tray rotation simulation started. Rotation path: {combined_path}")
    rotation_message = "Detection stopped, bin rotating"
    for i in range(1, len(combined_path)):
        step_msg = f"Rotating from {combined_path[i-1]} to {combined_path[i]}"
        print(step_msg)
        cv2.putText(frame, rotation_message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, step_msg, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Camera View", frame)
        cv2.waitKey(1000)
    print("Tray rotation complete. Returning to detection mode.")