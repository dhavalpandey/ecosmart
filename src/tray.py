import cv2

# Demo constant (wether we actually rotate servo or simulate)
DEMO = False

ROTATION_STEP_TIME = 1000

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

def simulate_tray_rotation(target_compartment, frame, rotation_delay=ROTATION_STEP_TIME):
    if DEMO:
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
            cv2.waitKey(rotation_delay)
        print("Tray rotation complete. Returning to detection mode.")
    else:
        import RPi.GPIO as GPIO
        from time import sleep
        print("Servo rotation simulation started using hardware control.")

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(11, GPIO.OUT)
        p = GPIO.PWM(11, 50)
        p.start(0)

        def set_servo_angle(angle):
            duty = 3 + (angle / 180) * 9
            p.ChangeDutyCycle(duty)
            sleep(1)
            return duty
        
        direction = "clockwise" if target_compartment in ["plastic", "paper", "metal"] else "anticlockwise"
        step = 90 if direction == "clockwise" else -90

        default = "plastic"
        path_to_target = find_shortest_path(graph, default, target_compartment)
        path_to_default = find_shortest_path(graph, target_compartment, default)
        combined_path = path_to_target + path_to_default[1:]
        print(f"Calculated servo path: {combined_path} using {direction} rotation.")

        current_angle = 0 
        print(f"Starting at {default} (angle: {current_angle}°)")
        for i in range(1, len(combined_path)):
            print(f"Rotating from {combined_path[i-1]} to {combined_path[i]}...")
            current_angle += step
            current_angle = max(0, min(180, current_angle))
            print(f"Now at angle: {current_angle}° corresponding to {combined_path[i]}")
            set_servo_angle(current_angle)

        print("Rotation to target completed. Now returning to default position.")

        # Rotate back to default position.
        for i in range(len(combined_path)-1, 0, -1):
            print(f"Rotating from {combined_path[i]} to {combined_path[i-1]}...")
            current_angle -= step
            current_angle = max(0, min(180, current_angle))
            print(f"Now at angle: {current_angle}° corresponding to {combined_path[i-1]}")
            set_servo_angle(current_angle)

        print("Servo rotation complete. Returning to detection mode.")
        p.stop()
        GPIO.cleanup()