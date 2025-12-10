import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import math

mp_hands = mp.solutions.hands
# Improved confidence settings based on MediaPipe documentation
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex)"""
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_finger_curled(landmarks, tip_idx, pip_idx, mcp_idx):
    """Check if finger is curled by comparing tip position to MCP"""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    
    # Finger is curled if tip is closer to wrist than PIP joint
    return tip.y > pip.y

def recognize_gesture(hand_landmarks):
    """Recognize hand gestures using MediaPipe 21 landmarks"""
    lm = hand_landmarks.landmark
    
    # MediaPipe Hand Landmarks (21 points total)
    # 0: Wrist
    # Thumb: 1, 2, 3, 4
    # Index: 5, 6, 7, 8
    # Middle: 9, 10, 11, 12
    # Ring: 13, 14, 15, 16
    # Pinky: 17, 18, 19, 20
    
    # Check if each finger is extended
    # For thumb: check if tip is far from palm
    thumb_extended = lm[4].x < lm[3].x if lm[4].x < lm[0].x else lm[4].x > lm[3].x
    
    # For other fingers: check if tip is above PIP joint
    index_extended = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    ring_extended = lm[16].y < lm[14].y
    pinky_extended = lm[20].y < lm[18].y
    
    # Count extended fingers
    extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Calculate key distances
    thumb_index_dist = calculate_distance(lm[4], lm[8])
    thumb_middle_dist = calculate_distance(lm[4], lm[12])
    
    # Palm size reference (distance from wrist to middle finger MCP)
    palm_size = calculate_distance(lm[0], lm[9])
    
    # Normalize distances by palm size
    thumb_index_norm = thumb_index_dist / palm_size if palm_size > 0 else 0
    thumb_middle_norm = thumb_middle_dist / palm_size if palm_size > 0 else 0
    
    # Store debug info
    debug_info = {
        "thumb": thumb_extended,
        "index": index_extended,
        "middle": middle_extended,
        "ring": ring_extended,
        "pinky": pinky_extended,
        "extended_count": extended_count,
        "thumb_index_norm": round(thumb_index_norm, 3),
        "thumb_middle_norm": round(thumb_middle_norm, 3),
        "palm_size": round(palm_size, 3)
    }
    
    # GESTURE RECOGNITION WITH CLEAR PRIORITY
    
    # 1. FIST - Check first before pinch (all fingers curled)
    if extended_count == 0 and not thumb_extended:
        return "fist", debug_info
    
    # 2. PINCH (thumb + index touching, AND index must be extended)
    if thumb_index_norm < 0.2 and index_extended:
        return "pinch", debug_info
    
    # 3. PINCH MIDDLE (thumb + middle touching, AND middle must be extended)
    if thumb_middle_norm < 0.2 and middle_extended:
        return "pinch_middle", debug_info
    
    # 4. OPEN HAND (all fingers extended)
    if extended_count == 4 and thumb_extended:
        return "open_hand", debug_info
    
    # 5. POINT (only index extended)
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "point", debug_info
    
    # 6. PEACE SIGN (index + middle extended)
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "peace", debug_info
    
    # 7. THREE FINGERS (index + middle + ring)
    if index_extended and middle_extended and ring_extended and not pinky_extended:
        return "three_fingers", debug_info
    
    # 8. FOUR FINGERS (all except thumb)
    if extended_count == 4 and not thumb_extended:
        return "four_fingers", debug_info
    
    # 9. SHAKA / HANG LOOSE (thumb + pinky)
    if thumb_extended and pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "shaka", debug_info
    
    # 10. OK SIGN (thumb + index circle, others extended)
    if thumb_index_norm < 0.5 and middle_extended and ring_extended:
        return "ok_sign", debug_info
    
    return "none", debug_info

async def send_data(websocket, path=None):
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror the frame for intuitive interaction
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = hands.process(rgb)
            
            gesture_data = {
                "gesture": "none",
                "dx": 0,
                "dy": 0,
                "dz": 0,
                "confidence": 0
            }
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Get hand landmarks
                lm = results.multi_hand_landmarks[0]
                wrist = lm.landmark[0]
                
                # Recognize gesture
                gesture, debug_info = recognize_gesture(results.multi_hand_landmarks[0])
                
                # Normalize coordinates to -1 to 1 range
                gesture_data["gesture"] = gesture
                gesture_data["dx"] = (wrist.x - 0.5) * 2
                gesture_data["dy"] = (0.5 - wrist.y) * 2  # Invert Y for Unity
                gesture_data["dz"] = wrist.z
                gesture_data["confidence"] = 1.0
                gesture_data["debug"] = debug_info
                
                # Display gesture on frame with background
                text = f"Gesture: {gesture}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(frame, (5, 5), (text_size[0] + 15, 45), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display debug info
                y_offset = 70
                debug_texts = [
                    f"Fingers: T:{int(debug_info['thumb'])} I:{int(debug_info['index'])} M:{int(debug_info['middle'])} R:{int(debug_info['ring'])} P:{int(debug_info['pinky'])}",
                    f"Extended: {debug_info['extended_count']}/4",
                    f"Thumb-Index: {debug_info['thumb_index_norm']}",
                    f"Thumb-Middle: {debug_info['thumb_middle_norm']}",
                    f"Palm Size: {debug_info['palm_size']}"
                ]
                
                for text in debug_texts:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (5, y_offset - 20), (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 30
            
            # Send data via WebSocket
            await websocket.send(json.dumps(gesture_data))
            
            # Display frame
            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(send_data, "127.0.0.1", 8765):
        print("=" * 60)
        print("WebSocket Server Started")
        print("=" * 60)
        print("Server: ws://127.0.0.1:8765")
        print("\nRecognized Gestures:")
        print("  • fist         - Closed fist (all fingers curled)")
        print("  • open_hand    - All five fingers extended")
        print("  • point        - Index finger pointing")
        print("  • peace        - Peace sign (index + middle)")
        print("  • three_fingers- Three fingers up")
        print("  • four_fingers - Four fingers (no thumb)")
        print("  • pinch        - Thumb touching index finger")
        print("  • pinch_middle - Thumb touching middle finger")
        print("  • shaka        - Hang loose (thumb + pinky)")
        print("  • ok_sign      - OK gesture")
        print("\nControls:")
        print("  Press ESC to quit")
        print("=" * 60)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())