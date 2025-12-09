import cv2
import mediapipe as mp
import asyncio
import websockets
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

async def send_data(websocket, path=None):  # <-- path has default value
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_data = {"gesture": "none", "dx": 0, "dy": 0, "dz": 0}

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

                lm = results.multi_hand_landmarks[0]
                wrist = lm.landmark[0]
                x, y = wrist.x, wrist.y
                gesture_data["dx"] = (x - 0.5) * 2
                gesture_data["dy"] = (y - 0.5) * 2
                gesture_data["gesture"] = "move"

            await websocket.send(json.dumps(gesture_data))

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(send_data, "127.0.0.1", 8765):
        print("WebSocket server started on ws://127.0.0.1:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
