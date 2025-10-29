import cv2
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize, CoCreateInstance, GUID
from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator

# ---------------------------------
# WINDOWS AUDIO COM SETUP
# ---------------------------------

# Core Audio constants
eRender = 0        # output/render device
eMultimedia = 1    # "multimedia" default role (normal playback)

# The system CLSID for MMDeviceEnumerator
CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
# The IID for IMMDeviceEnumerator interface
IID_IMMDeviceEnumerator = GUID(str(IMMDeviceEnumerator._iid_))

# Initialize COM for this thread
CoInitialize()

# Create the MMDeviceEnumerator COM object.
# We call CoCreateInstance with positional args only to satisfy your comtypes version:
# CoCreateInstance(clsid, punkOuter, clsctx, interface_iid)
enumerator_obj = CoCreateInstance(
    CLSID_MMDeviceEnumerator,
    None,
    CLSCTX_ALL,
    IID_IMMDeviceEnumerator
)

# Cast that raw COM object to IMMDeviceEnumerator interface
enumerator = cast(enumerator_obj, POINTER(IMMDeviceEnumerator)).contents

# Ask Windows for the default playback device
device = enumerator.GetDefaultAudioEndpoint(eRender, eMultimedia)

# Activate the IAudioEndpointVolume interface on that device
endpoint_interface = device.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

# Now cast the result to IAudioEndpointVolume so we get volume methods
volume = cast(endpoint_interface, POINTER(IAudioEndpointVolume))

# We'll control volume as scalar (0.0 mute -> 1.0 max)
# Track UI state
volBar = 400
volPer = 0

# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ---------------------------------
# CAMERA SETUP
# ---------------------------------
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# ---------------------------------
# MAIN LOOP
# ---------------------------------
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break

        # Convert for MediaPipe
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # We'll draw overlays on this copy
        image = frame.copy()

        # Draw hand landmarks (skeleton)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Extract first detected hand's landmarks into [id, x, y]
        lmList = []
        if results.multi_hand_landmarks:
            firstHand = results.multi_hand_landmarks[0]
            h, w, c = image.shape
            for idx, lm in enumerate(firstHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])

        # If we have a hand, use thumb/index distance to adjust volume
        if len(lmList) != 0:
            # Thumb tip (id=4), Index tip (id=8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Midpoint
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw helper graphics
            cv2.circle(image, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Distance between thumb tip and index tip
            length = math.hypot(x2 - x1, y2 - y1)

            # Visual "pinch" feedback if very close
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            # Map length 50..220 px -> volume scalar 0.0..1.0
            volScalar = np.interp(length, [50, 220], [0.0, 1.0])
            volScalar = float(np.clip(volScalar, 0.0, 1.0))

            # Set system master volume (0.0 = mute, 1.0 = full)
            volume.SetMasterVolumeLevelScalar(volScalar, None)

            # UI bar (400px is bottom, 150px is top)
            volBar = np.interp(volScalar, [0.0, 1.0], [400, 150])
            volPer = volScalar * 100.0

        # Draw volume bar on screen
        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
        cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)

        # Draw volume percent text
        cv2.putText(
            image,
            f'{int(volPer)} %',
            (40, 450),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            3
        )

        # Show frame
        cv2.imshow('handDetector', image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup camera + windows
cam.release()
cv2.destroyAllWindows()
