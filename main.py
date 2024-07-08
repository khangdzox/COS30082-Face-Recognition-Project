
# dependencies
from os import name
import keras
import tensorflow as tf
import gradio as gr
import dlib
import numpy as np

# model loading
embedding = keras.models.load_model('embedding_euclidean.keras')
threshold = 0.6
eye_threshold = 0.25

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

# face detector
detector = dlib.get_frontal_face_detector() # type: ignore
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # type: ignore

left_eye_landmarks = list(range(36, 42))
right_eye_landmarks = list(range(42, 48))

def eye_aspect_ratio(eye):
    # [pyimagesearch](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
    # euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def check_eye(image, face_det):
    landmarks = landmark_predictor(image, face_det)

    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_landmarks])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_landmarks])

    left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
    right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

    ear = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    if ear > eye_threshold:
        return "eyes_opened", ear
    else:
        return "eyes_closed", ear

def check_face(image, face_det):

    det_size = min(max(face_det.width(), face_det.height()), image.shape[0] // 2, image.shape[1] // 2)

    center_x = min(max(face_det.center().x, det_size), image.shape[1] - det_size)
    center_y = min(max(face_det.center().y, det_size), image.shape[0] - det_size)

    top = center_y - det_size
    bottom = center_y + det_size
    left = center_x - det_size
    right = center_x + det_size

    face_boundary = (left, top, right, bottom)

    face_img = image[top:bottom, left:right]

    face_img = keras.preprocessing.image.smart_resize(face_img, (64, 64), "bicubic")
    face_img_processed = keras.applications.resnet.preprocess_input(face_img)

    assert embedding is not None

    face_embedding = embedding(np.expand_dims(face_img_processed, axis=0)).numpy()[0]

    try:
        with open('embeddings.npy', 'rb') as f:
            embeddings = np.load(f)

        with open('names.txt', 'r') as f:
            names = f.read().splitlines()

    except FileNotFoundError:
        raise gr.Error("No faces registered. Please register a face before checking-in")

    distances = np.array([distance(face_embedding, emb) for emb in embeddings])
    face_min_distance = np.min(distances)
    face_min_name = names[np.argmin(distances)]

    if face_min_distance < threshold:
        return True, face_min_name, face_min_distance, face_boundary

    return False, "", face_min_distance, face_boundary

def check_in_face(image, state):
    if image is None:
        raise gr.Error("Press recording before checkin")

    dets = detector(image, 2)

    if (len(dets) > 1):
        state["taken_actions"] = set()
        return "", "Multiple faces detected. Please try one face at a time.", "", (image, []), state
    elif (len(dets) == 0):
        state["taken_actions"] = set()
        return "", "No face detected. Please try again.", "", (image, []), state

    face_det = dets[0]

    eye_action, ear = check_eye(image, face_det)

    registered, face_min_name, face_min_distance, face_boundary = check_face(image, face_det)

    if not registered:
        state["taken_actions"] = set()
        return (
            f"{eye_action}. Ratio: {ear}",
            f"Face not registered. Min distance: {face_min_distance}",
            "",
            (image, [(face_boundary, "Unregistered face")]),
            state
        )

    else:
        if face_min_name == state["last_face"]:
            state["taken_actions"].add(eye_action)

            if len(state["taken_actions"]) == 2:
                return (
                    f"{eye_action}. Ratio: {ear}",
                    f"Recognized {face_min_name}. Distance: {face_min_distance}",
                    f"{face_min_name} checked-in",
                    (image, [(face_boundary, f"{face_min_name}")]),
                    state
                )

            else:
                return (
                    f"{eye_action}. Ratio: {ear}",
                    f"Recognized {face_min_name}. Distance: {face_min_distance}",
                    f"Waiting for eye action",
                    (image, [(face_boundary, f"{face_min_name}")]),
                    state
                )

        else:
            state["taken_actions"] = {eye_action}
            state["last_face"] = face_min_name
            return (
                f"{eye_action}. Ratio: {ear}",
                f"Recognized {face_min_name}. Distance: {face_min_distance}",
                f"Waiting for eye action",
                (image, [(face_boundary, f"{face_min_name}")]),
                state
            )

def register_face(image, name: str):
    if image is None:
        raise gr.Error("Press recording before registering")
    if not len(name):
        raise gr.Error("Please enter a name before registering")

    detector = dlib.get_frontal_face_detector() # type: ignore

    dets = detector(image, 1)

    if (len(dets) > 1):
        return (
            "Multiple faces detected. Please register one face at a time.",
            (image, [((det.left(), det.top(), det.right(), det.bottom()), f"Face #{i}") for i, det in enumerate(dets)])
        )
    elif (len(dets) == 0):
        return "No face detected. Please try again.", (image, [])

    face_det = dets[0]
    det_size = max(face_det.width(), face_det.height())

    center_x = min(max(face_det.center().x, det_size), image.shape[1] - det_size)
    center_y = min(max(face_det.center().y, det_size), image.shape[0] - det_size)

    top = center_y - det_size
    bottom = center_y + det_size
    left = center_x - det_size
    right = center_x + det_size

    sections = [((left, top, right, bottom), name)]

    face_img = image[top:bottom, left:right]

    face_img = keras.preprocessing.image.smart_resize(face_img, (64, 64), "bicubic")
    face_img_processed = keras.applications.resnet.preprocess_input(face_img)

    assert embedding is not None

    face_embedding = embedding(np.expand_dims(face_img_processed, axis=0)).numpy()[0]

    try:
        with open('embeddings.npy', 'rb') as f:
            embeddings = np.load(f)
            embeddings = np.vstack([embeddings, face_embedding])

    except FileNotFoundError:
        embeddings = np.array([face_embedding])

    finally:
        with open('embeddings.npy', 'wb') as f:
            np.save(f, embeddings)
        with open('names.txt', 'a') as f:
            f.write(name + '\n')

    return f"Registered one face for {name}", (image, sections)

# gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(sources=["webcam"], streaming=True)
            with gr.Row():
                with gr.Column():
                    check_in_button = gr.Button("Check-in", variant="primary")
                with gr.Column():
                    register_button = gr.Button("Register", variant="primary")
                    name_text = gr.Textbox(label="Name")
        with gr.Column(scale=1):
            result_annotated_img = gr.AnnotatedImage()
            eye_result = gr.Textbox(label="Eyes result")
            img_result = gr.Textbox(label="Image result")
            chk_result = gr.Textbox(label="Checked-in result")

    state = gr.State(value={
        "last_face": "",
        "taken_actions": set()
    })

    check_in_button.click(
        check_in_face,
        inputs=[image, state],
        outputs=[eye_result, img_result, chk_result, result_annotated_img, state]
    )

    register_button.click(
        register_face,
        inputs=[image, name_text],
        outputs=[img_result, result_annotated_img]
    )

demo.launch()