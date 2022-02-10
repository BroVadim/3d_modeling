import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from edges import *
import time


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(5.0, -1.0, -10.0)  # поиграть с этими параметрами


def cube(vertices, edges):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def scale(n):
    return np.array([
        [n, 0, 0],
        [0, n, 0],
        [0, 0, n]
    ])


def create_3d(landmarks):
    points = []
    for landmark in landmarks:
        points.append((landmark.x, landmark.y, landmark.z))
    points = np.array(points)
    new_points = points @ scale(-5)
    new_points = [tuple(point) for point in new_points.tolist()]
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    cube(tuple(new_points), edges)
    pygame.display.flip()


def work_with_image(image_directory):
    img = cv2.imread(image_directory)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img_rgb)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            create_3d(faceLms.landmark)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def work_with_dynamic(directory=0):
    cap = cv2.VideoCapture(directory)
    pTime = 0
    while True:
        ret, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(img_rgb)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
                create_3d(faceLms.landmark)
        cv2.imshow('Image', img)
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# from numba import njit
# @njit - декоратор для библиотеки после писать функцию, которую следует разогнать

if __name__ == "__main__":
    # work_with_dynamic('videos/leha.mp4')
    work_with_dynamic()
    # work_with_image('videos/1.jpg')
