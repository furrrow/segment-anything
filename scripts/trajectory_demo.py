import numpy as np
import cv2
from cv2 import imshow
# Camera 395 parameters (example values)

def compute_trajectory(v, w, dt, steps):
    x, y, theta = 0, 0, 0  # Initial pose
    trajectory = []
    for _ in range(steps):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        trajectory.append([x, y, 0])
        # print(x,y, v,theta,dt )
    return np.array(trajectory)

def project_to_image(trajectory, K, R, t):
    trajectory_cam = (R @ trajectory.T + t).T

    # Project points into the image frame
    points_2d = K @ trajectory_cam.T
    # print(trajectory)
    # print(trajectory_cam)
    # print(points_2d)
    points_2d /= points_2d[2]
    return points_2d[:2].T


fx, fy = 395, 395  # Focal lengths
cx, cy = 480, 270  # image center
width,height = 960, 540

# Camera matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Camera rotaion + translation

R = np.array([[0,1,0],
              [0,0,1],
              [1,0,0]])
t = np.array([[0], [-0.2], [0]])  # Translation vector

# Parameters

vel = 1.5
w = [-0.5,0,1]
dt = 0.1
steps = 10

# image_path = "/home/jim/Downloads/VLM_images/image_000007.png"
image_path = "/home/jim/Downloads/VLM_images/image_000253.png"
image = cv2.imread(image_path)
# image = cv2.resize(image, (1280,720))
print("image shape:", image.shape)

for i in w:
  # print(vel,w)
  trajectory_robot = compute_trajectory(vel, i, dt, steps)

  # print(trajectory_robot)
  points_2d = project_to_image(trajectory_robot, K, R, t)

  points_2d = np.array([[width, height]]) - points_2d
  # Load a sample camera image


  valid_points = []
  for point in points_2d:
      u, v = int(point[0]), int(point[1])
      if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
          valid_points.append((u, v))

  if len(valid_points) > 1:
      valid_points = np.array(valid_points, dtype=np.int32)

      if i > 0:
        cv2.polylines(image, [valid_points], isClosed=False, color=(0, 0, 255), thickness=5)  # Blue arc
      elif i == 0:
        cv2.polylines(image, [valid_points], isClosed=False, color=(0, 255, 0), thickness=5)  # Blue arc
      else:
        cv2.polylines(image, [valid_points], isClosed=False, color=(255, 0, 0), thickness=5)  # Blue arc

# print(points_2d)
# Display the result
cv2.imshow("window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()