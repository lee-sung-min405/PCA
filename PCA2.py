import cv2
import numpy as np

# 학습 영상 경로
train_folder = 'face_img/train/'
# 실험 영상 경로
test_folder = 'face_img/test/'

# 학습 단계 1: 학습 영상 불러오기
print("학습 단계 1: 학습 영상 불러오기")
train_images = []
for i in range(310):
    filename = train_folder + f'train{i:03d}.jpg'
    image = cv2.imread(filename)
    train_images.append(image)

# 학습 단계 2: 컬러 영상을 명암도 영상으로 변환
print("학습 단계 2: 컬러 영상을 명암도 영상으로 변환")
train_gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images]

# 학습 단계 3: 영상 크기 변환
print("학습 단계 3: 영상 크기 변환")
resized_images = [cv2.resize(image, (120, 150)) for image in train_gray_images]

# 학습 단계 4: 영상을 벡터로 변환하여 저장
print("학습 단계 4: 영상을 벡터로 변환하여 저장")
x = np.array([image.flatten() for image in resized_images])

# 학습 단계 5: 배열 x를 float32 형으로 변환
print("학습 단계 5: 배열 x를 float32 형으로 변환")
x = x.astype(np.float32)

# 학습 단계 6: 학습 영상들의 평균 영상 계산
print("학습 단계 6: 학습 영상들의 평균 영상 계산")
mean_image = np.mean(x, axis=0)
mean_image = mean_image.reshape((150, 120)).astype(np.uint8)

# 평균 영상 저장
print("평균 영상 저장")
cv2.imwrite('mean_image.jpg', mean_image)

# 각 학습 영상과 평균 영상의 차 계산
print("각 학습 영상과 평균 영상의 차 계산")
diff_vectors = [image.flatten() - mean_image.flatten() for image in resized_images]

# 벡터를 행렬로 변환
print("벡터를 행렬로 변환")
diff_matrix = np.array(diff_vectors)
diff_matrix = diff_matrix.T
print(f"벡터를 행렬로 변환 : {diff_matrix.shape}")

# 공분산 행렬 계산 (Snapshot방법 사용)
print("공분산 행렬 계산 (Snapshot방법 사용)")
covariance_matrix = np.dot(diff_matrix.T, diff_matrix)
print(f"공분산 행렬 계산 : {covariance_matrix.shape}")

# 공분산 행렬의 고유벡터와 고유값 계산 (Snapshot방법 사용)
print("공분산 행렬의 고유벡터와 고유값 계산 (Snapshot방법 사용)")
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values = eigen_values[::-1]  # 고유값을 크기순으로 정렬 (내림차순)
eigen_vectors = eigen_vectors[:, ::-1]  # 고유벡터도 같은 순서로 정렬
print(f"공분산 행렬의 고유값 : {eigen_values.shape}")
print(f"공분산 행렬의 고유벡터 : {eigen_vectors.shape}")

# 정규화 시키는 공식
print("정규화 시키는 공식")
eigen_vectors = np.dot(diff_matrix, eigen_vectors)
eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=0)
print(f"정규화 이후 형상 : {eigen_vectors.shape}")

#고유값을 크기 순으로 정렬 (내림차순)
eigen_values = eigen_values[::-1] # 고유값을 크기순으로 정렬 (내림차순)
eigen_vectors = eigen_vectors[:, ::-1] # 고유벡터도 같은 순서로 정렬

#(고유값 전체 합) * K <= (정렬한 고유값의 합 1~v) 성립하는 최소 고유값 개수 v 선택
K = 0.91 # K 값에 따라 인식률 변화, 가장 좋은 결과 발생하는 K 값 선택 가능
eigen_sum = np.sum(eigen_values)
eigen_value_sum = np.cumsum(eigen_values)
v = np.sum(eigen_value_sum < (eigen_sum * K))

#v개의 고유값에 해당하는 v개의 고유벡터 선택
selected_eigen_values = eigen_values[:v]
selected_eigen_vectors = eigen_vectors[:, :v]
print(f"선택된 고유값: {selected_eigen_values.shape}")
print(f"선택된 고유벡터: {selected_eigen_vectors.shape}")

# 선택된 고유값에 해당하는 고유벡터로 이루어진 행렬 생성
selected_eigen_matrix = np.column_stack([selected_eigen_vectors[:, i] for i in range(selected_eigen_vectors.shape[1])])
print(f"선택된 고유벡터로 이루어진 행렬: {selected_eigen_matrix.shape}")

# 각 학습 영상과 평균 영상의 차를 구하고 변환 행렬에 곱하여 특징값 계산
print("각 학습 영상과 평균 영상의 차를 구하고 변환 행렬에 곱하여 특징값 계산")
features = []
for image in resized_images:
    diff_vector = image.flatten() - mean_image.flatten()
    feature_vector = np.dot(diff_vector, selected_eigen_matrix)
    features.append(feature_vector)

features = np.array(features)
print(f"학습 영상의 특징값: {features.shape}")

#########################TEST 단계#########################
# 실험 단계 1: 실험 영상 불러오기
print("실험 단계 1: 실험 영상 불러오기")
test_images = []
for i in range(92):
    filename = test_folder + f'test{i:03d}.jpg'
    image = cv2.imread(filename)
    test_images.append(image)

# 실험 단계 2: 컬러 영상을 명암도 영상으로 변환
print("실험 단계 2: 컬러 영상을 명암도 영상으로 변환")
test_gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images]

# 실험 단계 3: 영상 크기 변환
print("실험 단계 3: 영상 크기 변환")
resized_test_images = [cv2.resize(image, (120, 150)) for image in test_gray_images]

# 실험 단계 4: 영상을 벡터로 변환
print("실험 단계 4: 영상을 벡터로 변환")
test_x = np.array([image.flatten() for image in resized_test_images])

# 실험 단계 5: 배열 test_x를 float32 형으로 변환
print("실험 단계 5: 배열 test_x를 float32 형으로 변환")
test_x = test_x.astype(np.float32)

# 실험 단계 6: 각 실험 영상과 평균 영상의 차를 구하고 변환 행렬에 곱하여 특징값 계산
print("실험 단계 6: 각 실험 영상과 평균 영상의 차를 구하고 변환 행렬에 곱하여 특징값 계산")
test_features = []
for image in resized_test_images:
    diff_vector = image.flatten() - mean_image.flatten()
    feature_vector = np.dot(diff_vector, selected_eigen_matrix)
    test_features.append(feature_vector)

test_features = np.array(test_features)
print(f"입력 영상의 특징값: {test_features.shape}")

# 특징값과 테스트 영상의 유클리디안 거리 계산
print("특징값과 테스트 영상의 유클리디안 거리 계산")
distances = []
for feature in features:
    distance = np.linalg.norm(feature - test_features, axis=1)
    distances.append(distance)

distances = np.array(distances)
print(f"유클리디안 거리: {distances.shape}")

# 가장 작은 거리를 갖는 x^2 분류
print("가장 작은 거리를 갖는 x^2 분류")
min_distance_indices = np.argmin(distances, axis=0)
classified_images = [train_images[i] for i in min_distance_indices]

# x^2 영상 출력
current_index = 0
windows = []

def display_images(test_index):
    global current_index
    current_index = test_index
    if current_index < len(classified_images):
        window_name1 = f"test image : {current_index}"
        img1 = resized_test_images[current_index].copy()
        cv2.putText(img1, window_name1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.imshow(window_name1, img1)
        cv2.moveWindow(window_name1, 0, 0)  # 왼쪽 상단에 윈도우 위치 설정

        window_name2 = f"result image : {current_index}"
        img2 = classified_images[current_index].copy()
        cv2.putText(img2, window_name2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.imshow(window_name2, img2)
        cv2.moveWindow(window_name2, img1.shape[1] + 10, 0)  # 오른쪽 상단에 윈도우 위치 설정

        windows.append(window_name1)
        windows.append(window_name2)
        current_index += 1

test_index = int(input("test 입력 영상 번호 입력 (0 ~ 92): "))
display_images(test_index)

while True:
    key = cv2.waitKey(0)
    if key == ord(' '):
        for window in windows:
            cv2.destroyWindow(window)
        windows = []
        test_index = int(input("test 입력 영상 번호 입력 (0 ~ 92): "))
        display_images(test_index)
    elif key == 27:
        break

cv2.destroyAllWindows()