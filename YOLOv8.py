import cv2
from ultralytics import YOLO
import torch

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 더 정확한 YOLO 모델 불러오기 (yolov8m.pt 모델 사용)
model = YOLO("yolov8m.pt")
model.to(device)  # 모델을 선택한 장치로 이동

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 카메라 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 신뢰도 임계값 설정
CONFIDENCE_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    # YOLO를 사용하여 객체 감지
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    # 결과에서 바운딩 박스와 레이블 추출
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
        confidences = result.boxes.conf.cpu().numpy()  # 신뢰도
        class_ids = result.boxes.cls.cpu().numpy()  # 클래스 ID

        # 바운딩 박스 그리기
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(class_id)]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow("Real-Time Object Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
