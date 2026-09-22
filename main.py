import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# 1. 페이지 설정
st.set_page_config(page_title="Virtual Motion Capture TA Tool", page_icon="🎭")
st.title("🎭 버추얼 모션 캡처 & 그래픽 파이프라인 툴")
st.caption("TA/TD 진로 탐구: MediaPipe 기반 실시간 3D Face/Pose Landmarks 추적 및 그래픽스 렌더링")

# 2. MediaPipe Face Mesh / Pose 설정
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 3. 사이드바 - TA/아티스트용 옵션
st.sidebar.header("⚙️ 모션 캡처 파이프라인 옵션")
show_mesh = st.sidebar.checkbox("3D 모션 캡처 랜드마크 표시", value=True)
color_mode = st.sidebar.selectbox("선 그래픽 색상 모드", ["기본 (Green)", "버추얼 네온 (Cyan)", "흑백 렌더링"])

# 4. 파일 및 카메라 입력
uploaded_file = st.file_uploader("모션 캡처를 진행할 인물 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("👈 왼쪽 사이드바 옵션을 설정하고, 모션 캡처를 분석할 인물 이미지/셀카를 업로드하세요.")
else:
    # 이미지 읽기 및 RGB 변환
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    
    # MediaPipe 연산 (모션 추적 연산)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(img_array)
        annotated_image = img_array.copy()

        # 랜드마크(모션 키포인트)가 감지되었을 때 그래픽 렌더링
        if results.multi_face_landmarks and show_mesh:
            for face_landmarks in results.multi_face_landmarks:
                # 색상 옵션에 따른 시각화
                if color_mode == "버추얼 네온 (Cyan)":
                    spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
                elif color_mode == "흑백 렌더링":
                    spec = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1)
                else:
                    spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=spec
                )

        # 예외 처리: 얼굴 감지 실패 시
        if not results.multi_face_landmarks:
            st.warning("⚠️ 이미지에서 얼굴 관절(Landmarks)을 찾을 수 없습니다. 인물이 명확한 사진을 넣어주세요.")

    # 5. 화면 비교 출력
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 원본 인물 이미지")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("🎭 3D 모션 캡처 렌더링")
        st.image(annotated_image, use_container_width=True)
