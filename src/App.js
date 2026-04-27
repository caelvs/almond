import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";

// 벡터 각도 계산 함수
function getAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.sqrt(ab.x ** 2 + ab.y ** 2) * Math.sqrt(cb.x ** 2 + cb.y ** 2);
  return (Math.acos(dot / mag) * 180) / Math.PI;
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("모델 로딩 중...");
  const [count, setCount] = useState(0);
  const [phase, setPhase] = useState("standing");
  const phaseRef = useRef("standing");

  useEffect(() => {
    let detector;
    let animationId;

    const setup = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;

      videoRef.current.onloadeddata = () => {
        setStatus("감지 중...");
        detect(detector);
      };
    };

    const detect = async (detector) => {
      const poses = await detector.estimatePoses(videoRef.current);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (poses.length > 0) {
        const kp = poses[0].keypoints;

        // 왼쪽 무릎 각도 (엉덩이 - 무릎 - 발목)
        const leftHip = kp[11];
        const leftKnee = kp[13];
        const leftAnkle = kp[15];

        // 오른쪽 무릎 각도
        const rightHip = kp[12];
        const rightKnee = kp[14];
        const rightAnkle = kp[16];

        const leftAngle = getAngle(leftHip, leftKnee, leftAnkle);
        const rightAngle = getAngle(rightHip, rightKnee, rightAnkle);
        const avgAngle = (leftAngle + rightAngle) / 2;

        // FSM 횟수 카운트
        if (avgAngle < 100 && phaseRef.current === "standing") {
          phaseRef.current = "squatting";
          setPhase("squatting");
        } else if (avgAngle > 160 && phaseRef.current === "squatting") {
          phaseRef.current = "standing";
          setPhase("standing");
          setCount((prev) => prev + 1);
        }

        // 관절 그리기
        kp.forEach((point) => {
          if (point.score > 0.5) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
          }
        });

        // 무릎 각도에 따라 선 색상 변경
        const isGoodPose = avgAngle > 80 && avgAngle < 170;
        const lineColor = isGoodPose ? "lime" : "red";

        // 전체 스켈레톤 연결 정의
        const connections = [
          // 얼굴
          [0, 1], [0, 2], [1, 3], [2, 4],
          // 상체
          [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
          [5, 11], [6, 12], [11, 12],
          // 하체
          [11, 13], [13, 15], [12, 14], [14, 16],
        ];

        connections.forEach(([i, j]) => {
          const a = kp[i];
          const b = kp[j];
          if (a.score > 0.5 && b.score > 0.5) {
            // 하체 관절이면 자세 피드백 색상 적용
            const isLeg = [11, 12, 13, 14, 15, 16].includes(i) && [11, 12, 13, 14, 15, 16].includes(j);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = isLeg ? lineColor : "white";
            ctx.lineWidth = 3;
            ctx.stroke();
          }
        });
      }

      animationId = requestAnimationFrame(() => detect(detector));
    };

    setup();
    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
    <div style={{ textAlign: "center", background: "#111", minHeight: "100vh", color: "white" }}>
      <h1>ALMOND</h1>
      <p>{status}</p>
      <p>phase: {phase}</p>
      <h2>스쿼트 횟수: {count}</h2>
      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0 }} />
      </div>
    </div>
  );
}

export default App;