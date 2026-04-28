import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";

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
  const [set, setSet] = useState(1);
  const [angle, setAngle] = useState(0);
  const [feedback, setFeedback] = useState("");
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

        const leftHip = kp[11];
        const leftKnee = kp[13];
        const leftAnkle = kp[15];
        const rightHip = kp[12];
        const rightKnee = kp[14];
        const rightAnkle = kp[16];

        const leftAngle = getAngle(leftHip, leftKnee, leftAnkle);
        const rightAngle = getAngle(rightHip, rightKnee, rightAnkle);
        const avgAngle = (leftAngle + rightAngle) / 2;

        setAngle(Math.round(avgAngle));

        // 각도 구간별 피드백
        if (avgAngle > 90) {
          setFeedback("내려가세요!");
        } else if (avgAngle >= 60 && avgAngle <= 90) {
          setFeedback("좋은 자세!");
        } else if (avgAngle < 60) {
          setFeedback("올라오세요!");
        }

        // FSM 횟수 카운트
        if (avgAngle <= 90 && phaseRef.current === "standing") {
          phaseRef.current = "squatting";
        } else if (avgAngle > 160 && phaseRef.current === "squatting") {
          phaseRef.current = "standing";
          setCount((prev) => prev + 1);
        }

        // 각도 구간별 선 색상
        let lineColor;
        if (avgAngle > 90) {
          lineColor = "red";      // 내려가세요
        } else if (avgAngle >= 60) {
          lineColor = "lime";     // 좋은 자세
        } else {
          lineColor = "yellow";   // 올라오세요
        }

        // 전체 스켈레톤 연결 정의
        const connections = [
          [0, 1], [0, 2], [1, 3], [2, 4],
          [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
          [5, 11], [6, 12], [11, 12],
          [11, 13], [13, 15], [12, 14], [14, 16],
        ];

        connections.forEach(([i, j]) => {
          const a = kp[i];
          const b = kp[j];
          if (a.score > 0.5 && b.score > 0.5) {
            const isLeg = [11, 12, 13, 14, 15, 16].includes(i) && [11, 12, 13, 14, 15, 16].includes(j);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = isLeg ? lineColor : "white";
            ctx.lineWidth = 3;
            ctx.stroke();
          }
        });

        kp.forEach((point) => {
          if (point.score > 0.5) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
          }
        });
      }

      animationId = requestAnimationFrame(() => detect(detector));
    };

    setup();
    return () => cancelAnimationFrame(animationId);
  }, []);

  const handleReset = () => {
    setCount(0);
    setSet((prev) => prev + 1);
    setFeedback("");
    phaseRef.current = "standing";
  };

  return (
    <div style={{ textAlign: "center", background: "#111", minHeight: "100vh", color: "white", fontFamily: "sans-serif" }}>
      <h1>ALMOND</h1>
      <p>{status}</p>

      {/* 상태 정보 */}
      <div style={{ display: "flex", justifyContent: "center", gap: "40px", marginBottom: "10px" }}>
        <div>
          <p style={{ color: "#aaa", margin: 0 }}>세트</p>
          <h2 style={{ margin: 0 }}>{set}</h2>
        </div>
        <div>
          <p style={{ color: "#aaa", margin: 0 }}>횟수</p>
          <h2 style={{ margin: 0 }}>{count}</h2>
        </div>
        <div>
          <p style={{ color: "#aaa", margin: 0 }}>무릎 각도</p>
          <h2 style={{ margin: 0 }}>{angle}°</h2>
        </div>
      </div>

      {/* 텍스트 피드백 */}
      <p style={{
        fontSize: "24px",
        fontWeight: "bold",
        color: feedback === "좋은 자세!" ? "lime" : feedback === "올라오세요!" ? "yellow" : "red",
        height: "36px"
      }}>
        {feedback}
      </p>

      {/* 카메라 */}
      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0 }} />
      </div>

      {/* 다음 세트 버튼 */}
      <div style={{ marginTop: "20px" }}>
        <button
          onClick={handleReset}
          style={{
            padding: "12px 40px",
            fontSize: "18px",
            background: "#333",
            color: "white",
            border: "2px solid #555",
            borderRadius: "8px",
            cursor: "pointer"
          }}
        >
          다음 세트
        </button>
      </div>
    </div>
  );
}

export default App;