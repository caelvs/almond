import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "./ExerciseScreen.css";

const CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function ExerciseScreen({ exercise, onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const phaseRef = useRef(exercise.initialPhase);
  const exerciseRef = useRef(exercise);

  const [status, setStatus] = useState("모델 로딩 중...");
  const [count, setCount] = useState(0);
  const [phase, setPhase] = useState(exercise.initialPhase);
  const [angle, setAngle] = useState(null);

  useEffect(() => {
    let animationId;

    const setup = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const detector = await poseDetection.createDetector(
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
      const video = videoRef.current;
      if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
        animationId = requestAnimationFrame(() => detect(detector));
        return;
      }
      const poses = await detector.estimatePoses(video);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (poses.length > 0) {
        const kp = poses[0].keypoints;
        const ex = exerciseRef.current;

        const { newPhase, counted } = ex.detect(kp, phaseRef.current);
        if (newPhase !== phaseRef.current) {
          phaseRef.current = newPhase;
          setPhase(newPhase);
        }
        if (counted) setCount((c) => c + 1);

        const displayAngle = ex.getDisplayAngle(kp);
        setAngle(isNaN(displayAngle) ? null : displayAngle);

        const isGood = ex.isGoodPose(displayAngle);
        const feedbackColor = isGood ? "lime" : "red";

        kp.forEach((p) => {
          if (p.score > 0.5) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
          }
        });

        CONNECTIONS.forEach(([i, j]) => {
          const a = kp[i];
          const b = kp[j];
          if (a.score > 0.5 && b.score > 0.5) {
            const isFeedback =
              ex.feedbackKeypoints.includes(i) &&
              ex.feedbackKeypoints.includes(j);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = isFeedback ? feedbackColor : "rgba(255,255,255,0.6)";
            ctx.lineWidth = 3;
            ctx.stroke();
          }
        });
      }

      animationId = requestAnimationFrame(() => detect(detector));
    };

    setup();
    return () => {
      cancelAnimationFrame(animationId);
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const phaseLabel = exercise.getPhaseLabel(phase);

  return (
    <div className="exercise-screen">
      <div
        className="ex-header"
        style={{ borderBottom: `3px solid ${exercise.gradientFrom}` }}
      >
        <button className="back-btn" onClick={onBack}>
          ← 뒤로
        </button>
        <div className="ex-title">
          <span className="ex-emoji">{exercise.emoji}</span>
          <span className="ex-name">{exercise.name}</span>
        </div>
        <div className="ex-count">
          <span className="count-num">{count}</span>
          <span className="count-unit">회</span>
        </div>
      </div>

      <div className="video-area">
        <div className="video-wrapper">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} />
        </div>
        {status !== "감지 중..." && (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <p>{status}</p>
          </div>
        )}
      </div>

      <div className="ex-footer">
        <div className="stat-box">
          <span className="stat-label">상태</span>
          <span className={`stat-value ${phaseLabel === "준비" ? "phase-ready" : "phase-active"}`}>
            {phaseLabel}
          </span>
        </div>
        <div className="stat-box stat-box--center">
          <span className="stat-label">각도</span>
          <span className="stat-value">
            {angle !== null ? `${angle}°` : "—"}
          </span>
        </div>
        <div className="stat-box">
          <span className="stat-label">횟수</span>
          <span className="stat-value stat-count">{count}</span>
        </div>
      </div>
    </div>
  );
}

export default ExerciseScreen;
