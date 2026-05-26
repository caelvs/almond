import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "./ExerciseScreen.css";

const SIGNAL_LABELS = {
  depth_insufficient: "깊이 부족",
  depth_incomplete_top: "완전히 펴기 부족",
  knee_valgus: "무릎 안쪽 모임",
  back_tilt: "허리 기울어짐",
  knee_asymmetry: "좌우 무릎 비대칭",
  hip_misalign: "엉덩이 정렬",
  elbow_flare: "팔꿈치 벌어짐",
  elbow_asymmetry: "좌우 팔꿈치 비대칭",
};

const ERROR_MESSAGES = {
  camera_denied: "카메라 권한이 거부되었습니다.\n브라우저 주소창의 자물쇠 아이콘을 클릭해\n카메라 접근을 허용해주세요.",
  camera_error: "카메라를 시작할 수 없습니다.\n카메라가 연결되어 있는지 확인 후\n페이지를 새로고침해주세요.",
  load_error: "AI 모델 로딩에 실패했습니다.\n인터넷 연결을 확인 후\n페이지를 새로고침해주세요.",
};


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
  const goodFramesRef = useRef(0);
  const totalFramesRef = useRef(0);
  const signalCountRef = useRef({});
  const asymmetryTotalRef = useRef(0);
  const asymmetryCountRef = useRef(0);

  const [showGuide, setShowGuide] = useState(true);
  const [showResult, setShowResult] = useState(false);
  const [status, setStatus] = useState("모델 로딩 중...");
  const [count, setCount] = useState(0);
  const [phase, setPhase] = useState(exercise.initialPhase);
  const [angle, setAngle] = useState(null);
  const [signals, setSignals] = useState([]);
  const [detected, setDetected] = useState(true);

  useEffect(() => {
    if (showGuide) return;

    let animationId;

    const setup = async () => {
      try {
        setStatus("모델 로딩 중...");
        await tf.setBackend("webgl");
        await tf.ready();
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );

        setStatus("카메라 준비 중...");
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (e) {
          const denied = e.name === "NotAllowedError" || e.name === "PermissionDeniedError";
          setStatus(denied ? "camera_denied" : "camera_error");
          return;
        }

        if (!videoRef.current) return;
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
          setStatus("감지 중...");
          detect(detector);
        };
      } catch (e) {
        setStatus("load_error");
      }
    };

    const detect = async (detector) => {
      try {
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

      if (poses.length === 0) {
        setDetected(false);
        animationId = requestAnimationFrame(() => detect(detector));
        return;
      }

      setDetected(true);
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

      const { signals: newSignals } = ex.evaluate(kp, phaseRef.current);
      console.log("signals", newSignals);
      const hasError = newSignals.some((s) => s.severity === "error");
      totalFramesRef.current += 1;
      if (!hasError) goodFramesRef.current += 1;

      setSignals(newSignals);
      for (const sig of newSignals) {
        if (sig.severity === "error" || sig.severity === "warning") {
          signalCountRef.current[sig.id] = (signalCountRef.current[sig.id] ?? 0) + 1;
        }
        if (sig.value != null && (sig.id === "knee_asymmetry" || sig.id === "elbow_asymmetry")) {
          asymmetryTotalRef.current += sig.value;
          asymmetryCountRef.current += 1;
        }
      }

      // error=red, warning=yellow; error takes priority over warning
      const kpColorMap = {};
      for (const sig of newSignals) {
        if (sig.severity === "error" || sig.severity === "warning") {
          const color = sig.severity === "error" ? "#ef4444" : "#facc15";
          for (const idx of sig.affectedKeypoints) {
            if (!kpColorMap[idx] || sig.severity === "error") kpColorMap[idx] = color;
          }
        }
      }

      kp.forEach((p, i) => {
        if (p.score > 0.5) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = kpColorMap[i] ?? "lime";
          ctx.fill();
        }
      });

      CONNECTIONS.forEach(([i, j]) => {
        const a = kp[i];
        const b = kp[j];
        if (a.score > 0.5 && b.score > 0.5) {
          const ci = kpColorMap[i];
          const cj = kpColorMap[j];
          const strokeStyle =
            ci && cj
              ? ci === "#ef4444" || cj === "#ef4444" ? "#ef4444" : "#facc15"
              : "rgba(255,255,255,0.6)";
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      });

      animationId = requestAnimationFrame(() => detect(detector));
      } catch (e) {
        animationId = requestAnimationFrame(() => detect(detector));
      }
    };

    setup();
    const video = videoRef.current;
    return () => {
      cancelAnimationFrame(animationId);
      if (video?.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [showGuide]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFinish = () => setShowResult(true);

  const handleRestart = () => {
    phaseRef.current = exercise.initialPhase;
    goodFramesRef.current = 0;
    totalFramesRef.current = 0;
    signalCountRef.current = {};
    asymmetryTotalRef.current = 0;
    asymmetryCountRef.current = 0;
    setCount(0);
    setPhase(exercise.initialPhase);
    setAngle(null);
    setSignals([]);
    setDetected(true);
    setShowResult(false);
    setShowGuide(true);
  };

  const quality =
    totalFramesRef.current > 0
      ? Math.round((goodFramesRef.current / totalFramesRef.current) * 100)
      : 0;

  const top3 = Object.entries(signalCountRef.current)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)
    .map(([id, count]) => ({ label: SIGNAL_LABELS[id] ?? id, count }));

  const avgAsymmetry = asymmetryCountRef.current > 0
    ? Math.round(asymmetryTotalRef.current / asymmetryCountRef.current)
    : null;

  const phaseLabel = exercise.getPhaseLabel(phase);

  return (
    <div className="exercise-screen">
      {/* D: 자세 가이드 */}
      {showGuide && (
        <div className="guide-overlay">
          <div className="guide-card">
            <div className="guide-emoji">{exercise.emoji}</div>
            <h2 className="guide-name">{exercise.name}</h2>
            <p className="guide-muscles">{exercise.muscles}</p>
            <ul className="guide-points">
              {exercise.guidePoints.map((point, i) => (
                <li key={i}>{point}</li>
              ))}
            </ul>
            <button
              className="guide-start-btn"
              style={{
                background: `linear-gradient(135deg, ${exercise.gradientFrom}, ${exercise.gradientTo})`,
              }}
              onClick={() => setShowGuide(false)}
            >
              시작하기
            </button>
          </div>
        </div>
      )}

      {/* C: 결과 요약 */}
      {showResult && (
        <div className="result-overlay">
          <div className="result-card">
            <div className="result-emoji">🎉</div>
            <h2 className="result-title">운동 완료!</h2>
            <div className="result-stats">
              <div className="result-stat">
                <span className="result-stat-label">총 횟수</span>
                <span className="result-stat-value">{count}회</span>
              </div>
              <div className="result-stat">
                <span className="result-stat-label">자세 정확도</span>
                <span className="result-stat-value">{quality}%</span>
              </div>
              {avgAsymmetry !== null && (
                <div className="result-stat">
                  <span className="result-stat-label">비대칭 평균</span>
                  <span className="result-stat-value result-stat-value--asym">{avgAsymmetry}°</span>
                </div>
              )}
            </div>
            {top3.length > 0 && (
              <div className="result-top3">
                <p className="result-top3-title">오류 빈도 TOP {top3.length}</p>
                {top3.map(({ label, count: c }, i) => (
                  <div key={i} className="result-top3-item">
                    <span className="result-top3-rank">{i + 1}</span>
                    <span className="result-top3-label">{label}</span>
                    <span className="result-top3-count">{c}회</span>
                  </div>
                ))}
              </div>
            )}
            <div className="result-btns">
              <button className="result-btn result-btn--secondary" onClick={handleRestart}>
                다시하기
              </button>
              <button className="result-btn result-btn--primary" onClick={onBack}>
                홈으로
              </button>
            </div>
          </div>
        </div>
      )}

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
          <div className={`loading-overlay${status in ERROR_MESSAGES ? " loading-overlay--error" : ""}`}>
            {status in ERROR_MESSAGES ? (
              <>
                <p className="loading-error-icon">⚠️</p>
                <p className="loading-error-msg">{ERROR_MESSAGES[status]}</p>
                <button className="loading-back-btn" onClick={onBack}>홈으로 돌아가기</button>
              </>
            ) : (
              <>
                <div className="loading-spinner" />
                <p>{status}</p>
              </>
            )}
          </div>
        )}
        {/* B: 미감지 경고 / LOCKED 배지 */}
        {status === "감지 중..." && !detected && (
          <div className="no-detect-banner">
            전신이 카메라에 보이도록 위치를 조정하세요
          </div>
        )}
        {/* A: 상세 피드백 텍스트 */}
        {status === "감지 중..." && detected && signals.length > 0 && (() => {
          const top = signals.find((s) => s.severity === "error") ?? signals.find((s) => s.severity === "warning") ?? signals[0];
          return <div className="feedback-overlay">{top.message}</div>;
        })()}
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

      <div className="finish-bar">
        <button className="finish-btn" onClick={handleFinish}>
          운동 완료
        </button>
      </div>
    </div>
  );
}

export default ExerciseScreen;
