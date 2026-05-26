function getAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag =
    Math.sqrt(ab.x ** 2 + ab.y ** 2) * Math.sqrt(cb.x ** 2 + cb.y ** 2);
  if (mag === 0) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

// 어깨 너비 / 몸통 높이 비율로 정면 여부 판별
function isFrontFacing(kp) {
  if (kp[5].score < 0.3 || kp[6].score < 0.3) return false;
  const shoulderWidth = Math.abs(kp[5].x - kp[6].x);
  const torsoHeight = Math.abs((kp[5].y + kp[6].y) / 2 - (kp[11].y + kp[12].y) / 2);
  return torsoHeight > 0 && shoulderWidth / torsoHeight > 0.3;
}

// MoveNet keypoint indices:
// 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
// 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
// 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

export const EXERCISES = [
  {
    id: "squat",
    name: "스쿼트",
    emoji: "🏋️",
    description: "무릎을 구부려 하체를 단련하는 기본 운동",
    muscles: "허벅지 · 엉덩이 · 종아리",
    gradientFrom: "#4f46e5",
    gradientTo: "#7c3aed",
    initialPhase: "up",
    feedbackKeypoints: [11, 12, 13, 14, 15, 16],
    guidePoints: [
      "발을 어깨 너비로 벌리세요",
      "무릎이 발끝을 넘지 않도록 주의하세요",
      "허리를 곧게 펴고 가슴을 들어올리세요",
      "엉덩이를 의자에 앉듯이 천천히 내리세요",
    ],
    getPhaseLabel: (p) => (p === "up" ? "준비" : "운동 중"),
    evaluate: (kp, phase) => {
      const signals = [];
      const lKnee = getAngle(kp[11], kp[13], kp[15]);
      const rKnee = getAngle(kp[12], kp[14], kp[16]);
      const kneeAngle = (lKnee + rKnee) / 2;

      // 1. 깊이
      if (phase === "up") {
        if (kneeAngle < 160) {
          signals.push({ id: "depth_incomplete_top", severity: "error", affectedKeypoints: [11, 12, 13, 14, 15, 16], message: "무릎을 완전히 펴세요" });
        } else {
          signals.push({ id: "cue_descend", severity: "info", affectedKeypoints: [], message: "내려가세요!" });
        }
      } else {
        if (kneeAngle > 130) {
          signals.push({ id: "depth_insufficient", severity: "error", affectedKeypoints: [11, 12, 13, 14, 15, 16], message: "더 깊이 내려가세요" });
        } else {
          signals.push({ id: "cue_ascend", severity: "info", affectedKeypoints: [], message: "올라가세요!" });
        }

        // 2. 무릎 안쪽 모임
        const kneeWidth = Math.abs(kp[13].x - kp[14].x);
        const ankleWidth = Math.abs(kp[15].x - kp[16].x);
        if (ankleWidth > 0 && kneeWidth < ankleWidth * 0.8) {
          signals.push({ id: "knee_valgus", severity: "error", affectedKeypoints: [13, 14], message: "무릎이 안쪽으로 모이고 있어요" });
        }

        // 3. 등 굽음 (어깨 좌우 높이 차이로 감지)
        const torsoHeight = Math.abs((kp[11].y + kp[12].y) / 2 - (kp[5].y + kp[6].y) / 2);
        if (torsoHeight > 0 && Math.abs(kp[5].y - kp[6].y) / torsoHeight > 0.15) {
          signals.push({ id: "back_tilt", severity: "warning", affectedKeypoints: [5, 6, 11, 12], message: "허리를 곧게 펴세요" });
        }

        // 4. 좌우 비대칭 (정면 카메라에서만)
        if (isFrontFacing(kp)) {
          const asymmetry = Math.abs(lKnee - rKnee);
          if (asymmetry > 15) {
            signals.push({ id: "knee_asymmetry", severity: asymmetry > 25 ? "error" : "warning", affectedKeypoints: [13, 14, 15, 16], message: `좌우 무릎 각도 차이 ${Math.round(asymmetry)}°`, value: asymmetry });
          }
        }
      }

      return { signals };
    },
    getDisplayAngle: (kp) => {
      const l = getAngle(kp[11], kp[13], kp[15]);
      const r = getAngle(kp[12], kp[14], kp[16]);
      return Math.round((l + r) / 2);
    },
    detect: (kp, phase) => {
      const l = getAngle(kp[11], kp[13], kp[15]);
      const r = getAngle(kp[12], kp[14], kp[16]);
      const avg = (l + r) / 2;
      if (avg < 100 && phase === "up") return { newPhase: "down", counted: false };
      if (avg > 160 && phase === "down") return { newPhase: "up", counted: true };
      return { newPhase: phase, counted: false };
    },
  },
  {
    id: "pushup",
    name: "푸쉬업",
    emoji: "💪",
    description: "팔굽혀펴기로 가슴과 삼두를 강화하는 운동",
    muscles: "가슴 · 삼두 · 어깨",
    gradientFrom: "#059669",
    gradientTo: "#0891b2",
    initialPhase: "up",
    feedbackKeypoints: [5, 6, 7, 8, 9, 10],
    guidePoints: [
      "손을 어깨 너비보다 약간 넓게 짚으세요",
      "머리부터 발끝까지 일직선을 유지하세요",
      "팔꿈치를 몸통 가까이 당기며 내려가세요",
      "가슴이 바닥에 닿을 듯이 내려가세요",
    ],
    getPhaseLabel: (p) => (p === "up" ? "준비" : "운동 중"),
    evaluate: (kp, phase) => {
      const signals = [];
      const lElbow = getAngle(kp[5], kp[7], kp[9]);
      const rElbow = getAngle(kp[6], kp[8], kp[10]);
      const elbowAngle = (lElbow + rElbow) / 2;

      if (phase === "up") {
        if (elbowAngle < 150) {
          signals.push({ id: "depth_incomplete_top", severity: "error", affectedKeypoints: [5, 6, 7, 8, 9, 10], message: "팔을 완전히 펴세요" });
        } else {
          signals.push({ id: "cue_descend", severity: "info", affectedKeypoints: [], message: "내려가세요!" });
        }
      } else {
        if (elbowAngle > 100) {
          signals.push({ id: "depth_insufficient", severity: "error", affectedKeypoints: [5, 6, 7, 8, 9, 10], message: "더 깊이 내려가세요" });
        } else {
          signals.push({ id: "cue_ascend", severity: "info", affectedKeypoints: [], message: "올라가세요!" });
        }

        // 2. 엉덩이 처짐/들림
        const shoulderMidY = (kp[5].y + kp[6].y) / 2;
        const hipMidY = (kp[11].y + kp[12].y) / 2;
        const ankleMidY = (kp[15].y + kp[16].y) / 2;
        const bodyLength = Math.abs(ankleMidY - shoulderMidY);
        if (bodyLength > 50) {
          const hipExpectedY = (shoulderMidY + ankleMidY) / 2;
          const hipDeviation = hipMidY - hipExpectedY;
          if (Math.abs(hipDeviation) / bodyLength > 0.15) {
            const message = hipDeviation > 0 ? "엉덩이가 처지고 있어요" : "엉덩이가 너무 들렸어요";
            signals.push({ id: "hip_misalign", severity: "error", affectedKeypoints: [11, 12], message });
          }
        }

        // 3. 팔꿈치 벌어짐
        const shoulderWidth = Math.abs(kp[5].x - kp[6].x);
        const elbowWidth = Math.abs(kp[7].x - kp[8].x);
        if (shoulderWidth > 0 && elbowWidth > shoulderWidth * 1.4) {
          signals.push({ id: "elbow_flare", severity: "warning", affectedKeypoints: [7, 8], message: "팔꿈치를 몸통 가까이 유지하세요" });
        }

        // 4. 좌우 비대칭 (정면 카메라에서만)
        if (isFrontFacing(kp)) {
          const asymmetry = Math.abs(lElbow - rElbow);
          if (asymmetry > 15) {
            signals.push({ id: "elbow_asymmetry", severity: asymmetry > 25 ? "error" : "warning", affectedKeypoints: [7, 8, 9, 10], message: `좌우 팔꿈치 각도 차이 ${Math.round(asymmetry)}°`, value: asymmetry });
          }
        }
      }

      return { signals };
    },
    getDisplayAngle: (kp) => {
      const l = getAngle(kp[5], kp[7], kp[9]);
      const r = getAngle(kp[6], kp[8], kp[10]);
      return Math.round((l + r) / 2);
    },
    detect: (kp, phase) => {
      const l = getAngle(kp[5], kp[7], kp[9]);
      const r = getAngle(kp[6], kp[8], kp[10]);
      const avg = (l + r) / 2;
      if (avg < 90 && phase === "up") return { newPhase: "down", counted: false };
      if (avg > 150 && phase === "down") return { newPhase: "up", counted: true };
      return { newPhase: phase, counted: false };
    },
  },
];
