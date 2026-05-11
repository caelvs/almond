function getAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag =
    Math.sqrt(ab.x ** 2 + ab.y ** 2) * Math.sqrt(cb.x ** 2 + cb.y ** 2);
  if (mag === 0) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
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
    getDetailedFeedback: (angle, phase) => {
      if (phase === "up") {
        if (angle < 160) return "무릎을 완전히 펴세요";
        return "내려가세요!";
      }
      if (angle > 130) return "더 깊이 내려가세요";
      return "올라가세요!";
    },
    getDisplayAngle: (kp) => {
      const l = getAngle(kp[11], kp[13], kp[15]);
      const r = getAngle(kp[12], kp[14], kp[16]);
      return Math.round((l + r) / 2);
    },
    isGoodPose: (angle) => angle > 80 && angle < 170,
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
    id: "lateral-raise",
    name: "레터럴 레이즈",
    emoji: "🙆",
    description: "팔을 옆으로 들어 어깨 측면을 강화하는 운동",
    muscles: "삼각근 · 어깨",
    gradientFrom: "#d97706",
    gradientTo: "#dc2626",
    initialPhase: "down",
    feedbackKeypoints: [5, 6, 7, 8, 9, 10],
    guidePoints: [
      "양발을 어깨 너비로 벌리고 서세요",
      "팔꿈치를 약간 구부린 상태를 유지하세요",
      "팔을 어깨 높이까지만 올리세요",
      "천천히 조절하며 내리세요",
    ],
    getPhaseLabel: (p) => (p === "down" ? "준비" : "운동 중"),
    getDetailedFeedback: (angle, phase) => {
      if (phase === "down") return "팔을 올려보세요!";
      if (angle > 100) return "너무 높이 올렸어요";
      return "내려가세요!";
    },
    getDisplayAngle: (kp) => {
      const l = getAngle(kp[9], kp[5], kp[11]);
      const r = getAngle(kp[10], kp[6], kp[12]);
      return Math.round((l + r) / 2);
    },
    isGoodPose: (angle) => angle > 20 && angle < 110,
    detect: (kp, phase) => {
      const l = getAngle(kp[9], kp[5], kp[11]);
      const r = getAngle(kp[10], kp[6], kp[12]);
      const avg = (l + r) / 2;
      if (avg > 70 && phase === "down") return { newPhase: "up", counted: false };
      if (avg < 30 && phase === "up") return { newPhase: "down", counted: true };
      return { newPhase: phase, counted: false };
    },
  },
];
