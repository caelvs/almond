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
    getPhaseLabel: (p) => (p === "up" ? "준비" : "운동 중"),
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
    id: "lunge",
    name: "런지",
    emoji: "🦵",
    description: "한 발씩 앞으로 내딛어 하체를 강화하는 운동",
    muscles: "허벅지 · 힙 플렉서 · 코어",
    gradientFrom: "#059669",
    gradientTo: "#0891b2",
    initialPhase: "up",
    feedbackKeypoints: [11, 12, 13, 14, 15, 16],
    getPhaseLabel: (p) => (p === "up" ? "준비" : "운동 중"),
    getDisplayAngle: (kp) => {
      const l = getAngle(kp[11], kp[13], kp[15]);
      const r = getAngle(kp[12], kp[14], kp[16]);
      return Math.round(Math.min(l, r));
    },
    isGoodPose: (angle) => angle > 80 && angle < 165,
    detect: (kp, phase) => {
      const l = getAngle(kp[11], kp[13], kp[15]);
      const r = getAngle(kp[12], kp[14], kp[16]);
      const min = Math.min(l, r);
      if (min < 100 && phase === "up") return { newPhase: "down", counted: false };
      if (min > 155 && phase === "down") return { newPhase: "up", counted: true };
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
    getPhaseLabel: (p) => (p === "down" ? "준비" : "운동 중"),
    getDisplayAngle: (kp) => {
      // wrist - shoulder - hip angle measures how high arms are raised
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
  {
    id: "bicep-curl",
    name: "바이셉 컬",
    emoji: "💪",
    description: "팔꿈치를 굽혀 이두근을 강화하는 운동",
    muscles: "이두근 · 전완근",
    gradientFrom: "#db2777",
    gradientTo: "#9333ea",
    initialPhase: "down",
    feedbackKeypoints: [5, 6, 7, 8, 9, 10],
    getPhaseLabel: (p) => (p === "down" ? "준비" : "운동 중"),
    getDisplayAngle: (kp) => {
      // shoulder - elbow - wrist angle
      const l = getAngle(kp[5], kp[7], kp[9]);
      const r = getAngle(kp[6], kp[8], kp[10]);
      return Math.round((l + r) / 2);
    },
    isGoodPose: (angle) => angle > 40 && angle < 170,
    detect: (kp, phase) => {
      const l = getAngle(kp[5], kp[7], kp[9]);
      const r = getAngle(kp[6], kp[8], kp[10]);
      const avg = (l + r) / 2;
      if (avg < 60 && phase === "down") return { newPhase: "up", counted: false };
      if (avg > 150 && phase === "up") return { newPhase: "down", counted: true };
      return { newPhase: phase, counted: false };
    },
  },
];
