import "./LandingScreen.css";

const FEATURES = [
  {
    icon: "🎯",
    title: "다중 오류 동시 감지",
    desc: "어디가 왜 틀렸는지 실시간으로 정확히 짚어줍니다",
  },
  {
    icon: "📐",
    title: "좌우 비대칭 정량화",
    desc: "왼쪽 무릎 85°, 오른쪽 92° — 수치로 보여줍니다",
  },
  {
    icon: "🏠",
    title: "홈트 입문자 맞춤",
    desc: "별도 장비 없이 카메라 하나로 자세 교정을 받으세요",
  },
];

function LandingScreen({ onStart }) {
  return (
    <div className="landing">
      <div className="landing-hero">
        <p className="landing-eyebrow">AI 홈트 자세 코치</p>
        <h1 className="landing-logo">ALMOND</h1>
        <p className="landing-tagline">
          혼자 운동해도 자세가 흐트러지지 않도록.<br />
          AI가 실시간으로 잡아줍니다.
        </p>
      </div>

      <div className="landing-features">
        {FEATURES.map((f) => (
          <div key={f.title} className="landing-feature">
            <span className="landing-feature-icon">{f.icon}</span>
            <div>
              <p className="landing-feature-title">{f.title}</p>
              <p className="landing-feature-desc">{f.desc}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="landing-cta">
        <button className="landing-btn" onClick={onStart}>
          운동 시작하기
        </button>
        <p className="landing-cta-note">카메라 권한이 필요합니다</p>
      </div>
    </div>
  );
}

export default LandingScreen;
