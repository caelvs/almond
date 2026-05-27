import { EXERCISES } from "../exercises";
import "./HomeScreen.css";

function HomeScreen({ onSelect }) {
  return (
    <div className="home">
      <header className="home-header">
        <h1 className="home-logo">ALMOND</h1>
        <p className="home-subtitle">AI 홈트 자세 코치</p>
      </header>

      <main className="home-main">
        <p className="home-label">운동 선택</p>
        <div className="exercise-grid">
          {EXERCISES.map((ex) => (
            <button
              key={ex.id}
              className="exercise-card"
              style={{
                background: `linear-gradient(135deg, ${ex.gradientFrom}, ${ex.gradientTo})`,
              }}
              onClick={() => onSelect(ex)}
            >
              <span className="card-emoji">{ex.emoji}</span>
              <h3 className="card-name">{ex.name}</h3>
              <p className="card-desc">{ex.description}</p>
              <span className="card-muscles">{ex.muscles}</span>
              <span className="card-arrow">→</span>
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}

export default HomeScreen;
