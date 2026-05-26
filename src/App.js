import { useState } from "react";
import "./App.css";
import LandingScreen from "./screens/LandingScreen";
import HomeScreen from "./screens/HomeScreen";
import ExerciseScreen from "./screens/ExerciseScreen";

function App() {
  const [screen, setScreen] = useState("landing");
  const [exercise, setExercise] = useState(null);

  if (screen === "landing") {
    return <LandingScreen onStart={() => setScreen("home")} />;
  }
  if (exercise) {
    return <ExerciseScreen exercise={exercise} onBack={() => setExercise(null)} />;
  }
  return <HomeScreen onSelect={setExercise} />;
}

export default App;
