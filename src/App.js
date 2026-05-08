import { useState } from "react";
import "./App.css";
import HomeScreen from "./screens/HomeScreen";
import ExerciseScreen from "./screens/ExerciseScreen";

function App() {
  const [exercise, setExercise] = useState(null);

  if (exercise) {
    return <ExerciseScreen exercise={exercise} onBack={() => setExercise(null)} />;
  }
  return <HomeScreen onSelect={setExercise} />;
}

export default App;
