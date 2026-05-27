import { useState, useEffect } from "react";
import "./App.css";
import LandingScreen from "./screens/LandingScreen";
import HomeScreen from "./screens/HomeScreen";
import ExerciseScreen from "./screens/ExerciseScreen";
import { EXERCISES } from "./exercises";

function App() {
  const [screen, setScreen] = useState("landing");
  const [exercise, setExercise] = useState(null);

  useEffect(() => {
    window.history.replaceState({ appPage: "landing" }, "");

    const handlePop = (e) => {
      const state = e.state;
      if (!state?.appPage) return;
      if (state.appPage === "landing") {
        setScreen("landing");
        setExercise(null);
      } else if (state.appPage === "home") {
        setScreen("home");
        setExercise(null);
      } else if (state.appPage === "exercise") {
        const ex = EXERCISES.find((ex) => ex.id === state.exerciseId);
        if (ex) setExercise(ex);
        else {
          setScreen("home");
          setExercise(null);
        }
      }
    };

    window.addEventListener("popstate", handlePop);
    return () => window.removeEventListener("popstate", handlePop);
  }, []);

  const goToHome = () => {
    window.history.pushState({ appPage: "home" }, "");
    setScreen("home");
  };

  const goToExercise = (ex) => {
    window.history.pushState({ appPage: "exercise", exerciseId: ex.id }, "");
    setExercise(ex);
  };

  const goBack = () => window.history.back();

  if (screen === "landing") {
    return <LandingScreen onStart={goToHome} />;
  }
  if (exercise) {
    return <ExerciseScreen exercise={exercise} onBack={goBack} />;
  }
  return <HomeScreen onSelect={goToExercise} />;
}

export default App;
