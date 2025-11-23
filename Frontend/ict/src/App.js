import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import ChartPage from "./components/ChartPage";
// Example pages
function Home() {
  return (
    <div className="App-header">
      <h1>Welcome Home</h1>
      <p>This is the home page.</p>
      <Link className="App-link" to="/chart">Go to Chart</Link>
    </div>
  );
}



function NotFound() {
  return (
    <div className="App-header">
      <h1>404</h1>
      <p>Page not found</p>
      <Link className="App-link" to="/">Go Home</Link>
    </div>
  );
}

// Main App Component
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chart" element={<ChartPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default App;