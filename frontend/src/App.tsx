import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Predict from './Predict'
import './App.css'

function Home() {
  return (
    <div className="container">
      <h1>House Price Regression</h1>
      <p>Welcome to the House Price Prediction System</p>
      <nav>
        <Link to="/predict">Predict</Link> | <Link to="/dashboard">Dashboard</Link>
      </nav>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/dashboard" element={<div className="container"><h1>Dashboard (Coming Day 5)</h1></div>} />
      </Routes>
    </Router>
  )
}

export default App
