import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import { ethers } from "ethers";
import faucetContract from "./ethereum/faucet";
import Rewards from "./components/Rewards";
import Home from "./components/Home";
import Exersise from "./components/Exersise";
import Yoga from "./components/Yoga";
import Navbar from "./components/Navbar";
function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <p className="line"></p>
      <Routes>
        <Route exact path="/" element={<Home />}></Route>
        <Route exact path="/yoga" element={<Yoga />}></Route>
        <Route exact path="/exersise" element={<Exersise />}></Route>
        <Route exact path="/rewards" element={<Rewards />}></Route>
      </Routes>
    </BrowserRouter>
  );
}
export default App;
