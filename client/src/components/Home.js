import React from "react";
import yoga from "../images/1.png";
import { Link } from "react-router-dom";
export default function Home() {
  return (
    <>
      <p className="aboutPro">About Project</p>
      <p className="smolline"></p>
      <p className="para">
        Namastecrypto is a platform that allows users to learn and practice yoga
        poses and earn rewards in the form of our own cryptocurrency, Elixirium
        (EXR), using Mediapipe posture detection and OpenCV. By accurately
        performing yoga postures, users are eligible for rewards which are
        stored on the blockchain using smart contracts. Users can visit the
        decentralized application (DApp) to claim their rewards by connecting
        their wallet. The system preserves privacy by not associating personal
        identity with user data and all transactions are conducted using
        pseudonymous blockchain addresses.
      </p>
      <Link to="/yoga">
        <button className="startBut"></button>
        <p className="startTxt">Start Exercising</p>
      </Link>
      <img className="img1" src={yoga} alt="imag" />
    </>
  );
}
