import React from "react";
import "../css/style.css";
import Navbar from "./Navbar";
export default function Yoga() {
  return (
    <>
      <Navbar />

      <p className="videoHeader">Follow your personal trainer!</p>
      <img
        className="videoBox1"
        src="http://127.0.0.1:5000/video_feed"
        alt="your camera"
      ></img>
      <img
        className="videoBox2"
        src="http://127.0.0.1:5000/stream"
        alt="trainer's camera"
      ></img>
      <p className="video1">Your camera</p>
      <p className="video2">Match this!!</p>
    </>
  );
}
