import React from "react";
import { Link } from "react-router-dom";
import "../css/style.css";

export default function Navbar() {
  return (
    <>
      <Link to="/">
        <p className="title">namaste crypto</p>
      </Link>
      <Link to="/">
        <p className="home">Home</p>
      </Link>
      <Link to="/yoga">
        <p className="yoga">Yoga</p>
      </Link>
      <Link to="/rewards">
        <button type="button" className="rewardsButton"></button>
        <div className="rewards">Rewards</div>
      </Link>
    </>
  );
}
