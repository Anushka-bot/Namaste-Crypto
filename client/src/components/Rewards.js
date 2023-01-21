import { useEffect, useState } from "react";
// import "../css/rewards.css";
import { ethers } from "ethers";
import faucetContract from "../ethereum/faucet";
import axios from "axios";

export default function Rewards() {
  const [walletAddress, setWalletAddress] = useState("");
  const [signer, setSigner] = useState();
  const [fcContract, setFcContract] = useState();
  const [withdrawError, setWithdrawError] = useState("");
  const [withdrawSuccess, setWithdrawSuccess] = useState("");
  const [transactionData, setTransactionData] = useState("");
  const [rewards, setRewards] = useState(3);
  useEffect(() => {
    getCurrentWalletConnected();
    addWalletListener();
  }, [walletAddress]);

  useEffect(() => {
    if (rewards !== 0) {
      axios("/rewards")
        .then((res) => setRewards(res.data.class))
        .catch((err) => console.log(err));
    }
  });

  const connectWallet = async () => {
    if (typeof window != "undefined" && typeof window.ethereum != "undefined") {
      try {
        /* get provider */
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        /* get accounts */
        const accounts = await provider.send("eth_requestAccounts", []);
        /* get signer */
        setSigner(provider.getSigner());
        /* local contract instance */
        setFcContract(faucetContract(provider));
        /* set active wallet address */
        setWalletAddress(accounts[0]);
      } catch (err) {
        console.error(err.message);
      }
    } else {
      /* MetaMask is not installed */
      console.log("Please install MetaMask");
    }
  };

  const getCurrentWalletConnected = async () => {
    if (typeof window != "undefined" && typeof window.ethereum != "undefined") {
      try {
        /* get provider */
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        /* get accounts */
        const accounts = await provider.send("eth_accounts", []);
        if (accounts.length > 0) {
          /* get signer */
          setSigner(provider.getSigner());
          /* local contract instance */
          setFcContract(faucetContract(provider));
          /* set active wallet address */
          setWalletAddress(accounts[0]);
        } else {
          console.log("Connect to MetaMask using the Connect Wallet button");
        }
      } catch (err) {
        console.error(err.message);
      }
    } else {
      /* MetaMask is not installed */
      console.log("Please install MetaMask");
    }
  };

  const addWalletListener = async () => {
    if (typeof window != "undefined" && typeof window.ethereum != "undefined") {
      window.ethereum.on("accountsChanged", (accounts) => {
        setWalletAddress(accounts[0]);
      });
    } else {
      /* MetaMask is not installed */
      setWalletAddress("");
      console.log("Please install MetaMask");
    }
  };

  const getOCTHandler = async () => {
    setWithdrawError("");
    setWithdrawSuccess("");
    try {
      const fcContractWithSigner = fcContract.connect(signer);
      const resp = await fcContractWithSigner.requestTokens(rewards);
      setWithdrawSuccess("Operation succeeded - enjoy your tokens!");
      setTransactionData(resp.hash);
      setRewards(0);
    } catch (err) {
      setWithdrawError(err.message);
    }
  };

  return (
    // <div>
    //   <nav className="navbar">
    //     <div className="container">
    //       <div className="navbar-brand">
    //         <h1 className="navbar-item is-size-4">Ocean Token (OCT)</h1>
    //       </div>
    //       <div id="navbarMenu" className="navbar-menu">
    //         <div className="navbar-end is-align-items-center">
    //           <button
    //             className="button is-white connect-wallet"
    //             onClick={connectWallet}
    //           >
    //             <span className="is-link has-text-weight-bold">
    //               {walletAddress && walletAddress.length > 0
    //                 ? `Connected: ${walletAddress.substring(
    //                     0,
    //                     6
    //                   )}...${walletAddress.substring(38)}`
    //                 : "Connect Wallet"}
    //             </span>
    //           </button>
    //         </div>
    //       </div>
    //     </div>
    //   </nav>
    //   <section className="hero is-fullheight">
    //     <div className="faucet-hero-body">
    //       <div className="container has-text-centered main-content">
    //         <h1 className="title is-1">Faucet</h1>
    //         <p>Fast and reliable. 50 OCT/day.</p>
    //         <div className="mt-5">
    //           {withdrawError && (
    //             <div className="withdraw-error">{withdrawError}</div>
    //           )}
    //           {withdrawSuccess && (
    //             <div className="withdraw-success">{withdrawSuccess}</div>
    //           )}{" "}
    //         </div>
    //         <div className="box address-box">
    //           <div className="columns">
    //             <div className="column is-four-fifths">
    //               {/* <input
    //                   className="input is-medium"
    //                   type="text"
    //                   placeholder="Enter your wallet address (0x...)"
    //                   defaultValue={walletAddress}
    //                 /> */}
    //               <input
    //                 className="input is-medium"
    //                 type="text"
    //                 placeholder=""
    //                 value={rewards}
    //                 onChange={(e) => {
    //                   setRewards(e.target.value);
    //                 }}
    //               />
    //             </div>
    //             <div className="column">
    //               <button
    //                 className="button is-link is-medium"
    //                 onClick={getOCTHandler}
    //                 disabled={walletAddress ? false : true}
    //               >
    //                 GET TOKENS
    //               </button>
    //             </div>
    //           </div>
    //           <article className="panel is-grey-darker">
    //             <p className="panel-heading">Transaction Data</p>
    //             <div className="panel-block">
    //               <p>
    //                 {transactionData
    //                   ? `Transaction hash: ${transactionData}`
    //                   : "--"}
    //               </p>
    //             </div>
    //           </article>
    //         </div>
    //       </div>
    //     </div>
    //   </section>
    // </div>
    <>
      <p className="rewardsHeader">Collect your crypto rewards</p>
      <p className="rewardBox">{rewards}</p>
      <button type="button" className="connectBox"></button>
      <div className="connect" onClick={connectWallet}>
        {" "}
        {walletAddress && walletAddress.length > 0
          ? `  Connected`
          : "Connect Wallet"}
      </div>
      <button type="button" className="collectBox"></button>
      <div
        className="collect"
        onClick={getOCTHandler}
        disabled={walletAddress ? false : true}
      >
        Collect
      </div>

      {/* <button type="button" className="btn btn-primary" onClick={() => {toast('Hello Geeks')}}>
      Show live toast
    </button> */}
    </>
  );
}