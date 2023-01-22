# Namaste-Crypto
Namaste Crypto is a platform that allows users to learn and practice yoga poses and earn rewards in the form of our own cryptocurrency, Elixirium (EXR), using Mediapipe posture detection and OpenCV. By accurately performing yoga postures, users are eligible for rewards which are stored on the blockchain using smart contracts. Users can visit the decentralized application (DApp) to claim their rewards by connecting their wallet. The system preserves privacy by not associating personal identity with user data and all transactions are conducted using pseudonymous blockchain addresses.

## Check out our [Video Demo]([https://pages.github.com/](https://youtu.be/P6UTE-mbK8A)).

![Screenshot (7)](https://user-images.githubusercontent.com/75663460/213910487-4a8338e7-4125-4650-b56a-b640b7b3fcc6.png)

# ✨Inspiration for the Project Name

As this hackathon centers around fitness, yoga plays a significant role. As an APAC developer, particularly from India, and given yoga's origin in India, we thought it fitting to infuse our project with an Indian touch. The inclusion of "Namaste" in the project name is a greeting in Hindi language, which is traditionally used to show respect to those older than oneself. In addition, in the video stream, we specifically demonstrate the practice of "Surya Namaskar," a well-known yoga practice. The inclusion of "crypto" in our project name also serves as a subtle reference to cryptocurrency.


# 💡Inspiration
Yoga offers numerous health benefits to people of all ages and fitness levels. It improves overall health and fitness. However, in today's fast-paced world, finding the motivation to practice yoga daily can be difficult. Learning yoga has traditionally been exhaustive, and the rewards for doing it daily are often intangible. We identified a need for a strong motivational factor to attract more people to practice yoga. Studies show that incentive programs can positively impact motivation, particularly for physical activities. We decided to incorporate the latest buzz in technology- cryptocurrency, blockchain, and machine learning- to make the process of learning yoga more engaging. Imagine learning yoga correctly with guidance and earning crypto rewards for doing it correctly and improving fitness. Achieving fitness goals is challenging and requires commitment, but our application makes it fun and easier to achieve.

# 🔧 How we built it
The project combines Deep Learning's computer vision and Blockchain technologies. We built an application that checks the correctness of yoga postures by comparing them to an expert's pose and informing the user of the results. We connected the video feeds to our website and used Mediapipe's posture detection model to evaluate the user's posture. We also created our own cryptocurrency, Elixirium (EXR), using ERC20 smart contract and Solidity, deploying it on the Goerli test network. For every correct pose, users are awarded 5 EXR tokens. We also built a dapp using web3 client Ether.js where users can collect their rewards by connecting their Metamask wallet. We designed a user-friendly website using React JS and Figma.

# 🏃 Challenges we ran into
Creating a posture detector using Mediapipe's full body posture detector required extensive testing and experimentation. The main challenge was ensuring smooth video performance while processing and comparing each frame with an ML model. We also faced difficulty connecting the video feeds to our website but overcame it using Flask and yield return statement. As it was our first time using ML and blockchain, identifying and solving issues took longer. We also implemented ERC20 contract for our coin and product and deployed it on Goerli test network, and integrating it with our dapp and Metamask wallet took a while.

# ⭐ Accomplishments that we're proud of
We are extremely proud to have successfully developed a working prototype within the limited timeframe of 48 hours, as our concept was ambitious and challenging to execute. We take great satisfaction in having created a project that aims to motivate individuals to prioritize their health and regularly practice yoga. It is well-known that a lack of physical activity can have detrimental effects on one's well-being and we are delighted that we were able to develop a solution to address this issue.

# 📝 What we learned
We learned trending tech stacks like blockchain, deep learning, and web3, also took UI/UX inspiration from MidJourney and ChatGPT for error corrections.

# 🎯 What's next for Namaste Crypto
We plan to add more Yoga poses in Namaste-crypto and aim to improve the accuracy of the ML model.
