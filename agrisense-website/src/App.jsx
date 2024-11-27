import { useEffect, useState } from "react";
import ReactGA from "react-ga4";

import Header from "./components/Header/Header";
import Footer from "./components/Footer/Footer";

import CardHero from "./components/HomeCards/CardHero";
import CardAPIGuide from "./components/HomeCards/CardAPIGuide";
// import CardPricing from "./components/HomeCards/CardPricing";

import { MyThemeContext } from "./provider/ThemeContext";

import "./assets/styles/scrollbar.css";
import "./assets/styles/markdown-container.css";
import "./assets/styles/index.css";

ReactGA.initialize("G-D4RB4X5VGJ");

export default function App() {
  const [themeClass, setThemeClass] = useState("dark");

  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: window.location.pathname });
  }, []);

  return (
    <MyThemeContext.Provider value={{ themeClass, setThemeClass }}>
      <main className={themeClass}>
        <div className="dark:bg-dark ">
          <Header />
          <CardHero />
          <CardAPIGuide />
          {/* <CardPricing /> */}
          <Footer />
        </div>
      </main>
    </MyThemeContext.Provider>
  );
}
