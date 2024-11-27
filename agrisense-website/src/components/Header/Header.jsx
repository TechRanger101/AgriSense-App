import { useEffect, useState } from "react";
import ThemeToggler from "../../components/Helper/ThemeToggler";

export default function Header() {
  const [sticky, setSticky] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);

  useEffect(() => {
    const _onScroll = function () {
      if (window.pageYOffset > 0 || window.scrollY > 0) {
        setSticky(true);
      } else {
        setSticky(false);
      }

      // Show/hide scroll to top button
      if (window.pageYOffset > 1000 || window.scrollY > 1000) {
        setShowScrollTop(true);
      } else {
        setShowScrollTop(false);
      }
    };

    window.addEventListener("scroll", _onScroll);

    return () => {
      window.removeEventListener("scroll", _onScroll);
    };
  }, []); // Empty dependency array ensures this useEffect only runs on mount/unmount

  const handleScrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div>
      <header
        className={`header absolute top-0 left-0 w-full ${
          sticky ? "sticky" : ""
        }`}
      >
        <div className="flex w-full flex-wrap px-5 lg:flex-nowrap lg:items-center lg:px-5 xl:px-10 2xl:px-20">
          <div className="relative z-[99] max-w-[200px] lg:w-full xl:max-w-[240px]">
            <a href="/" className="inline-block flex cursor-pointer">
              <img src="/images/logo.png" alt="logo" className="h-[50px]" />
              <p className="text-white text-xl leading-[50px] px-5 hidden dark:block">
                AgriSense
              </p>
              <p className="text-black text-xl leading-[50px] px-5 dark:hidden">
                AgriSense
              </p>
            </a>
          </div>

          <div className="menu-wrapper fixed top-0 left-0 z-50 h-screen w-full justify-center bg-white p-5 dark:bg-dark xl:visible xl:static xl:flex xl:h-auto xl:justify-start xl:bg-transparent xl:p-0 xl:opacity-100 dark:xl:bg-transparent">
            <div className="w-full self-center">
              <nav>
                <ul className="navbar flex flex-col items-center justify-center space-y-5 text-center lg:flex-row lg:justify-start lg:space-x-10 lg:space-y-0">
                  <li>
                    <a
                      href="/#home"
                      className="cursor-pointer menu-scroll inline-flex items-center justify-center text-center font-heading text-base text-dark-text hover:text-primary dark:hover:text-white"
                    >
                      Home
                    </a>
                  </li>
                  <li>
                    <a
                      href="/#api-guide"
                      className="cursor-pointer menu-scroll inline-flex items-center justify-center text-center font-heading text-base text-dark-text hover:text-primary dark:hover:text-white"
                    >
                      API Guide
                    </a>
                  </li>
                  {/* <li>
                    <a
                      href="/#pricing"
                      className="cursor-pointer menu-scroll inline-flex items-center justify-center text-center font-heading text-base text-dark-text hover:text-primary dark:hover:text-white"
                    >
                      Pricing
                    </a>
                  </li> */}
                  <li>
                    <a
                      href="/#contact"
                      className="cursor-pointer menu-scroll inline-flex items-center justify-center text-center font-heading text-base text-dark-text hover:text-primary dark:hover:text-white"
                    >
                      Contact
                    </a>
                  </li>
                </ul>
              </nav>
            </div>
          </div>

          <div className="flex w-full absolute top-1/2 right-5 z-50 flex -translate-y-1/2 items-center lg:static lg:translate-y-0">
            <div className="flex flex-wrap items-center justify-center lg:justify-start ml-auto mr-[20px]">
              <a
                href="https://docs.google.com/forms/d/e/1FAIpQLSdrnrzYqXkg_dKKCxyxYVoP8IFDDi5lKIx7z9sGAKUfRu1Crg/viewform"
                target="new"
                className="inline-flex items-center rounded bg-primary py-[10px] px-6 font-heading text-base text-white hover:opacity-80 transition duration-300 ease-in-out md:py-[14px] md:px-8"
              >
                <span>Sign Up</span>
                <span className="hidden md:block">
                  : Request an account to use API.
                </span>
              </a>
            </div>

            <ThemeToggler />
          </div>
        </div>
      </header>

      {/* Scroll to Top Button */}
      {showScrollTop && (
        <button
          onClick={handleScrollToTop}
          className="fixed bottom-10 right-10 bg-primary z-50 p-[10px] font-heading text-base text-white hover:bg-opacity-90 rounded-full transition"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
            className="w-6 h-6"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M5 15l7-7 7 7"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
