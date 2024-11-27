// const plugin = require('tailwindcss/plugin')

module.exports = {
  content: ["./src/*.{js,jsx}", "./src/**/*.{js,jsx}", "./public/index.html"],
  darkMode: "class",
  theme: {
    extend: {
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            "--tw-prose-headings": theme("colors.gray.300"),
            "--tw-prose-body": theme("colors.gray.300"),
            // h2: {
            //     color: theme("colors.gray.300")
            // },
            // h3: {
            //     color: theme("colors.gray.300")
            // },
            // h4: {
            //     color: theme("colors.gray.300")
            // },
            // p: {
            //     color: theme("colors.gray.300")
            // },
            a: {
              color: theme("colors.header"),
              textDecoration: "none",
              "&:hover": {
                textDecoration: "underline",
              },
            },
          },
        },
        slate: {
          css: {
            a: {
              color: theme("colors.accent"),
              textDecoration: "none",
              "&:hover": {
                textDecoration: "underline",
              },
            },
          },
        },
      }),
      screens: {
        sm: "540px",
        // => @media (min-width: 576px) { ... }

        md: "768px",
        // => @media (min-width: 768px) { ... }

        lg: "992px",
        // => @media (min-width: 992px) { ... }

        xl: "1140px",
        // => @media (min-width: 1200px) { ... }

        "2xl": "1320px",
        // => @media (min-width: 1400px) { ... }
      },
      colors: {
        current: "currentColor",
        transparent: "transparent",
        white: "#FFFFFF",
        primary: "#4A6CF7",
        "dark-text": "#79808A",
        dark: "#111722",
        stroke: "#e5e7eb",
      },
      fontFamily: {
        heading: "'Lexend', sans-serif",
        body: "'Inter', sans-serif",
      },
      container: {
        center: true,
        padding: "1rem",
      },
    },
  },

  plugins: [require("@tailwindcss/typography")],
};
