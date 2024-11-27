import { createContext } from "react";

const MyThemeContext = createContext({
  themeClass: null,
  setThemeClass: null,
});

export { MyThemeContext };
