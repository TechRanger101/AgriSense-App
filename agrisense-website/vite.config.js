import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import legacy from "@vitejs/plugin-legacy";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    legacy({
      targets: ["ie >= 11", "chrome >= 49", "safari >= 10"],
      additionalLegacyPolyfills: ["regenerator-runtime/runtime"],
    }),
  ],
  // build: {
  //   outDir: "dist",
  //   assetsDir: "",
  //   rollupOptions: {
  //     output: {
  //       entryFileNames: "[name].js",
  //       chunkFileNames: "[name].js",
  //       assetFileNames: "[name].[ext]",
  //     },
  //   },
  // },
});
