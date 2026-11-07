import { defineConfig } from "vite";

// Built into ../docs, which GitHub Pages serves from main:/docs. docs/data is written by
// scripts/04_export_dashboard.py and is NOT touched by the build (emptyOutDir: false).
export default defineConfig({
  base: "./",
  build: { outDir: "../docs", emptyOutDir: false, assetsDir: "assets", chunkSizeWarningLimit: 8000 },
  worker: { format: "es" },
});
