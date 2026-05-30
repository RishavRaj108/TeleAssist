import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api calls to FastAPI during development so you can hit
    // /api/chat instead of http://localhost:8000/chat if preferred.
    // The React App.jsx directly calls localhost:8000 for simplicity.
  },
});
