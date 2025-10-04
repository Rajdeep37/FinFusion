import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import topLevelAwait from 'vite-plugin-top-level-await';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    topLevelAwait({
      promiseExportName: "__tla",
      promiseImportName: "__tlaModule"
    }),
  ],
  // This is the key change to resolve the error.
  optimizeDeps: {
    exclude: ['pdf-parse'],
  },
});