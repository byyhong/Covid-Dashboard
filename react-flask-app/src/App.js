import LineGraph from "./components/LineGraph.jsx"
import React from 'react'
import Boxplot from "./components/Boxplot.jsx"
import {
  QueryClient,
  QueryClientProvider,
} from 'react-query'
import { ErrorBoundary } from "react-error-boundary";

function App() {
    const queryClient = new QueryClient();
  return (
    <ErrorBoundary fallback={<div>Something went wrong</div>}>
        <QueryClientProvider client={queryClient}>
            <div className="App">
                <LineGraph />
                <Boxplot />
            </div>
        </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
