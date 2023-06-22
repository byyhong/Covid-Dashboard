import { render, screen } from '@testing-library/react';
import App from './App';
import { ErrorBoundary } from "react-error-boundary";
import {rest} from 'msw'
import {setupServer} from 'msw/node'
import 'whatwg-fetch';
import DatePicker from 'react-datepicker'; // Import DatePicker

const server = setupServer(
    rest.get('http://localhost:5000/prediction',  async (req,res,ctx) => {
        return res(ctx.json([['2023-01-03', 70], ['2023-01-04', 80]]));

    }),
    rest.get('http://localhost:5000/prediction/:date',  async (req,res,ctx) => {
        const date =  await req.params.date;
        return res(ctx.json([[date, 70], [date, 80]]));

    }),
);

beforeAll(() => {
  // Start the mock server before running tests
  server.listen();
});
afterAll(() => {
  // Clean up the mock server after running tests

  server.close();
});
test('Test text', async () => {
    const {container} = render(
        <ErrorBoundary fallback={<div>Something went wrong</div>}>
            <App />
        </ErrorBoundary>
   )
    await fetch('http://localhost:5000/prediction/2023-01-03');
    expect(screen.getByText(/Predicted Death Cases Range:/i)).toBeInTheDocument();
    screen.debug(container, { maxDepth: 20 });
});

test('Test date-picker', async () => {
    const {container} = render(
        <ErrorBoundary fallback={<div>Something went wrong</div>}>
            <App />
        </ErrorBoundary>
   )

    await fetch('http://localhost:5000/prediction/2023-01-04');
    expect(screen.getByRole('textbox')).toHaveValue('01/03/2023');
});

