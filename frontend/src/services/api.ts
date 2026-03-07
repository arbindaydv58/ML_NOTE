const API_BASE = "http://127.0.0.1:8000";

export async function askQuestion(
  question: string,
  session: string,
  onToken: (token: string) => void
) {
  const url = new URL(`${API_BASE}/ask_stream`);
  url.searchParams.set("question", question);
  url.searchParams.set("session", session);

  const response = await fetch(url.toString());

  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    const chunk = decoder.decode(value);

    // send each token immediately
    onToken(chunk);
  }
}