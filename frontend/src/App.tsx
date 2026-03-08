import { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import { askQuestion } from "./services/api";

type Message = {
  role: "user" | "assistant";
  text: string;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Ask anything from your ML study material." },
  ]);

  const [loading, setLoading] = useState(false);

  const session = "web";

  async function send(question: string) {
    // Add user + empty assistant message
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "" },
    ]);

    setLoading(true);

    try {
      await askQuestion(question, session, (token: string) => {
        setMessages((prev) => {
          const updated = [...prev];

          const lastIndex = updated.length - 1;

          // Append token to last assistant message
          updated[lastIndex] = {
            ...updated[lastIndex],
            text: updated[lastIndex].text + token,
          };

          return updated;
        });
      });
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];

        updated[updated.length - 1] = {
          role: "assistant",
          text: "Server error. Make sure FastAPI is running on port 8000.",
        };

        return updated;
      });
    }

    setLoading(false);
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">ML Notes Assistant</div>
        <div className="profile">AY</div>
      </header>

      <ChatWindow messages={messages} />

      <ChatInput onSend={send} loading={loading} />
    </div>
  );
}
