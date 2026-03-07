import { useState, FormEvent } from "react";

export default function ChatInput({
  onSend,
  loading,
}: {
  onSend: (text: string) => void;
  loading: boolean;
}) {
  const [input, setInput] = useState("");

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;

    onSend(input);
    setInput("");
  }

  return (
    <form className="chat-input" onSubmit={submit}>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask from your ML notes..."
      />

      <button disabled={loading}>
        {loading ? "Thinking..." : "Send"}
      </button>
    </form>
  );
}