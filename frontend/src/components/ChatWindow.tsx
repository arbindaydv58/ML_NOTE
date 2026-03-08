import MessageBubble from "./MessageBubble";
import { useAutoScroll } from "../hooks/useAutoScroll";

type Message = {
  role: "user" | "assistant";
  text: string;
};

export default function ChatWindow({ messages }: { messages: Message[] }) {
  const bottomRef = useAutoScroll(messages);

  return (
    <div className="chat-window">

      {messages.length === 1 && (
        <div className="welcome">
          <h2>How can I help with your notes?</h2>
         <p>Ask questions related to your Machine Learning study materials.</p>
        </div>
      )}

      {messages.map((m, i) => (
        <MessageBubble key={i} role={m.role} text={m.text} />
      ))}

      <div ref={bottomRef}></div>

    </div>
  );
}