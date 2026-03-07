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
      {messages.map((m, i) => (
        <MessageBubble key={i} role={m.role} text={m.text} />
      ))}
      <div ref={bottomRef}></div>
    </div>
  );
}