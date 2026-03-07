import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

type Props = {
  role: "user" | "assistant";
  text: string;
};

export default function MessageBubble({ role, text }: Props) {

  function copyText() {
    navigator.clipboard.writeText(text);
  }

  return (
    <div className={`message ${role}`}>
      <div className="bubble">

        {role === "assistant" && (
          <button className="copy-btn" onClick={copyText}>
            Copy
          </button>
        )}

        <ReactMarkdown
          components={{
            code({ className, children }) {
              const match = /language-(\w+)/.exec(className || "");

              if (match) {
                return (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  >
                    {String(children)}
                  </SyntaxHighlighter>
                );
              }

              return <code className="inline-code">{children}</code>;
            },
          }}
        >
          {text}
        </ReactMarkdown>

        {role === "assistant" && text === "" && (
          <span className="cursor">▌</span>
        )}

      </div>
    </div>
  );
}