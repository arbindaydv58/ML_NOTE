import { useEffect, useRef } from "react";

export function useAutoScroll(messages: any[]) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return ref;
}