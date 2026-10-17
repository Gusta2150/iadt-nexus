import { Suspense } from "react";
import { Bench } from "@/components/bench";
import type { Overview } from "@/lib/types";

async function ReadyBench() {
  let initialOverview: Overview | null = null;
  let initialError: string | null = null;
  try {
    const response = await fetch("http://127.0.0.1:8741/api/overview", { cache: "no-store" });
    if (!response.ok) {
      initialError = "O serviço de modelos não respondeu.";
    } else {
      initialOverview = (await response.json()) as Overview;
    }
  } catch {
    initialError = "O serviço de modelos não respondeu. Confira se a API na porta 8741 está no ar.";
  }
  return <Bench initialOverview={initialOverview} initialError={initialError} />;
}

export default function Home() {
  return (
    <Suspense fallback={<p className="px-4 py-6 text-sm text-[#6a6158]">Carregando os modelos originais…</p>}>
      <ReadyBench />
    </Suspense>
  );
}
