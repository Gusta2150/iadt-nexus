"use client";

import { useEffect, useState } from "react";
import { explainCase, explainReport, getCases, getLogs, getMetrics, getOverview, optimize } from "@/lib/api";
import { fitness, measure, pct, points, seconds } from "@/lib/format";
import type { CaseRow, Crossover, LogEvent, Metrics, ModelId, Narrative, OptimizeResponse, Overview, RunSummary } from "@/lib/types";
import { Convergence, Meter } from "@/components/charts";
import { MethodDialog } from "@/components/method-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

type Pane = "ajuste" | "resultados" | "caso";
type CaseFilter = "todos" | "divergentes" | "malignos";

const MODELS: ModelId[] = ["knn", "logistica", "arvore", "floresta"];

export function Bench({
  initialOverview = null,
  initialError = null,
}: {
  initialOverview?: Overview | null;
  initialError?: string | null;
}) {
  const [overview, setOverview] = useState<Overview | null>(initialOverview);
  const [loadError, setLoadError] = useState<string | null>(initialError);
  const [pane, setPane] = useState<Pane>("ajuste");
  const [model, setModel] = useState<ModelId>("knn");
  const [population, setPopulation] = useState(18);
  const [generations, setGenerations] = useState(12);
  const [mutation, setMutation] = useState(0.18);
  const [crossover, setCrossover] = useState<Crossover>("mistura");
  const [seed, setSeed] = useState(42);
  const [experiments, setExperiments] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [selected, setSelected] = useState(0);
  const [filter, setFilter] = useState<CaseFilter>("todos");
  const [reading, setReading] = useState<Narrative | null>(null);
  const [readingBusy, setReadingBusy] = useState(false);
  const [report, setReport] = useState<Narrative | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [solvedSignature, setSolvedSignature] = useState<string | null>(null);

  const signature = JSON.stringify({ model, population, generations, mutation, crossover, seed, experiments });

  useEffect(() => {
    if (initialOverview) return;
    let alive = true;
    getOverview()
      .then((data) => {
        if (alive) setOverview(data);
      })
      .catch((error: Error) => {
        if (alive) setLoadError(error.message);
      });
    return () => {
      alive = false;
    };
  }, [initialOverview]);

  const original = result?.runs.find((item) => item.label === "Modelo original");
  const chosen = result?.runs.find((item) => item.label === result.chosen_label);
  const visibleCases = cases.filter((item) => {
    if (filter === "divergentes") return !item.correct;
    if (filter === "malignos") return item.predicted === "maligno";
    return true;
  });
  const currentCase = cases.find((item) => item.index === selected) ?? visibleCases[0];
  const stale = Boolean(result && solvedSignature && solvedSignature !== signature);
  const groups = groupFeatures(currentCase);

  function chooseModel(next: ModelId) {
    setModel(next);
    if (next === "floresta") {
      setPopulation(10);
      setGenerations(6);
      return;
    }
    if (model === "floresta") {
      setPopulation(18);
      setGenerations(12);
    }
  }

  async function runSearch() {
    setRunning(true);
    setRunError(null);
    setReading(null);
    setReport(null);
    try {
      const response = await optimize({
        model,
        population_size: population,
        generations,
        mutation_rate: mutation,
        crossover,
        seed,
        experiments,
      });
      const loaded = await getCases(response.run_id);
      setResult(response);
      setCases(loaded.cases);
      setSelected(loaded.cases[0]?.index ?? 0);
      setSolvedSignature(signature);
      setPane("resultados");
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "A busca não terminou.");
    } finally {
      setRunning(false);
    }
  }

  async function loadReading(index: number) {
    if (!result) return;
    setReadingBusy(true);
    setRunError(null);
    try {
      setReading(await explainCase(result.run_id, index));
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "A leitura não foi gerada.");
    } finally {
      setReadingBusy(false);
    }
  }

  async function loadReport() {
    if (!result) return;
    setReportBusy(true);
    setRunError(null);
    try {
      setReport(await explainReport(result.run_id));
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "O relatório não foi gerado.");
    } finally {
      setReportBusy(false);
    }
  }

  async function loadMonitor() {
    const [nextLogs, nextMetrics] = await Promise.all([getLogs(), getMetrics()]);
    setLogs(nextLogs.events);
    setMetrics(nextMetrics);
  }

  const durationHint =
    model === "floresta"
      ? experiments
        ? "A floresta com três configurações pode levar cerca de meio minuto."
        : "A floresta leva cerca de 15 segundos: cada indivíduo treina várias árvores."
      : experiments
        ? "As três configurações levam alguns segundos."
        : "Esta busca leva cerca de dois segundos.";

  return (
    <div className="min-h-dvh">
      <header className="border-b border-border bg-[#fffcf8]">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-primary font-serif text-lg text-primary-foreground">
              A
            </div>
            <div>
              <p className="font-serif text-xl leading-none tracking-tight">Acuria</p>
              <p className="mt-1 text-xs text-muted-foreground">Hospital universitário · bancada de modelos</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <MethodDialog />
            <a href="/solver/docs" className="hidden text-sm text-muted-foreground underline-offset-4 hover:underline sm:inline">
              API
            </a>
          </div>
        </div>
      </header>
      <p className="bg-[#f6efe6] px-4 py-2 text-center text-xs leading-5 text-[#6d5648]">
        Ensino no conjunto público WDBC, 569 exames de núcleo celular. Não usar para decisão clínica.
      </p>

      <div className="mx-auto flex max-w-[1440px] gap-2 px-4 py-3 lg:hidden">
        {(["ajuste", "resultados", "caso"] as Pane[]).map((item) => (
          <Button
            key={item}
            type="button"
            size="sm"
            variant={pane === item ? "default" : "outline"}
            className="flex-1"
            onClick={() => setPane(item)}
          >
            {item === "ajuste" ? "Ajuste" : item === "resultados" ? "Resultados" : "Caso"}
          </Button>
        ))}
      </div>

      {loadError && (
        <div className="mx-auto max-w-[1440px] px-4">
          <div className="rounded-lg border border-destructive/30 bg-[#fdf2f0] px-3 py-2 text-sm text-destructive">
            {loadError} Confira se a API na porta 8741 está no ar.
          </div>
        </div>
      )}

      <main className="mx-auto grid max-w-[1440px] gap-4 px-4 pb-8 lg:grid-cols-[300px_minmax(0,1fr)_390px] lg:pt-4">
        <section className={cn("flex flex-col gap-4", pane !== "ajuste" && "hidden lg:flex")}>
          <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
            <p className="text-xs tracking-wide text-muted-foreground uppercase">Modelo</p>
            <div className="mt-3 grid gap-2">
              {MODELS.map((id) => {
                const item = overview?.baselines.find((candidate) => candidate.model === id);
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => chooseModel(id)}
                    className={cn(
                      "rounded-lg border px-3 py-2 text-left transition-colors",
                      model === id ? "border-primary bg-[#e7f3ef]" : "border-border bg-transparent hover:bg-muted",
                    )}
                  >
                    <span className="block text-sm font-medium">{item?.label ?? labelOf(id)}</span>
                    {item?.blurb ? (
                      <span className="mt-0.5 block text-xs leading-5 text-muted-foreground">{item.blurb}</span>
                    ) : null}
                    {item?.test && (
                      <span className="mt-1 block text-xs text-foreground">
                        Teste original: F1 {pct(item.test.f1)} · recall {pct(item.test.recall)}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
            <p className="text-xs tracking-wide text-muted-foreground uppercase">Algoritmo genético</p>
            <div className="mt-3 space-y-3">
              <Slider label="População" value={population} min={6} max={28} disabled={experiments || running} onChange={setPopulation} />
              <Slider label="Gerações" value={generations} min={2} max={20} disabled={experiments || running} onChange={setGenerations} />
              <Slider
                label="Mutação por gene"
                value={Math.round(mutation * 100)}
                min={2}
                max={60}
                suffix="%"
                disabled={experiments || running}
                onChange={(value) => setMutation(value / 100)}
              />
              <div className="block text-xs text-muted-foreground">
                Cruzamento
                <Select
                  value={crossover}
                  disabled={experiments || running}
                  items={[
                    { value: "mistura", label: "Mistura" },
                    { value: "uniforme", label: "Uniforme" },
                    { value: "um_ponto", label: "Um ponto" },
                  ]}
                  onValueChange={(value) => {
                    if (value === "mistura" || value === "uniforme" || value === "um_ponto") setCrossover(value);
                  }}
                >
                  <SelectTrigger className="mt-1 w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mistura">Mistura</SelectItem>
                    <SelectItem value="uniforme">Uniforme</SelectItem>
                    <SelectItem value="um_ponto">Um ponto</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <label className="block text-xs text-muted-foreground">
                Semente
                <Input
                  type="number"
                  min={0}
                  value={seed}
                  disabled={running}
                  onChange={(event) => setSeed(Number(event.target.value) || 0)}
                  className="mt-1"
                />
              </label>
              <div className="flex items-center justify-between gap-3 pt-1">
                <div>
                  <p className="text-sm">Três configurações</p>
                  <p className="text-xs text-muted-foreground">Exploratória, balanceada e intensiva.</p>
                </div>
                <Switch checked={experiments} disabled={running} onCheckedChange={setExperiments} />
              </div>
            </div>
            <p className="mt-3 text-xs leading-5 text-muted-foreground">{durationHint}</p>
            <Button type="button" className="mt-3 h-10 w-full" disabled={running || !overview} onClick={runSearch}>
              {running ? "Treinando a população…" : experiments ? "Comparar 3 configurações" : "Otimizar hiperparâmetros"}
            </Button>
            {runError && <p className="mt-2 text-xs leading-5 text-destructive">{runError}</p>}
          </div>

          {overview && (
            <p className="px-1 text-xs leading-5 text-muted-foreground">
              {overview.samples} casos, {overview.malignant} malignos e {overview.benign} benignos. Recorte:{" "}
              {overview.split.train} treino, {overview.split.validation} validação, {overview.split.test} teste.
            </p>
          )}
        </section>

        <section className={cn("min-w-0", pane !== "resultados" && "hidden lg:block")}>
          {!overview && !loadError && <Skeleton />}
          {overview && !result && <BaselineBoard overview={overview} active={model} />}
          {result && original && chosen && (
            <div className="space-y-4">
              {stale && (
                <p className="rounded-lg bg-[#f6efe6] px-3 py-2 text-sm text-[#6d5648]">
                  Você mudou o ajuste depois desta busca. Otimize de novo para atualizar os números.
                </p>
              )}
              <div className="grid gap-3 sm:grid-cols-4">
                <Stat label="Fitness na validação" value={fitness(chosen.validation.fitness)} detail={points(chosen.validation.fitness, original.validation.fitness)} />
                <Stat label="Recall no teste" value={pct(chosen.test.recall)} detail={points(chosen.test.recall, original.test.recall)} />
                <Stat label="F1 no teste" value={pct(chosen.test.f1)} detail={points(chosen.test.f1, original.test.f1)} />
                <Stat label="Avaliações" value={String(result.evaluations)} detail={seconds(result.elapsed_ms)} />
              </div>
              <p className="text-sm leading-6 text-muted-foreground">{transferSentence(original, chosen)}</p>

              <Tabs
                defaultValue="comparar"
                onValueChange={(value) => {
                  if (value === "registros") void loadMonitor();
                }}
              >
                <TabsList>
                  <TabsTrigger value="comparar">Comparar</TabsTrigger>
                  <TabsTrigger value="curva">Curva</TabsTrigger>
                  <TabsTrigger value="registros">Registros</TabsTrigger>
                </TabsList>
                <TabsContent value="comparar" className="mt-3 space-y-4">
                  <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div>
                        <h2 className="font-serif text-lg">{result.model_label}</h2>
                        <p className="text-xs text-muted-foreground">
                          Destaque: {result.chosen_label}. Cinza é o modelo original, verde é a busca.
                        </p>
                      </div>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <Meter label="Recall de malignidade no teste" original={original.test.recall} evolved={chosen.test.recall} />
                      <Meter label="Precisão no teste" original={original.test.precision} evolved={chosen.test.precision} />
                      <Meter label="F1 no teste" original={original.test.f1} evolved={chosen.test.f1} />
                      <Meter label="Acurácia no teste" original={original.test.accuracy} evolved={chosen.test.accuracy} />
                    </div>
                    <ul className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
                      {chosen.params.map((param) => (
                        <li key={param.label} className="flex justify-between gap-3 border-b border-border py-1">
                          <span className="text-muted-foreground">{param.label}</span>
                          <span>{param.value}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="overflow-x-auto rounded-xl bg-card ring-1 ring-foreground/10">
                    <table className="w-full min-w-[640px] text-left text-sm">
                      <thead className="text-xs text-muted-foreground">
                        <tr>
                          <th className="px-3 py-2 font-medium">Configuração</th>
                          <th className="px-3 py-2 font-medium">Fitness val.</th>
                          <th className="px-3 py-2 font-medium">Recall teste</th>
                          <th className="px-3 py-2 font-medium">Precisão teste</th>
                          <th className="px-3 py-2 font-medium">F1 teste</th>
                          <th className="px-3 py-2 font-medium">Avaliações</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.runs.map((run) => (
                          <tr key={run.label} className={run.label === result.chosen_label ? "bg-[#e7f3ef]" : undefined}>
                            <td className="px-3 py-2">{run.label}</td>
                            <td className="px-3 py-2">{fitness(run.validation.fitness)}</td>
                            <td className="px-3 py-2">{pct(run.test.recall)}</td>
                            <td className="px-3 py-2">{pct(run.test.precision)}</td>
                            <td className="px-3 py-2">{pct(run.test.f1)}</td>
                            <td className="px-3 py-2">{run.evaluations || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
                    <h3 className="text-sm font-medium">
                      {result.model === "knn" ? "Onde as classes mais se separam" : "Variáveis de maior peso no modelo ajustado"}
                    </h3>
                    <ul className="mt-3 space-y-2">
                      {result.importance.map((item) => (
                        <li key={item.label} className="grid grid-cols-[1fr_auto] items-center gap-3 text-sm">
                          <span>{item.label}</span>
                          <span className="text-muted-foreground">{pct(item.share)}</span>
                          <span className="col-span-2 h-1.5 overflow-hidden rounded-full bg-[#ebe4da]">
                            <span
                              className="block h-full rounded-full bg-[#0e4f45]"
                              style={{ width: `${Math.max(8, (item.share / (result.importance[0]?.share || 1)) * 100)}%` }}
                            />
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h3 className="text-sm font-medium">Relatório para a equipe</h3>
                        <p className="text-xs text-muted-foreground">Traduz o comparativo em um texto curto, sem inventar número.</p>
                      </div>
                      <Button type="button" variant="outline" disabled={reportBusy} onClick={loadReport}>
                        {reportBusy ? "Escrevendo…" : "Gerar relatório"}
                      </Button>
                    </div>
                    {report && <NarrativeBlock narrative={report} />}
                  </div>
                </TabsContent>
                <TabsContent value="curva" className="mt-3">
                  <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
                    <div className="mb-2 flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1"><i className="inline-block h-0.5 w-4 bg-[#0e4f45]" /> melhor</span>
                      <span className="inline-flex items-center gap-1"><i className="inline-block h-0.5 w-4 bg-[#c4b8a8]" /> média da população</span>
                    </div>
                    <Convergence history={chosen.history} />
                    <p className="mt-2 text-xs text-muted-foreground">
                      Geração 0 é a população inicial, que já contém o modelo de fábrica. A curva é a fitness de validação.
                    </p>
                  </div>
                </TabsContent>
                <TabsContent value="registros" className="mt-3 space-y-3">
                  {metrics && (
                    <div className="grid gap-3 sm:grid-cols-3">
                      <Stat label="Buscas nesta sessão" value={String(metrics.optimizations)} detail={`${metrics.evaluations} treinos`} />
                      <Stat label="Recusadas por ocupação" value={String(metrics.rejected_busy)} detail="uma busca por vez" />
                      <Stat label="Falhas" value={String(metrics.failures)} detail={`${metrics.requests} requisições`} />
                    </div>
                  )}
                  <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
                    {logs.length === 0 ? (
                      <p className="text-sm text-muted-foreground">Ainda não há eventos.</p>
                    ) : (
                      <ul className="space-y-2 text-sm">
                        {logs.map((event) => (
                          <li key={`${event.at}-${event.message}`} className="border-b border-border pb-2">
                            <span className="block text-xs text-muted-foreground">{event.at.replace("T", " ").replace("+00:00", " UTC")}</span>
                            {event.message}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </section>

        <aside className={cn("min-w-0", pane !== "caso" && "hidden lg:block")}>
          <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="font-serif text-lg">Caso do teste</h2>
                <p className="text-xs leading-5 text-muted-foreground">
                  O gabarito fica visível para o estudo. A leitura gerada não recebe essa resposta.
                </p>
              </div>
            </div>
            {!result && (
              <p className="mt-4 text-sm leading-6 text-muted-foreground">
                Otimize um modelo para revisar os {overview?.split.test ?? 114} casos escondidos da busca e gerar a leitura.
              </p>
            )}
            {result && (
              <>
                <div className="mt-3 flex gap-2">
                  {(["todos", "divergentes", "malignos"] as CaseFilter[]).map((item) => (
                    <Button key={item} type="button" size="xs" variant={filter === item ? "default" : "outline"} onClick={() => setFilter(item)}>
                      {item === "todos" ? "Todos" : item === "divergentes" ? "Divergentes" : "Malignos"}
                    </Button>
                  ))}
                </div>
                <ScrollArea className="mt-3 h-52 rounded-lg border border-border">
                  <ul>
                    {visibleCases.length === 0 && (
                      <li className="px-3 py-4 text-sm text-muted-foreground">Nenhum caso neste filtro.</li>
                    )}
                    {visibleCases.map((item) => (
                      <li key={item.index}>
                        <button
                          type="button"
                          onClick={() => {
                            setSelected(item.index);
                            setReading(null);
                          }}
                          className={cn(
                            "flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm",
                            item.index === currentCase?.index ? "bg-[#e7f3ef]" : "hover:bg-muted",
                          )}
                        >
                          <span>Caso {item.index + 1}</span>
                          <span className="flex items-center gap-2">
                            <ClassMark value={item.predicted} />
                            <span className="text-xs text-muted-foreground">{item.correct ? "confere" : "diverge"}</span>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </ScrollArea>

                {currentCase && (
                  <div className="mt-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <ClassMark value={currentCase.predicted} />
                      <Badge variant="outline">{pct(currentCase.probability)} maligno</Badge>
                      <Badge variant="outline">gabarito {currentCase.actual}</Badge>
                    </div>
                    <ScrollArea className="mt-3 h-44">
                      {groups.map((group) => (
                        <div key={group.name} className="mb-3">
                          <p className="text-xs text-muted-foreground">{group.name}</p>
                          <ul>
                            {group.items.map((feature) => (
                              <li key={feature.label} className="flex justify-between gap-3 py-0.5 text-sm">
                                <span>{feature.label}</span>
                                <span className="text-muted-foreground">{measure(feature.value)}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </ScrollArea>
                    <Button
                      type="button"
                      className="mt-2 h-10 w-full"
                      disabled={readingBusy}
                      onClick={() => loadReading(currentCase.index)}
                    >
                      {readingBusy ? "Escrevendo a leitura…" : "Gerar leitura para a equipe"}
                    </Button>
                    {reading && <NarrativeBlock narrative={reading} />}
                  </div>
                )}
              </>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

function BaselineBoard({ overview, active }: { overview: Overview; active: ModelId }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-serif text-2xl tracking-tight">Modelos originais, antes da busca</h2>
        <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">{overview.note}</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {overview.baselines.map((item) => (
          <article
            key={item.model}
            className={cn("rounded-xl bg-card p-4 ring-1", item.model === active ? "ring-primary" : "ring-foreground/10")}
          >
            <h3 className="font-medium">{item.label}</h3>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">{item.blurb}</p>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div>
                <dt className="text-xs text-muted-foreground">Recall no teste</dt>
                <dd>{pct(item.test.recall)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">F1 no teste</dt>
                <dd>{pct(item.test.f1)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Precisão no teste</dt>
                <dd>{pct(item.test.precision)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Fitness na validação</dt>
                <dd>{fitness(item.validation.fitness)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
      <p className="text-sm leading-6 text-muted-foreground">{overview.fitness}</p>
    </div>
  );
}

function NarrativeBlock({ narrative }: { narrative: Narrative }) {
  return (
    <div className="mt-4 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">{narrative.source === "llm" ? "Modelo de linguagem" : "Redator local"}</Badge>
        <Badge variant={narrative.quality.score >= 85 ? "secondary" : "destructive"}>
          Qualidade {narrative.quality.score}
        </Badge>
      </div>
      {narrative.warning && <p className="text-xs text-[#8a5a12]">{narrative.warning}</p>}
      <div className="whitespace-pre-wrap text-sm leading-6">{narrative.text}</div>
      <p className="text-xs text-muted-foreground">{narrative.quality.summary}</p>
      <ul className="space-y-1 text-xs">
        {narrative.quality.checks.map((check) => (
          <li key={check.name} className={check.passed ? "text-[#0e4f45]" : "text-destructive"}>
            {check.passed ? "Coberto" : "Faltou"} · {check.name}
          </li>
        ))}
      </ul>
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">Prompt usado</summary>
        <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-[#f6f1ea] p-3">{narrative.prompt}</pre>
      </details>
    </div>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-xl bg-card px-3 py-3 ring-1 ring-foreground/10">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-serif text-2xl leading-none">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  suffix,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  suffix?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className={cn("block text-xs text-muted-foreground", disabled && "opacity-60")}>
      <span className="flex justify-between">
        <span>{label}</span>
        <span className="text-foreground">
          {value}
          {suffix ?? ""}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1 w-full"
      />
    </label>
  );
}

function ClassMark({ value }: { value: "maligno" | "benigno" }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-full px-2 text-xs font-medium",
        value === "maligno" ? "bg-[#9f2d20] text-white" : "bg-[#e5f3ef] text-[#0e4f45]",
      )}
    >
      {value}
    </span>
  );
}

function Skeleton() {
  return (
    <div className="space-y-3">
      <div className="h-8 w-64 animate-pulse rounded-lg bg-[#e7e0d6]" />
      <div className="grid gap-3 sm:grid-cols-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="h-32 animate-pulse rounded-xl bg-[#e7e0d6]" />
        ))}
      </div>
    </div>
  );
}

function labelOf(model: ModelId) {
  return {
    knn: "K vizinhos mais próximos",
    logistica: "Regressão logística",
    arvore: "Árvore de decisão",
    floresta: "Floresta aleatória",
  }[model];
}

function transferSentence(original: RunSummary, chosen: RunSummary) {
  const recall = chosen.test.recall - original.test.recall;
  const f1 = chosen.test.f1 - original.test.f1;
  if (f1 > 0.005 || recall > 0.005) {
    return "A fitness subiu na validação, e parte desse ganho apareceu no teste, que a busca não viu.";
  }
  if (f1 < -0.005 || recall < -0.005) {
    return "A fitness subiu na validação. No teste, separado antes da busca, o ganho não se repetiu por completo: o limiar ou a complexidade encontrados na validação não viajaram juntos.";
  }
  return "A validação melhorou e o teste ficou praticamente no mesmo lugar.";
}

function groupFeatures(item: CaseRow | undefined) {
  if (!item) return [];
  const groups = ["Médias", "Erro padrão", "Maiores valores"];
  return groups.map((name) => ({
    name,
    items: item.features.filter((feature) => feature.group === name),
  }));
}
