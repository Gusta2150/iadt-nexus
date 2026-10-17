export function pct(value: number) {
  return (
    (value * 100).toLocaleString("pt-BR", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }) + "%"
  );
}

export function points(after: number, before: number) {
  const delta = (after - before) * 100;
  const sign = delta > 0.05 ? "+" : "";
  return (
    sign +
    delta.toLocaleString("pt-BR", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }) +
    " pp"
  );
}

export function measure(value: number) {
  return value.toLocaleString("pt-BR", {
    maximumFractionDigits: value >= 100 ? 1 : value >= 10 ? 2 : 3,
  });
}

export function seconds(ms: number) {
  return (ms / 1000).toLocaleString("pt-BR", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }) + " s";
}

export function fitness(value: number) {
  return value.toLocaleString("pt-BR", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });
}
