import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function MethodDialog() {
  return (
    <Dialog>
      <DialogTrigger render={<Button variant="outline" size="sm" type="button" />}>
        Método
      </DialogTrigger>
      <DialogContent className="max-h-[85dvh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Como a busca escolhe o modelo</DialogTitle>
          <DialogDescription>
            Algoritmo genético sobre hiperparâmetros. O conjunto de teste não entra na função fitness.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 text-sm leading-6 text-foreground">
          <section>
            <h3 className="font-medium">Codificação</h3>
            <p className="text-muted-foreground">
              Cada indivíduo é um vetor de genes entre 0 e 1. A decodificação transforma esse vetor em
              hiperparâmetros reais: o C da logística, a profundidade da árvore, o número de vizinhos e,
              em todos os modelos, o limiar a partir do qual o caso é lido como maligno (de 0,30 a 0,70).
            </p>
          </section>
          <section>
            <h3 className="font-medium">Operadores</h3>
            <p className="text-muted-foreground">
              A seleção é por torneio. O cruzamento pode ser mistura, uniforme ou de um ponto. A mutação,
              aplicada por gene, soma um ruído gaussiano ou troca o gene por um valor novo. Os melhores
              indivíduos passam intactos para a geração seguinte.
            </p>
          </section>
          <section>
            <h3 className="font-medium">Fitness</h3>
            <p className="text-muted-foreground">
              50% recall de malignidade, 35% F1 e 15% precisão, calculados só na validação. O teste, com
              114 casos, fica guardado para o comparativo final. A população inicial inclui o modelo de
              fábrica, então a validação da busca não fica pior do que a baseline.
            </p>
          </section>
          <section>
            <h3 className="font-medium">Três configurações</h3>
            <p className="text-muted-foreground">
              Exploratória usa população menor e mutação alta. Balanceada é o meio termo. Intensiva reduz
              a mutação, aumenta a população e troca o cruzamento. Na floresta aleatória as populações são
              menores porque cada indivíduo treina dezenas de árvores.
            </p>
          </section>
          <section>
            <h3 className="font-medium">Leitura clínica</h3>
            <p className="text-muted-foreground">
              O texto para a equipe usa só os números do caso e do teste. Sem chave de API, um redator
              local segue o mesmo prompt. Com OPENAI_API_KEY, o prompt vai para o modelo configurado. Uma
              checagem marca se a leitura cita a classe, as características, a probabilidade e a ressalva
              de que não substitui a equipe.
            </p>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
