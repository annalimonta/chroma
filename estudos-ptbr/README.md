# Estudos em português — limiar de distância no Chroma

> **Material de estudo pessoal, adicionado neste fork.** Não faz parte do
> projeto [Chroma](https://github.com/chroma-core/chroma) e não foi proposto
> como contribuição para ele.

A pergunta: **dá para usar um limiar de distância para descartar contexto
irrelevante antes de montar o prompt do RAG?**

A busca vetorial sempre devolve os `n_results` mais próximos, mesmo quando
nenhum documento tem relação com a pergunta. Num pipeline RAG, esse contexto
irrelevante vai parar no prompt e vira alucinação. O limiar seria a defesa — se
funcionar.

## O experimento

[`rag_ptbr.py`](rag_ptbr.py) indexa 6 trechos em português e consulta com 7
perguntas: 4 com resposta no corpus e 3 completamente fora dele. Mede a
distância do vizinho mais próximo em cada caso.

```bash
pip install chromadb
python rag_ptbr.py
```

## Resultado medido

Modelo: `all-MiniLM-L6-v2` — o padrão do Chroma, **treinado em inglês**.
Métrica: distância de cosseno (`hnsw:space: cosine`).

| Pergunta | Top-1 | Distância | |
| --- | --- | ---: | --- |
| o que o controlador PID comanda? | borboleta-1 | 0.371 | ok |
| por que o saldo não é guardado em coluna? | estoque-1 | 0.285 | ok |
| qual o problema de trechos muito grandes? | chunking-1 | 0.394 | ok |
| como os dois rankings de busca são combinados? | busca-1 | 0.396 | ok |
| qual é a receita do bolo de cenoura? | estoque-1 | 0.485 | sem resposta |
| quanto custa a passagem de ônibus para Belém? | estoque-1 | 0.545 | sem resposta |
| quem ganhou a copa do mundo de 1994? | estoque-1 | 0.559 | sem resposta |

```
top-1 correto ................... 4/4
distância média com resposta .... 0.362
distância média sem resposta .... 0.530
separação ....................... +0.168
pior caso com resposta .......... 0.396
melhor caso sem resposta ........ 0.485
```

## O que eu concluí

**O limiar funciona aqui — com margem estreita.** As faixas não se sobrepõem:
o pior caso com resposta (0.396) fica antes do melhor caso sem resposta
(0.485). Um corte em ~0.44 separa os dois grupos. A margem é de 0.09, o que é
pouco: dois trechos a mais no corpus podem fechá-la.

**Eu esperava que não funcionasse.** O modelo é treinado em inglês e o corpus é
português — a intuição dizia que a distância seria ruído. Ela não foi: o top-1
acertou 4 de 4. Foi a medição que corrigiu a suposição, e é por isso que ela
existe.

**A pergunta sem resposta ainda devolve documento.** Sem limiar, "qual é a
receita do bolo de cenoura?" traz o trecho de estoque com toda a confiança do
mundo. O limiar não é refinamento: é o que separa "não sei" de resposta
inventada.

**O limiar é do corpus, não do modelo.** 0.44 vale para estes 6 trechos e estas
perguntas. Trocar o corpus, o idioma ou o modelo exige medir de novo — copiar o
número de um tutorial é o mesmo que chutar.

**Filtro de metadado é mais confiável que limiar.** Quando o corpus tem
estrutura conhecida (`where={"area": "ia"}`), o filtro corta o espaço de busca
sem depender de o modelo entender o idioma. Onde der para usar, use.

## Próximo passo

Repetir a medição com um modelo multilíngue
(`paraphrase-multilingual-MiniLM-L12-v2`) para ver se a margem de 0.09 aumenta.
A hipótese é que sim — mas, como este experimento mostrou, a hipótese não vale
nada até virar número.

## Relacionado

- [rag-do-zero](https://github.com/annalimonta/rag-do-zero) — o mesmo corte por
  similaridade, implementado à mão em Python.
