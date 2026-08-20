#!/usr/bin/env python3
"""O modelo padrão do Chroma medido em português.

Material de estudo pessoal adicionado neste fork — não faz parte do projeto
Chroma.

A pergunta que originou o script: dá para usar um limiar de distância para
descartar contexto irrelevante antes de montar o prompt do RAG?

Para o limiar existir, toda pergunta com resposta no corpus precisa ficar mais
perto do que qualquer pergunta sem resposta. Isso não é dado: o modelo padrão
do Chroma é o `all-MiniLM-L6-v2`, treinado em inglês, e aqui o corpus está em
português. O script mede a separação em vez de supor — e imprime a conclusão
que os números sustentam, seja ela qual for.

Uso:
    pip install chromadb
    python rag_ptbr.py
"""

from __future__ import annotations

import chromadb

DOCUMENTOS = [
    (
        "telemetria-1",
        "O sistema de telemetria registra rotação do motor, temperatura e "
        "posição do pedal do acelerador durante a prova.",
        {"area": "automotivo"},
    ),
    (
        "telemetria-2",
        "Antes de calcular médias, o tratamento remove leituras fora da faixa "
        "física do sensor e amostras repetidas por falha de comunicação.",
        {"area": "automotivo"},
    ),
    (
        "borboleta-1",
        "A unidade de controle lê a posição do pedal e comanda a abertura do "
        "corpo de borboleta eletrônico com um controlador PID.",
        {"area": "automotivo"},
    ),
    (
        "estoque-1",
        "O saldo do estoque é derivado dos movimentos de entrada e saída, e "
        "não guardado em coluna própria, para não divergir do histórico.",
        {"area": "dados"},
    ),
    (
        "busca-1",
        "A busca híbrida combina BM25 com embeddings densos e funde os "
        "rankings pelo Reciprocal Rank Fusion.",
        {"area": "ia"},
    ),
    (
        "chunking-1",
        "Trechos grandes demais diluem o sinal do embedding; trechos pequenos "
        "demais chegam ao gerador sem contexto suficiente.",
        {"area": "ia"},
    ),
]

# (pergunta, id esperado). None = não há resposta no corpus.
PERGUNTAS = [
    ("o que o controlador PID comanda?", "borboleta-1"),
    ("por que o saldo não é guardado em coluna?", "estoque-1"),
    ("qual o problema de trechos muito grandes?", "chunking-1"),
    ("como os dois rankings de busca são combinados?", "busca-1"),
    ("qual é a receita do bolo de cenoura?", None),
    ("quanto custa a passagem de ônibus para Belém?", None),
    ("quem ganhou a copa do mundo de 1994?", None),
]


def montar_colecao():
    """Cria uma coleção em memória usando distância de cosseno.

    O padrão do Chroma é L2. Com o cosseno, a distância fica numa escala
    comparável entre corpora — que é o mínimo necessário para sequer tentar
    calibrar um limiar.
    """
    cliente = chromadb.Client()
    colecao = cliente.get_or_create_collection(
        name="estudos_ptbr",
        metadata={"hnsw:space": "cosine"},
    )
    colecao.add(
        ids=[doc_id for doc_id, _, _ in DOCUMENTOS],
        documents=[texto for _, texto, _ in DOCUMENTOS],
        metadatas=[meta for _, _, meta in DOCUMENTOS],
    )
    return colecao


def medir(colecao) -> None:
    """Mede a distância do vizinho mais próximo de cada pergunta."""
    print(f"{'pergunta':<48} {'top-1':<14} {'dist':>6}  acerto")
    print("-" * 82)

    com_resposta: list[float] = []
    sem_resposta: list[float] = []
    acertos = 0
    esperadas = 0

    for pergunta, esperado in PERGUNTAS:
        resposta = colecao.query(query_texts=[pergunta], n_results=1)
        encontrado = resposta["ids"][0][0]
        distancia = resposta["distances"][0][0]

        if esperado is None:
            sem_resposta.append(distancia)
            marca = "—"
        else:
            com_resposta.append(distancia)
            esperadas += 1
            acertou = encontrado == esperado
            acertos += int(acertou)
            marca = "ok" if acertou else f"ERROU (esperava {esperado})"

        print(f"{pergunta[:47]:<48} {encontrado:<14} {distancia:>6.3f}  {marca}")

    media_com = sum(com_resposta) / len(com_resposta)
    media_sem = sum(sem_resposta) / len(sem_resposta)

    print("\nResumo")
    print(f"  top-1 correto ................... {acertos}/{esperadas}")
    print(f"  distância média com resposta .... {media_com:.3f}")
    print(f"  distância média sem resposta .... {media_sem:.3f}")
    print(f"  separação ....................... {media_sem - media_com:+.3f}")
    print(f"  pior caso com resposta .......... {max(com_resposta):.3f}")
    print(f"  melhor caso sem resposta ........ {min(sem_resposta):.3f}")

    if min(sem_resposta) < max(com_resposta):
        print(
            "\n  As faixas se sobrepõem: existe pergunta sem resposta mais\n"
            "  próxima do que pergunta com resposta. Nenhum limiar de\n"
            "  distância separa os dois casos neste corpus."
        )
    else:
        limiar = (min(sem_resposta) + max(com_resposta)) / 2
        print(f"\n  As faixas não se sobrepõem: um limiar em {limiar:.3f} separa.")


def demonstrar_filtro(colecao) -> None:
    """Filtro por metadado: corta o espaço antes da busca vetorial."""
    print("\n" + "=" * 82)
    print("Filtro por metadado (area = 'ia')")
    resposta = colecao.query(
        query_texts=["como combinar dois rankings de busca?"],
        n_results=2,
        where={"area": "ia"},
    )
    for doc_id, distancia in zip(resposta["ids"][0], resposta["distances"][0]):
        print(f"  {doc_id:<14} {distancia:.3f}")
    print(
        "\n  Quando o corpus tem estrutura conhecida, o filtro de metadado é\n"
        "  mais confiável que qualquer limiar: ele não depende de o modelo\n"
        "  entender o idioma."
    )


def main() -> None:
    colecao = montar_colecao()
    print(f"{colecao.count()} documentos em português indexados")
    print("modelo: all-MiniLM-L6-v2 (padrão do Chroma, treinado em inglês)\n")
    medir(colecao)
    demonstrar_filtro(colecao)


if __name__ == "__main__":
    main()
