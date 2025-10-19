# Diamond Classification - GitHub Actions CI/CD

## Visão Geral

Este projeto demonstra a automação completa de um pipeline de Machine Learning utilizando GitHub Actions. O sistema treina um modelo de classificação com base no dataset data/diamonds.csv e gera relatórios automáticos publicados via GitHub Pages.

## Como Funciona

### Workflows Configurados

#### 1. ML Workflow (ml.yml)
Acionado a cada push na branch `main`:
- Instala dependências Python (pandas, scikit-learn)
- Executa o treinamento do modelo
- Gera relatório de classificação em `report.txt`
- Publica os arquivos no GitHub Pages

#### 2. Slides Workflow (slides.yml)
Acionado quando há alterações em `docs/*.md` ou `docs/index.html`:
- Instala Pandoc e LaTeX
- Converte `docs/presentation.md` para PDF
- Copia `slides.pdf` e `index.html` para diretório de deployment
- Publica na branch `gh-pages`

### Página GitHub Pages

O arquivo `index.html` atua como dashboard central que:
- Carrega dinamicamente o `report.txt` gerado pelo treinamento
- Exibe o `slides.pdf` em embed
- Fornece botões de download para ambos os arquivos
- Atualiza-se automaticamente quando novos arquivos são publicados

## O Modelo de Classificação

### O que train.py Classifica

O script treina um modelo de Regressão Logística Multinomial que classifica a qualidade de corte (cut) de diamantes em 5 categorias:
- Ideal
- Premium
- Very Good
- Good
- Fair

### Processo

1. **Carregamento**: Lê dados de `data/diamonds.csv`
2. **Pré-processamento**: 
   - Remove coluna de índice
   - Codifica variáveis categóricas (color, clarity)
3. **Separação**: X contém 9 features (carat, color, clarity, depth, table, price, x, y, z) e y contém a classe de corte
4. **Divisão**: 80% treino, 20% teste
5. **Treinamento**: Logistic Regression com 1000 iterações
6. **Avaliação**: Gera relatório com precisão, recall e f1-score por classe

## Arquitetura de Deployment

```
GitHub Repository
    |
    +-- ml.yml (push -> main)
    |   +-- Treina modelo
    |   +-- Gera report.txt
    |   +-- Deploy gh-pages
    |
    +-- slides.yml (push -> docs/*.md)
    |   +-- Gera slides.pdf
    |   +-- Deploy gh-pages
    |
    +-- gh-pages branch (GitHub Pages)
        +-- index.html
        +-- slides.pdf
        +-- report.txt
```

## Dependências Relacionadas ao Projeto

- Python 3.11
    - pandas
    - scikit-learn

- Ubuntu-latest
    - Pandoc
    - texlive-latex-base
    - texlive-latex-extra