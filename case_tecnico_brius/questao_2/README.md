# README - Agente de Aprendizado para Otimização de Experiência do Usuário

Este README descreve as decisões de projeto e fornece instruções sobre como implementar e treinar um agente de aprendizado para otimizar a experiência do usuário em um site de e-commerce, aumentando o número de conversões por meio de recomendações personalizadas. Todo o processo é detalhado no notebook `Brius_questao2.ipynb`.

## Decisões de Projeto

### Autenticação e Acesso aos Dados

Para acessar os dados do Google BigQuery, optei por instalar o `gcloud CLI` na minha máquina e realizar a autenticação através dele. Esta decisão foi motivada pelos seguintes fatores:

1. **Segurança**: A autenticação via `gcloud CLI` é mais segura, garantindo a proteção dos dados.
2. **Familiaridade**: Tenho mais familiaridade com ferramentas de linha de comando, o que facilita o processo de autenticação e gerenciamento dos dados.

### Manipulação dos Dados

Para acessar e manipular os dados do BigQuery, utilizei queries SQL simples. Essa escolha se baseou em minha expertise prévia com SQL, que é uma linguagem poderosa e fácil de usar para manipulação de dados. Para obter os dados de forma pandas-like e manipulável, optei pela biblioteca `pandas_gbq`, que permite a execução de queries SQL diretamente no BigQuery e a obtenção dos resultados em um dataframe pandas. Esta abordagem é recomendada pelo próprio pandas, uma vez que a função original de conexão com o BigQuery será descontinuada.

### Implementação do Agente de Aprendizado

Por limitações de acesso a servidores com  GPU e tempo de treinamento, o agente proposto não foi treinado. No entanto, o notebook contém instruções detalhadas sobre como implementar o modelo e treiná-lo.