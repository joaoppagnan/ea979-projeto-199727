# Computação Gráfica utilizando Redes Neurais Adversárias Generativas
## Computer Graphics with Generative Adversarial Neural Networks

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *EA979A - Introdução a Computação Gráfica e Processamento de Imagens*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

**Grupo:**
|Nome  | RA | Curso|
|--|--|--|
| João Pedro de Oliveira Pagnan | 199727  | Eng. Elétrica|


## Descrição do Projeto

Este projeto tem como objetivo implementar um modelo de rede neural adversário generativo de forma a estudar esta maneira de se realizar computação gráfica.

### Contexto e Motivação

Com o crescimento do poderio computacional nos últimos anos, em especial, das placas de vídeo, o uso e a popularidade de modelos com redes neurais artificiais em tarefas de aprendizado de máquina, como predição, classificação, regressão etc., aumentou de maneira considerável. 

Uma arquitetura de rede neural que ficou bastante conhecida foi as rede neural adversária generativa. Apresentada original em 2014 [neste artigo](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) por Ian Goodfellow e os outros membros do seu grupo de pesquisa, este modelo é capaz de, após um treinamento em um conjunto de dados, gerar novas amostras com os mesmos parâmetros das que foram utilizados para treinar a rede.

Este tipo de rede neural é bastante utilizado, por exemplo, em tarefas de computação gráfica, sendo possível gerar fotos de [rostos humanos](https://thispersondoesnotexist.com/), de [animais](https://thiscatdoesnotexist.com/), de [artes](https://thisartworkdoesnotexist.com/), dentre vários outros exemplos. 

Apesar das redes tradicionalmente funcionarem com um ruído como entrada, a incorporação de atributos retirados do espaço latente da representação interna da rede através de operações aritméticas é capaz de gerar uma imagem com base em uma descrição fornecida pelo usuário do modelo. Desta forma, esta arquitetura de rede neural se encaixa na definição de computação gráfica ao ser uma ferramenta que gera imagens a partir de dados de entrada.

### Objetivo Principal

O objetivo principal deste trabalho é, como o nome já indica, realizar um estudo de como a tarefa de computação gráfica pode ser realizada com esse tipo de rede neural artificial, assim como estudar como as aritméticas no espaço latente podem ser utilizadas para gerar imagens com certas características. 

Neste caso, vale mencionar que, o que será gerado pelo modelo ainda não está definido. Pode ser uma arte, um rosto humano, um animal etc.

## Plano de Trabalho

* Etapa 1 (1 semanas): Estudo de redes neurais artificiais adversárias generativas;
* Etapa 2 (1 semana): Definição do tipo de imagem que será gerada pela rede e procura de bases de dados públicas para serem utilizadas;
* Etapa 3 (3 semanas): Programação e testes com os modelos;
* Etapa 4 (1 semana): Análise dos resultados obtidos;
* Etapa 5 (2 semanas): Escrita do relatório final do projeto.

## Referências Bibliográficas

1. Géron, A., 2019. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".
2. Goodfellow, I., Bengio, Y. and Courville, A., 2016. Deep learning. MIT press.
3. Bishop, C.M. and Nasrabadi, N.M., 2006. Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.
4. Hastie, T., Tibshirani, R., Friedman, J.H. and Friedman, J.H., 2009. The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
5. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Generative adversarial nets. Advances in neural information processing systems, 27.
6. Karras, T., Laine, S. and Aila, T., 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4401-4410).
7. Ling, H., Kreis, K., Li, D., Kim, S.W., Torralba, A. and Fidler, S., 2021. EditGAN: High-Precision Semantic Image Editing. Advances in Neural Information Processing Systems, 34.