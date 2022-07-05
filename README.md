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

Uma arquitetura de rede neural que ficou bastante conhecida foi a rede neural adversária generativa. Apresentada originalmente em 2014 [neste artigo](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) por Ian Goodfellow e os outros membros do seu grupo de pesquisa, este modelo é capaz de, após um treinamento em um conjunto de dados, gerar novas amostras com os mesmos parâmetros das imagens que foram utilizados para treinar a rede.

Este tipo de rede neural é bastante utilizado, por exemplo, em tarefas de computação gráfica, sendo possível gerar fotos de [rostos humanos](https://thispersondoesnotexist.com/), de [animais](https://thiscatdoesnotexist.com/), de [artes](https://thisartworkdoesnotexist.com/), dentre vários outros exemplos. 

Apesar das redes tradicionalmente funcionarem com um ruído como entrada, a incorporação de atributos retirados do espaço latente da representação interna da rede através de operações aritméticas é capaz de gerar uma imagem com base em uma descrição fornecida pelo usuário do modelo. Desta forma, esta arquitetura de rede neural se encaixa na definição de computação gráfica ao ser uma ferramenta que gera imagens a partir de dados de entrada.

### Objetivo Principal

O objetivo principal deste trabalho é, como o nome já indica, realizar um estudo de como a tarefa de computação gráfica pode ser realizada com esse tipo de rede neural artificial, assim como estudar como as aritméticas no espaço latente podem ser utilizadas para gerar imagens com certas características. 

Neste caso, vale mencionar que, o que será gerado pelo modelo ainda não está definido. Pode ser uma arte, um rosto humano, um animal etc.

### Atualização - Entrega 2

Depois de uma busca na literatura, foi escolhido utilizar a [EditGAN](https://arxiv.org/pdf/2111.03186.pdf), para implementar este projeto. No caso, esta configuração de rede GAN permite a edição de alguns parâmetros das imagens sendo geradas (características como, caso estiver sendo gerados rostos, a direção para onde os olhos estão olhando, ou a expressão desejada) através de um reconhecimento semântico de um texto fornecido. 

Também foi escolhido gerar imagens de carros por serem mais simples que rostos. 

Infelizmente, ainda não foi possível iniciar a implementação do código desta configuração de GAN. Por sorte, o código utilizado pelos autores do artigo está no **Github** e a implementação foi feita em **Python** utilizando a biblioteca **PyTorch**.

### Atualizações - Entrega 3

Houveram algumas atualizações entre a etapa 2 e a etapa 3: 

1. As tentativas iniciais de implementar a EditGAN não deram certo, mesmo com os pesos dos modelos pré-treinados. Devido a isso, optou-se por implementar a [StyleGAN](https://arxiv.org/pdf/1812.04948.pdf);
1. Optou-se por implementar a primeira versão desta arquitetura de GANs;
1. Originalmente planejava-se estudar como a edição de imagens pode ser feita através do espaço latente desta rede, mas, devido a falta de tempo, este objetivo foi descontinuado e o foco foi a geração de imagens sintéticas de forma aleatória;
1. Além disso, as imagens que seriam geradas originalmente eram imagens de carros, porém, foi utilizada a proposta original da StyleGAN que era a de gerar rostos humanos.

## Abordagem Adotada

Como dito na seção anterior, foi utilizada a primeira versão da StyleGAN, que é uma arquitetura de rede neural adversária generativa originalmente proposta em 2018 pelo NVlabs. Devido ao fato de que este é um modelo bem famoso, foram utilizados códigos já implementados através do pacote **PyTorch**. Neste caso a rede neural estará gerando faces humanas.

O treinamento deste tipo de modelo é bem custoso computacionalmente. Devido a este fato, foi utilizada a abordagem de *transfer learning* de forma a usar um modelo já treinado, disponível [neste](https://www.kaggle.com/code/songseungwon/image-generation-using-stylegan-pre-trained-model/data?select=karras2019stylegan-ffhq-1024x1024.for_g_all.pt) *link*. O modelo foi treinado com a base de dados **Flickr-Faces-HQ**, disponível [neste](https://github.com/NVlabs/ffhq-dataset) repositório. Devido a este motivo, a rede neural só vai gerar rostos humanos. 

A base mencionada consiste de setenta mil imagens de alta qualidade no formato **PNG** na resolução de 1024x1024 de rostos humanos de homens e mulheres de diversas idades e etnias, podendo ou não conter acessórios como óculos, chapéus etc., retiradas do [Flickr](https://www.flickr.com/). Esta base foi construída para ser utilizada no artigo original da StyleGAN mas foi disponibilizada após a publicação do trabalho.

Grande parte do código foi retirado [deste](https://www.kaggle.com/code/songseungwon/image-generation-using-stylegan-pre-trained-model/notebook) *notebook* do **Kaggle**, mas foi feita uma grande adaptação nos códigos pois originalmente estes estavam em um único arquivo do **Jupyter** e, neste repositório, cada elemento da rede neural será um objeto de uma classe, o que facilita a modificação e correção de erros.

Os *scripts* da StyleGAN estão no diretório `src/`, enquanto que os *notebooks* criados estão no diretório `notebooks/`. O arquivo [1.0-jpp-download-data.ipynb](https://github.com/joaoppagnan/ea979-projeto-199727/blob/main/notebooks/1.0-jpp-download-data.ipynb) serve para baixar as *thumbnails* das imagens utilizadas para treinar o modelo, o [1.0-jpp-download-pretrained.ipynb](https://github.com/joaoppagnan/ea979-projeto-199727/blob/main/notebooks/1.0-jpp-download-pretrained.ipynb) para fazer o *download* dos pesos do modelo pré-treinado e, por fim, o *notebook* [1.0-jpp-stylegan.ipynb](https://github.com/joaoppagnan/ea979-projeto-199727/blob/main/notebooks/1.0-jpp-stylegan.ipynb) é onde a geração de novas imagens é feita utilizando o código presente no `src/` e o modelo pré-treinado. 

São gerados cinco conjuntos de nove faces, estando estes armazenados no diretório `figures/`. Além disso, há um outro conjunto no último *notebook* mencionado, mas estes não é salvo em um arquivo.

## Como reproduzir os resultados?

Os resultados podem ser reproduzidos com os seguintes passos:
1. Clonar o repositório;
1. Executar a *script* `build_docker.sh` com o comando `sh build_docker.sh` para montar a imagem do Docker utilizada;
1. Executar a *script* `run_docker.sh` com `sh run_docker.sh` para rodar a imagem e acessar o ambiente do **Jupyter Lab**;
1. (OPCIONAL) Executar o *notebook* `1.0-jpp-download-data.ipynb` para baixar as *thumbnails* das imagens utilizadas para treinar o modelo;
1. Executar o *notebook* `1.0-jpp-download-pretrained.ipynb` para fazer o *download* dos pesos do modelo pré-treinado;
1. Executar o *notebook* `1.0-jpp-stylegan.ipynb` para gerar as imagens de rostos.

## Resultados Finais

Os seguintes conjuntos de nove faces foram gerados:

#### Conjunto 1: ####
![Conjunto 1](https://raw.githubusercontent.com/joaoppagnan/ea979-projeto-199727/main/figures/face-set-0.png)

#### Conjunto 2: ####
![Conjunto 2](https://raw.githubusercontent.com/joaoppagnan/ea979-projeto-199727/main/figures/face-set-1.png)

#### Conjunto 3: ####
![Conjunto 3](https://raw.githubusercontent.com/joaoppagnan/ea979-projeto-199727/main/figures/face-set-2.png)

#### Conjunto 4: ####
![Conjunto 4](https://raw.githubusercontent.com/joaoppagnan/ea979-projeto-199727/main/figures/face-set-3.png)

#### Conjunto 5: ####
![Conjunto 5](https://raw.githubusercontent.com/joaoppagnan/ea979-projeto-199727/main/figures/face-set-4.png)

## Discussão

Através da StyleGAN foi possível gerar, no total, 45 rostos humanos. O realismo e a acurácia das imagens em relação a faces reais varia consideravelmente: no conjunto 3 vê-se que o rosto da esquerda na linha do meio foi bem realista, enquanto que a imagem no canto inferior direito do primeiro conjunto ficou bem estranha pois parece que há um rosto se "desintegrando" no canto superior direito da imagem. Além disso, os fundos das imagens que os têm não estão muito bem definidos: no geral observa-se, quando há fundo, um borrão de cores. No geral, os elementos que ficam nos cantos das imagens não são muito bem gerados. 

Para resolver estes problemas foram feitas atualizações na StyleGAN que melhoram significativamente os problemas observados nesta implementação.

A principal dificuldade encontrada para implementar a EditGAN foi que, por tratar-se de um modelo proposto recentemente, não há exemplos de implementações realizadas a não ser a presente no repositório do **NVlabs** e houve uma grande dificuldade para utilizar aquele código. Enquanto isso, não houveram grandes dificuldades para implementar a StyleGAN.

A abordagem de *transfer learning* utilizada possui uma limitação considerável de que este modelo só irá gerar rostos humanos. Caso queira-se gerar artes, faces animais, objetos, ou outras coisas, seria necessário realizar o *download* de um modelo treinado para gerar estes tipos de imagem. 

Conclui-se que este projeto cumpriu o primeiro objetivo proposto, que é a computação gráfica através de redes neurais adversárias generativas, mas não foi possível alcançar a meta de estudar como modificações no espaço latente podem ser feitas de forma a editar as imagens geradas. Isto ocorreu pois um tempo considerável foi gasto tentando implementar a EditGAN, que seria uma GAN em que o grande diferencial dela seria uma edição mais simplificada do espaço latente de forma a alterar características da imagem.

## Referências Bibliográficas

1. Géron, A., 2019. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".
2. Goodfellow, I., Bengio, Y. and Courville, A., 2016. Deep learning. MIT press.
3. Bishop, C.M. and Nasrabadi, N.M., 2006. Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.
4. Hastie, T., Tibshirani, R., Friedman, J.H. and Friedman, J.H., 2009. The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
5. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Generative adversarial nets. Advances in neural information processing systems, 27.
6. Karras, T., Laine, S. and Aila, T., 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4401-4410).
7. Ling, H., Kreis, K., Li, D., Kim, S.W., Torralba, A. and Fidler, S., 2021. EditGAN: High-Precision Semantic Image Editing. Advances in Neural Information Processing Systems, 34.