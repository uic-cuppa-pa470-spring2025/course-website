library(tidyverse)
library(text)
Sys.unsetenv("RETICULATE_PYTHON")
textrpp_initialize(virtualenv = "~/global_virtualenvs/misc")

input_texts <- c(
  "Which baseball team plays in Chicago?",
  "George",
  "Chelsea",
  "Donuts",
  "The Commanders",
  "The Chiefs",
  "The Cubs"
  )

# Defaults
embeddings <- textEmbed(input_texts)
print(embeddings)


embedding_text <- embeddings$texts$texts

textSimilarity(embedding_text, embedding_text %>% slice(rep(1:1, each = 7)), method = "cosine")

#######################

texts <- c(
  "Man",
  "Woman",
  "Pilot",
  "Air Hostess",
  "King",
  "Queen"
  # "Tree",
  # "Forest",
  # "Ocean",
  # "Nature",
  # "Airplane",
  # "Home",
  # "Office",
  # "Palace"
)

# Defaults
embeddings <- textEmbed(texts)
print(embeddings)


embedding_text <- embeddings$texts$texts
embedding_text_matrix <- embedding_text %>% as.matrix() 
(embedding_text[1,] - embedding_text[2,] + embedding_text[4,]) %>% as_tibble() %>%
  textSimilarity(embedding_text[3,] %>% as_tibble())

(embedding_text[1,] - embedding_text[2,] + embedding_text[6,]) %>% as_tibble() %>%
  textSimilarity(embedding_text[5,] %>% as_tibble())

pca_res <- prcomp(embedding_text, scale = TRUE) 

pca_res$sdev^2 / sum(pca_res$sdev^2)

ggplot(data = pca_res$x) +
  geom_label(aes(x = PC1, y = PC2), label = texts)


#######################

texts <- c(
  "Cricket",
  "Baseball",
  "Basketball",
  "Football",
  "Soccer",
  "American Football"
)

embeddings <- textEmbed(texts)
print(embeddings)
embedding_text <- embeddings$texts$texts
pca_res <- prcomp(embedding_text, scale = TRUE) 

pca_res$sdev^2 / sum(pca_res$sdev^2)

pca_res_tibble <- pca_res$x %>% as_tibble()

library(plotly)
plot_ly(x=pca_res_tibble$PC1, y=pca_res_tibble$PC2, z=pca_res_tibble$PC3, type="scatter3d", mode="text",
        text = texts)


#########################
generated_text <- textGeneration("Springfield is the capital of the state of", model = "gpt2")
print(generated_text[[1]])
  