# Sentiment Classification of Movie Reviews using Transformer-based Models

This project investigates sentiment classification of movie reviews using Transformer-based language models. Two main approaches are compared:

1. **Fine-tuned Transformer (RoBERTa)**  
2. **Sentence Embedding + Logistic Regression**



## Dataset
The Rotten Tomatoes dataset from Hugging Face was used:
```python
from datasets import load_dataset
data = load_dataset("rotten_tomatoes")
```
Results Summary

| Model | Accuracy | Macro F1 | Highlights |
| --- | --- | --- | --- |
| RoBERTa | 0.80 | 0.80 | Strong precision on positive reviews |
| Sentence-Embedding + Logistic Regression | 0.85 | 0.85 | More balanced, slightly higher accuracy |


# Key Takeaways

- Transformer models effectively capture sentiment context.

- Embedding-based models perform competitively with less computational cost.

- Misclassifications highlight challenges with nuanced reviews.

For detailed methodology and evaluation see  [Projects](https://ernselito.github.io/sentiment/)


[Google Colab](https://colab.research.google.com/drive/1lMBCqLYvD5qSB7O0vQbOaDk1o8DV8ZV_?usp=sharing)


