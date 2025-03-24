"""
Functions to evaluate image captioning using machine translation metrics
"""
"""
BLEU (Bilingual Evaluation Understudy) Score
"""
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import torch

def metrics(predict_sentence, referenc_sentences):
    """
    Args:
        predict_sentence (str): a predicted sentence
        referenc_sentences (list[str]): a list of reference sentences
        Returns: 
            bleu_score: BLEU score
        """
    hypothesis = predict_sentence.split()

    reference = [sentence.split() for sentence in referenc_sentences]

    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method4)
    return bleu_score

""" Example """
predict_sentence = "the cat is on the mat"
referenc_sentences = ["the cat is on the mat", "there is a cat on the mat"]
bleu_score = metrics(predict_sentence, referenc_sentences)
print(bleu_score)
