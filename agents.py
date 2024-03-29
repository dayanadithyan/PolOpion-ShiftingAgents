import spacy
from transformers import pipeline, BertTokenizer, BertModel
import torch
from collections import Counter
import math

# Load spaCy English model
nlp = spacy.load("en_core_web_lg")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_pipeline = pipeline("sentiment-analysis")

def bayesian_calculation(prior_probability, likelihood_positive, likelihood_negative, evidence_positive):
    """
    Perform Bayesian calculation.

    Parameters:
    - prior_probability (float): Prior probability of the hypothesis (patient having the disease).
    - likelihood_positive (float): Likelihood of a positive test result given the hypothesis (sensitivity).
    - likelihood_negative (float): Likelihood of a negative test result given the hypothesis (1 - specificity).
    - evidence_positive (float): Probability of a positive test result (marginal likelihood).

    Returns:
    - posterior_probability (float): Posterior probability of the hypothesis given the evidence (positive test result).
    """
    # Calculate the evidence of a negative test result
    evidence_negative = 1 - evidence_positive

    # Calculate the denominator of Bayes' theorem (marginal likelihood)
    marginal_likelihood = (prior_probability * likelihood_positive) + ((1 - prior_probability) * likelihood_negative)

    # Calculate the posterior probability using Bayes' theorem
    posterior_probability = (prior_probability * likelihood_positive) / marginal_likelihood

    return posterior_probability

class Agent:
    def __init__(self, name, opinion):
        self.name = name
        self.opinion = opinion

    def update_opinion(self, news, cognitive_tool):
        """
        Update opinion based on news content and cognitive tool.

        Parameters:
        - news (str): The news content to evaluate.
        - cognitive_tool (str): The cognitive tool or theory used to evaluate the news content.
        """
        raise NotImplementedError("The update_opinion method must be implemented by subclasses.")

class ModelBasedReflexAgent(Agent):
    def update_opinion(self, news, cognitive_tool, test_result):
        """
        Update opinion based on news content, cognitive tool, and test result.

        Parameters:
        - news (str): The news content to evaluate.
        - cognitive_tool (str): The cognitive tool or theory used to evaluate the news content.
        - test_result (str): The result of the diagnostic test (positive or negative).
        """
        # Perform sentiment analysis
        sentiment_score = analyze_sentiment(news)
        # Perform relevance analysis
        relevance_score = analyze_relevance(news)

        # Update opinion based on news content and cognitive tool
        if cognitive_tool == "Neoliberal Capitalism":
            # Neoliberal Capitalism strategy
            if sentiment_score >= 0:  # Positive sentiment
                self.opinion += 0.1 * relevance_score
            else:  # Negative sentiment
                self.opinion -= 0.1 * relevance_score
        elif cognitive_tool == "Marxist-Socialism":
            # Marxist-Socialism strategy
            if sentiment_score >= 0:  # Positive sentiment
                self.opinion -= 0.1 * relevance_score
            else:  # Negative sentiment
                self.opinion += 0.1 * relevance_score

        # Perform Bayesian update based on the test result
        if test_result == "positive":
            # Bayesian update based on positive test result
            self.opinion = bayesian_calculation(self.opinion, 0.95, 0.10, 0.015)  # Example likelihoods and evidence for positive result
        elif test_result == "negative":
            # Bayesian update based on negative test result
            self.opinion = bayesian_calculation(self.opinion, 0.05, 0.90, 0.985)  # Example likelihoods and evidence for negative result

        # Ensure the updated opinion is within the range [0, 1]
        self.opinion = max(0.0, min(1.0, self.opinion))



class LearningAgent(Agent):
    def __init__(self, name, opinion):
        super().__init__(name, opinion)
        self.learning_rate = 0.1

    def update_opinion(self, news, cognitive_tool):
        """
        Update opinion based on news content using learning strategy.

        Parameters:
        - news (str): The news content to evaluate.
        - cognitive_tool (str): The cognitive tool or theory used to evaluate the news content.
        """
        # Perform sentiment analysis
        sentiment_score = analyze_sentiment(news)
        # Perform relevance analysis
        relevance_score = analyze_relevance(news)

        # Update opinion based on learning strategy
        if sentiment_score >= 0:  # Positive sentiment
            self.opinion += self.learning_rate * relevance_score
        else:  # Negative sentiment
            self.opinion -= self.learning_rate * relevance_score

        # Ensure the updated opinion is within the range [0, 1]
        self.opinion = max(0.0, min(1.0, self.opinion)))

class HybridAgent(Agent):
    def __init__(self, name, opinion):
        super().__init__(name, opinion)

    def update_opinion(self, news, cognitive_tool):
        """
        Update opinion based on news content using hybrid strategy.

        Parameters:
        - news (str): The news content to evaluate.
        - cognitive_tool (str): The cognitive tool or theory used to evaluate the news content.
        """
        # Perform sentiment analysis
        sentiment_score = analyze_sentiment(news)
        # Perform relevance analysis
        relevance_score = analyze_relevance(news)

        # Apply hybrid strategy
        if cognitive_tool == "Hybrid Approach":
            # Hybrid approach combining model-based reflex and learning
            model_based_reflex_opinion = self.opinion
            learning_opinion = self.opinion

            # Update opinion using model-based reflex
            if sentiment_score >= 0:  # Positive sentiment
                model_based_reflex_opinion += 0.1 * relevance_score
            else:  # Negative sentiment
                model_based_reflex_opinion -= 0.1 * relevance_score

            # Update opinion using learning strategy
            if sentiment_score >= 0:  # Positive sentiment
                learning_opinion += 0.1 * relevance_score
            else:  # Negative sentiment
                learning_opinion -= 0.1 * relevance_score

            # Combine opinions from both strategies
            self.opinion = (model_based_reflex_opinion + learning_opinion) / 2

        # Ensure the updated opinion is within the range [0, 1]
        self.opinion = max(0.0, min(1.0, self.opinion)))

def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text using a deep learning model.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - sentiment_score (float): The sentiment score ranging from 0 (most negative) to 1 (most positive).
    """
    # Perform sentiment analysis using the pre-trained sentiment analysis model
    result = sentiment_pipeline(text)[0]
    sentiment_score = result["score"]
    return sentiment_score

def analyze_relevance(text):
    """
    Perform relevance analysis on the given text using a deep learning model.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - relevance_score (float): The relevance score representing the importance of the text.
    """
    # Tokenize the text using spaCy
    tokens = [token.text for token in nlp(text) if not token.is_punct and not token.is_stop and token.is_alpha]
    
    # Convert tokens to BERT input format
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [1] * len(indexed_tokens)  # Single sequence for relevance analysis

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Forward pass, get hidden states from BERT model
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
    
    # Use the final layer's hidden states to compute relevance scores
    relevance_scores = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

    # Aggregate relevance scores to obtain overall relevance score
    relevance_score = sum(relevance_scores) / len(relevance_scores)
    return relevance_score

# Create an agent with dynamic cognitive tool determination
class DynamicCognitiveAgent(Agent):
    def update_opinion(self, news, input_theory):
        """
        Update opinion based on news content and dynamically determined cognitive tool.

        Parameters:
        - news (str): The news content to evaluate.
        - input_theory (str): The input theory used to determine the cognitive tool.
        """
        # Determine cognitive tool using large language model
        cognitive_tool = determine_cognitive_tool(input_theory)

        # Choose appropriate agent based on cognitive tool
        if cognitive_tool == "Neoliberal Capitalism" or cognitive_tool == "Marxist-Socialism":
            agent = ModelBasedReflexAgent(self.name, self.opinion)
        elif cognitive_tool == "Hybrid Approach":
            agent = HybridAgent(self.name, self.opinion)
        else:
            agent = LearningAgent(self.name, self.opinion)

        # Update opinion using selected agent
        agent.update_opinion(news, cognitive_tool)
        self.opinion = agent.opinion

class CognitiveTool:
    """A class representing a cognitive tool for reasoning."""
    def __init__(self, model):
        """
        Initializes a CognitiveTool.

        Args:
            model: The large language model used for reasoning.
        """
        self.model = model

    def generate_reasoning(self, input_theory, news_content):
        """
        Generates reasoning based on input theory and news content.

        Args:
            input_theory (str): The input theory provided by the user.
            news_content (str): The news content to evaluate.

        Returns:
            str: The reasoning generated by the cognitive tool.
        """
        # Placeholder implementation, replace with actual rule-based expert system
        reasoning = []
        return reasoning


##################################


# Sample usage of the DynamicCognitiveAgent
agent_dynamic = DynamicCognitiveAgent(name="Dynamic Agent", opinion=0.5)

# Sample news content and input theory
news_content = []
input_theory = []

# Update opinion for the DynamicCognitiveAgent
agent_dynamic.update_opinion(news_content, input_theory)