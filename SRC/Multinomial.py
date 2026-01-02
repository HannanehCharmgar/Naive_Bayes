import math
from collections import Counter

def fmt(x):
    """Format numbers for better display"""
    return f"{x:.4f}"

print("\n" + "="*80)
print("📘 Multinomial Naive Bayes - Educational Version (For Text/Count Data)")
print("="*80 + "\n")

# ==================== Section 1: Training Data ====================
print("🔹 **Section 1: Training Data - Text Documents**\n")

# Training data: word counts in documents
# Vocabulary: ["free", "win", "money", "meeting", "project", "urgent"]
# Each document represented as word frequency vector
spam_docs = [
    [3, 2, 4, 0, 0, 1],   # Document 1: high frequency of "free", "win", "money"
    [2, 1, 3, 0, 0, 0],   # Document 2
    [4, 3, 5, 0, 1, 2]    # Document 3
]

ham_docs = [
    [0, 0, 1, 3, 2, 0],   # Document 1: business terms
    [1, 0, 0, 2, 3, 1],   # Document 2
    [0, 1, 0, 4, 3, 2]    # Document 3
]

vocabulary = ["free", "win", "money", "meeting", "project", "urgent"]
V = len(vocabulary)  # Vocabulary size

print("Vocabulary: " + ", ".join(vocabulary))
print(f"Vocabulary size (V) = {V}\n")

print("Spam documents (word frequencies):")
print("Doc | " + " | ".join(vocabulary))
for i, doc in enumerate(spam_docs):
    print(f" {i+1}  | " + " | ".join(f"{count:^6}" for count in doc))

print("\nHam documents (word frequencies):")
print("Doc | " + " | ".join(vocabulary))
for i, doc in enumerate(ham_docs):
    print(f" {i+1}  | " + " | ".join(f"{count:^6}" for count in doc))

# Test document
test_doc = [2, 1, 3, 0, 0, 1]  # Similar to spam
print(f"\n📄 **Test Document Word Counts:**")
for i, word in enumerate(vocabulary):
    print(f"  {word:8s}: {test_doc[i]}")

# ==================== Section 2: Calculating Parameters ====================
print("\n" + "-"*80)
print("🔹 **Section 2: Calculating Multinomial Parameters with Laplace Smoothing**\n")

def calculate_multinomial_params(docs, class_name, alpha=1):
    """Calculate word probabilities for multinomial distribution"""
    total_words = sum(sum(doc) for doc in docs)
    n_docs = len(docs)
    
    # Count total occurrences of each word in class
    word_counts = [0] * V
    for doc in docs:
        for i in range(V):
            word_counts[i] += doc[i]
    
    print(f"📊 **For {class_name} class:**")
    print(f"  Total words in all documents: {total_words}")
    print(f"  Number of documents: {n_docs}")
    
    # Calculate probabilities with Laplace smoothing
    probabilities = []
    for i in range(V):
        count = word_counts[i]
        prob = (count + alpha) / (total_words + alpha * V)
        probabilities.append(prob)
        
        print(f"\n  P('{vocabulary[i]}' | {class_name}):")
        print(f"    Count of '{vocabulary[i]}' in {class_name}: {count}")
        print(f"    Total words in {class_name}: {total_words}")
        print(f"    Vocabulary size (V): {V}")
        print(f"    Laplace smoothing with α = {alpha}:")
        print(f"    Formula: (count + α) / (total_words + α × V)")
        print(f"    Calculation: ({count} + {alpha}) / ({total_words} + {alpha} × {V})")
        print(f"    = {count + alpha} / {total_words + alpha * V}")
        print(f"    = {fmt(prob)}")
    
    return probabilities, total_words

alpha = 1  # Laplace smoothing parameter
print(f"Using Laplace smoothing parameter α = {alpha}\n")

p_words_spam, total_spam_words = calculate_multinomial_params(spam_docs, "spam", alpha)
print("\n" + "-"*40)
p_words_ham, total_ham_words = calculate_multinomial_params(ham_docs, "ham", alpha)

# ==================== Section 3: Likelihood Calculation ====================
print("\n" + "-"*80)
print("🔹 **Section 3: Calculating Document Likelihood**\n")

def multinomial_likelihood(doc, word_probs, class_name):
    """Calculate multinomial likelihood with log probabilities"""
    likelihood = 0
    log_likelihood = 0
    
    print(f"📌 **Calculating P(x | {class_name}):**")
    print("  Multinomial formula: P(x | class) ∝ Π_i [P(word_i | class)^{count_i}]")
    print("  Where Π (pi) means product over all words")
    print("  In practice, we use log to avoid underflow:")
    print("  log(P(x | class)) = Σ_i [count_i × log(P(word_i | class))]")
    print()
    
    print("  Word-by-word calculation:")
    for i in range(V):
        count = doc[i]
        prob = word_probs[i]
        if count > 0 and prob > 0:
            term = (prob ** count)
            log_term = count * math.log(prob)
            likelihood_term = term
            log_likelihood += log_term
            
            print(f"    '{vocabulary[i]}': count={count}, P={fmt(prob)}")
            print(f"      Contribution: {fmt(prob)}^{count} = {fmt(term)}")
            print(f"      Log contribution: {count} × log({fmt(prob)}) = {fmt(log_term)}")
    
    likelihood = math.exp(log_likelihood)
    print(f"\n  Total log-likelihood = {fmt(log_likelihood)}")
    print(f"  Likelihood (exp of log-likelihood) = e^{fmt(log_likelihood)} = {fmt(likelihood)}")
    
    return likelihood, log_likelihood

print("🎯 **For Spam class:**")
p_x_spam, log_p_x_spam = multinomial_likelihood(test_doc, p_words_spam, "spam")

print("\n🎯 **For Ham class:**")
p_x_ham, log_p_x_ham = multinomial_likelihood(test_doc, p_words_ham, "ham")

# ==================== Section 4: Prior Probabilities ====================
print("\n" + "-"*80)
print("🔹 **Section 4: Prior Probabilities**\n")

n_spam = len(spam_docs)
n_ham = len(ham_docs)
n_total = n_spam + n_ham

p_spam_prior = n_spam / n_total
p_ham_prior = n_ham / n_total

print("Prior probabilities based on document frequency:")
print(f"  Total documents: {n_total}")
print(f"  Spam documents: {n_spam}, Ham documents: {n_ham}")
print(f"  P(spam) = {n_spam} / {n_total} = {fmt(p_spam_prior)}")
print(f"  P(ham) = {n_ham} / {n_total} = {fmt(p_ham_prior)}")

# ==================== Section 5: Posterior Calculation ====================
print("\n" + "-"*80)
print("🔹 **Section 5: Posterior Probability Calculation**\n")

print("Using log probabilities for numerical stability:")
print("  log(P(spam | x)) ∝ log(P(x | spam)) + log(P(spam))")
print("  log(P(ham | x)) ∝ log(P(x | ham)) + log(P(ham))")

log_posterior_spam = log_p_x_spam + math.log(p_spam_prior)
log_posterior_ham = log_p_x_ham + math.log(p_ham_prior)

print(f"\n📊 **Log Posteriors:**")
print(f"  log(P(spam | x)) = {fmt(log_p_x_spam)} + log({fmt(p_spam_prior)})")
print(f"                    = {fmt(log_p_x_spam)} + {fmt(math.log(p_spam_prior))}")
print(f"                    = {fmt(log_posterior_spam)}")

print(f"\n  log(P(ham | x)) = {fmt(log_p_x_ham)} + log({fmt(p_ham_prior)})")
print(f"                   = {fmt(log_p_x_ham)} + {fmt(math.log(p_ham_prior))}")
print(f"                   = {fmt(log_posterior_ham)}")

# Convert back to probabilities (using log-sum-exp trick)
max_log = max(log_posterior_spam, log_posterior_ham)
log_sum = max_log + math.log(math.exp(log_posterior_spam - max_log) + 
                            math.exp(log_posterior_ham - max_log))

p_spam_given_x = math.exp(log_posterior_spam - log_sum)
p_ham_given_x = math.exp(log_posterior_ham - log_sum)

print(f"\n📊 **Normalized Probabilities (using log-sum-exp):**")
print(f"  max(log values) = {fmt(max_log)}")
print(f"  log(P(x)) = {fmt(log_sum)}")
print(f"  P(spam | x) = exp({fmt(log_posterior_spam)} - {fmt(log_sum)})")
print(f"              = exp({fmt(log_posterior_spam - log_sum)})")
print(f"              = {fmt(p_spam_given_x)} = {fmt(p_spam_given_x*100)}%")

print(f"\n  P(ham | x) = exp({fmt(log_posterior_ham)} - {fmt(log_sum)})")
print(f"             = exp({fmt(log_posterior_ham - log_sum)})")
print(f"             = {fmt(p_ham_given_x)} = {fmt(p_ham_given_x*100)}%")

# ==================== Section 6: Final Decision ====================
print("\n" + "-"*80)
print("🔹 **Section 6: Final Decision**\n")

print("🎯 **Classification Results:**")
print(f"  P(spam | document) = {fmt(p_spam_given_x*100)}%")
print(f"  P(ham | document)  = {fmt(p_ham_given_x*100)}%")

if p_spam_given_x > p_ham_given_x:
    decision = "SPAM"
    confidence = (p_spam_given_x - p_ham_given_x) / p_spam_given_x
    print(f"\n✅ **Decision: {decision}** (higher probability of being spam)")
else:
    decision = "HAM"
    confidence = (p_ham_given_x - p_spam_given_x) / p_ham_given_x
    print(f"\n✅ **Decision: {decision}** (higher probability of being legitimate)")

print(f"📊 **Confidence: {fmt(confidence*100)}%**")

# Show word analysis
print(f"\n🔍 **Word Analysis for Test Document:**")
for i, word in enumerate(vocabulary):
    if test_doc[i] > 0:
        print(f"  '{word}': {test_doc[i]} occurrences")
        print(f"    P('{word}' | spam) = {fmt(p_words_spam[i])}")
        print(f"    P('{word}' | ham)  = {fmt(p_words_ham[i])}")

print("\n" + "="*80)
print("✅ Multinomial Naive Bayes Text Classification Completed")
print("="*80)