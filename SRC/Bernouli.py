import math

def fmt(x):
    """Format numbers for better display"""
    return f"{x:.4f}"

print("\n" + "="*60)
print("📘 Bernoulli Naive Bayes - Educational Version")
print("="*60 + "\n")

# ==================== Section 1: Training Data ====================
print("🔹 **Section 1: Training Data**\n")

# Training data for spam and ham
# [presence of 'free', presence of 'win']
spam = [[1, 1], [1, 0], [1, 1]]
ham  = [[0, 0], [0, 1], [0, 0]]

print("Spam training data:")
print("   Email | free | win")
for i, email in enumerate(spam):
    print(f"   Email{i+1} |   {email[0]}   |  {email[1]}")

print("\nHam training data:")
print("   Email | free | win")
for i, email in enumerate(ham):
    print(f"   Email{i+1} |   {email[0]}   |  {email[1]}")

# Test email
x = [1, 0]
print(f"\n📧 **Test Email:** free={x[0]}, win={x[1]}")

# ==================== Section 2: Feature Probability Calculation ====================
print("\n" + "-"*60)
print("🔹 **Section 2: Feature Probability Calculation with Laplacian Smoothing**\n")

def feature_probability(data, index, feature_name, class_name):
    """Calculate feature probability with Laplacian Smoothing"""
    count = sum(row[index] for row in data)
    n = len(data)
    
    print(f"Calculating P({feature_name}=1 | {class_name}):")
    print(f"  Number of {class_name} emails with {feature_name}=1: {count}")
    print(f"  Total number of {class_name} emails: {n}")
    print(f"  Formula: (count + 1) / (n + 2)")
    print(f"  Calculation: ({count} + 1) / ({n} + 2) = {count+1} / {n+2}")
    
    probability = (count + 1) / (n + 2)
    print(f"  Result: {fmt(probability)}\n")
    
    return probability

# Calculate probabilities for each feature
print("📊 **For Spam class:**")
p_free_spam = feature_probability(spam, 0, "free", "spam")
p_win_spam = feature_probability(spam, 1, "win", "spam")

print("📊 **For Ham class:**")
p_free_ham = feature_probability(ham, 0, "free", "ham")
p_win_ham = feature_probability(ham, 1, "win", "ham")

# ==================== Section 3: Conditional Probability Calculation ====================
print("-"*60)
print("🔹 **Section 3: Calculating P(x | class)**\n")

print("Formula: P(x | class) = PRODUCT of all P(feature_i | class)")
print("The symbol Π (capital pi) means 'product' - multiply all terms together")
print(f"For test email: x = [free={x[0]}, win={x[1]}]")

print("\n📌 **For Spam class:**")
if x[0] == 1:
    prob_free_spam = p_free_spam
    print(f"  P(free=1 | spam) = {fmt(p_free_spam)}")
else:
    prob_free_spam = 1 - p_free_spam
    print(f"  P(free=0 | spam) = 1 - P(free=1 | spam) = 1 - {fmt(p_free_spam)} = {fmt(prob_free_spam)}")

if x[1] == 1:
    prob_win_spam = p_win_spam
    print(f"  P(win=1 | spam)  = {fmt(p_win_spam)}")
else:
    prob_win_spam = 1 - p_win_spam
    print(f"  P(win=0 | spam)  = 1 - P(win=1 | spam) = 1 - {fmt(p_win_spam)} = {fmt(prob_win_spam)}")

p_x_spam = prob_free_spam * prob_win_spam
print(f"\n  P(x | spam) = P(free={x[0]} | spam) × P(win={x[1]} | spam)")
print(f"              = {fmt(prob_free_spam)} × {fmt(prob_win_spam)}")
print(f"              = {fmt(p_x_spam)}")

print("\n📌 **For Ham class:**")
if x[0] == 1:
    prob_free_ham = p_free_ham
    print(f"  P(free=1 | ham)  = {fmt(p_free_ham)}")
else:
    prob_free_ham = 1 - p_free_ham
    print(f"  P(free=0 | ham)  = 1 - P(free=1 | ham) = 1 - {fmt(p_free_ham)} = {fmt(prob_free_ham)}")

if x[1] == 1:
    prob_win_ham = p_win_ham
    print(f"  P(win=1 | ham)   = {fmt(p_win_ham)}")
else:
    prob_win_ham = 1 - p_win_ham
    print(f"  P(win=0 | ham)   = 1 - P(win=1 | ham) = 1 - {fmt(p_win_ham)} = {fmt(prob_win_ham)}")

p_x_ham = prob_free_ham * prob_win_ham
print(f"\n  P(x | ham) = P(free={x[0]} | ham) × P(win={x[1]} | ham)")
print(f"             = {fmt(prob_free_ham)} × {fmt(prob_win_ham)}")
print(f"             = {fmt(p_x_ham)}")

# ==================== Section 4: Prior Probabilities ====================
print("\n" + "-"*60)
print("🔹 **Section 4: Incorporating Prior Probabilities (Priors)**\n")

p_spam = 0.5
p_ham = 0.5

print("Assuming balanced training data:")
print(f"  P(spam) = {p_spam}")
print(f"  P(ham)  = {p_ham}")

print("\n📌 **Calculating Posterior (without normalization):**")
print("  Bayes formula: P(class | x) is PROPORTIONAL TO P(x | class) × P(class)")
print("  The symbol ∝ means 'is proportional to' (we'll normalize later)")

posterior_spam = p_x_spam * p_spam
posterior_ham = p_x_ham * p_ham

print(f"\n  For spam: P(spam | x) ∝ P(x | spam) × P(spam)")
print(f"           ∝ {fmt(p_x_spam)} × {fmt(p_spam)}")
print(f"           ∝ {fmt(posterior_spam)}")

print(f"\n  For ham:  P(ham | x) ∝ P(x | ham) × P(ham)")
print(f"           ∝ {fmt(p_x_ham)} × {fmt(p_ham)}")
print(f"           ∝ {fmt(posterior_ham)}")

# ==================== Section 5: Final Decision ====================
print("\n" + "-"*60)
print("🔹 **Section 5: Final Decision**\n")

print("Comparing unnormalized posterior probabilities:")
print(f"  P(spam | x) ∝ {fmt(posterior_spam)}")
print(f"  P(ham | x)  ∝ {fmt(posterior_ham)}")

print("\n📌 **Calculating normalized probabilities:**")
total = posterior_spam + posterior_ham
print(f"  Total probability = P(spam | x) + P(ham | x)")
print(f"                     = {fmt(posterior_spam)} + {fmt(posterior_ham)}")
print(f"                     = {fmt(total)}")

if total > 0:
    normalized_spam = posterior_spam / total
    normalized_ham = posterior_ham / total
    print(f"\n  P(spam | x) = P(spam | x) / Total")
    print(f"               = {fmt(posterior_spam)} / {fmt(total)}")
    print(f"               = {fmt(normalized_spam)}")
    print(f"               = {fmt(normalized_spam*100)}%")
    
    print(f"\n  P(ham | x) = P(ham | x) / Total")
    print(f"              = {fmt(posterior_ham)} / {fmt(total)}")
    print(f"              = {fmt(normalized_ham)}")
    print(f"              = {fmt(normalized_ham*100)}%")
else:
    normalized_spam = normalized_ham = 0

print("\n🎯 **Final Result:**")
if posterior_spam > posterior_ham:
    decision = "Spam"
    # Confidence calculation explained
    print(f"  P(spam | x) > P(ham | x), so email is classified as: {decision}")
    print(f"\n  📊 **Confidence Calculation:**")
    print(f"    Confidence = |P(spam|x) - P(ham|x)| / max(P(spam|x), P(ham|x))")
    print(f"    Confidence = |{fmt(posterior_spam)} - {fmt(posterior_ham)}| / max({fmt(posterior_spam)}, {fmt(posterior_ham)})")
    print(f"    Confidence = |{fmt(posterior_spam - posterior_ham)}| / {fmt(posterior_spam)}")
    
    confidence = abs(posterior_spam - posterior_ham) / max(posterior_spam, posterior_ham)
    print(f"    Confidence = {fmt(confidence)}")
    print(f"    Confidence = {fmt(confidence*100)}%")
    
    print(f"\n  Decision: {decision} (higher probability of being spam)")
    print(f"  Confidence: {fmt(confidence*100)}%")
else:
    decision = "Ham"
    # Confidence calculation explained
    print(f"  P(ham | x) > P(spam | x), so email is classified as: {decision}")
    print(f"\n  📊 **Confidence Calculation:**")
    print(f"    Confidence = |P(spam|x) - P(ham|x)| / max(P(spam|x), P(ham|x))")
    print(f"    Confidence = |{fmt(posterior_spam)} - {fmt(posterior_ham)}| / max({fmt(posterior_spam)}, {fmt(posterior_ham)})")
    print(f"    Confidence = |{fmt(posterior_spam - posterior_ham)}| / {fmt(posterior_ham)}")
    
    confidence = abs(posterior_spam - posterior_ham) / max(posterior_spam, posterior_ham)
    print(f"    Confidence = {fmt(confidence)}")
    print(f"    Confidence = {fmt(confidence*100)}%")
    
    print(f"\n  Decision: {decision} (higher probability of being non-spam)")
    print(f"  Confidence: {fmt(confidence*100)}%")

print("\n" + "="*60)
print("✅ Bernoulli Naive Bayes training and testing completed")
print("="*60)