import math
import numpy as np

def fmt(x):
    """Format numbers for better display"""
    return f"{x:.4f}"

print("\n" + "="*80)
print("📘 Gaussian Naive Bayes - Educational Version (For Continuous Data)")
print("="*80 + "\n")

# ==================== Section 1: Training Data ====================
print("🔹 **Section 1: Training Data**\n")

# Training data for classifying fruits based on weight and sugar content
# [weight (grams), sugar_content (%)]
apple = [[150, 12], [160, 11], [170, 13], [155, 12.5]]
banana = [[120, 18], [130, 19], [125, 17.5], [135, 18.5]]

print("Apple training data (class 1):")
print("   Fruit  | Weight(g) | Sugar(%)")
for i, fruit in enumerate(apple):
    print(f"   Apple{i+1} |    {fruit[0]}     |    {fruit[1]}")

print("\nBanana training data (class 2):")
print("   Fruit   | Weight(g) | Sugar(%)")
for i, fruit in enumerate(banana):
    print(f"   Banana{i+1} |    {fruit[0]}     |    {fruit[1]}")

# Test fruit
x_test = [140, 15]
print(f"\n🍎 **Test Fruit:** weight={x_test[0]}g, sugar_content={x_test[1]}%")

# ==================== Section 2: Calculating Gaussian Parameters ====================
print("\n" + "-"*80)
print("🔹 **Section 2: Calculating Gaussian Distribution Parameters**\n")

def calculate_gaussian_params(data, class_name, feature_names):
    """Calculate mean and standard deviation for each feature"""
    n = len(data)
    k = len(data[0])  # number of features
    
    means = []
    stds = []
    
    for j in range(k):
        feature_values = [row[j] for row in data]
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values) if n > 1 else 0.0001  # Avoid division by zero
        
        means.append(mean_val)
        stds.append(std_val)
        
        print(f"📊 **For {class_name} - {feature_names[j]}:**")
        print(f"  Feature values: {feature_values}")
        print(f"  Mean (μ_{class_name[0].lower()}{j+1}) = Σx_i / n = {sum(feature_values)} / {n} = {fmt(mean_val)}")
        print(f"  Standard Deviation (σ_{class_name[0].lower()}{j+1}) = √[Σ(x_i - μ)²/(n-1)] = {fmt(std_val)}")
        print()
    
    return means, stds

feature_names = ["Weight", "Sugar Content"]
apple_means, apple_stds = calculate_gaussian_params(apple, "Apple", feature_names)
banana_means, banana_stds = calculate_gaussian_params(banana, "Banana", feature_names)

# ==================== Section 3: Gaussian Probability Density Function ====================
print("-"*80)
print("🔹 **Section 3: Gaussian Probability Density Function (PDF)**\n")

def gaussian_pdf(x, mean, std):
    """Calculate Gaussian PDF with explanation"""
    if std == 0:
        std = 0.0001  # Prevent division by zero
    
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    coefficient = 1 / (std * math.sqrt(2 * math.pi))
    pdf_value = coefficient * math.exp(exponent)
    
    print(f"  Gaussian PDF formula: f(x; μ, σ) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))")
    print(f"  Where:")
    print(f"    μ = {fmt(mean)} (mean)")
    print(f"    σ = {fmt(std)} (standard deviation)")
    print(f"    x = {x} (feature value)")
    print(f"  Calculation:")
    print(f"    (x - μ)² = ({x} - {fmt(mean)})² = {fmt((x - mean) ** 2)}")
    print(f"    2σ² = 2 * ({fmt(std)})² = {fmt(2 * (std ** 2))}")
    print(f"    Exponent = -{fmt((x - mean) ** 2)} / {fmt(2 * (std ** 2))} = {fmt(exponent)}")
    print(f"    e^(exponent) = e^({fmt(exponent)}) = {fmt(math.exp(exponent))}")
    print(f"    Coefficient = 1 / ({fmt(std)} * √(2π)) = 1 / ({fmt(std)} * {fmt(math.sqrt(2 * math.pi))})")
    print(f"    Coefficient = 1 / {fmt(std * math.sqrt(2 * math.pi))} = {fmt(coefficient)}")
    print(f"    PDF = {fmt(coefficient)} * {fmt(math.exp(exponent))} = {fmt(pdf_value)}")
    print()
    
    return pdf_value

print("📌 **Calculating P(x | Apple) using Gaussian PDF:**")
p_x_apple = 1.0
for i, (x_val, mean, std) in enumerate(zip(x_test, apple_means, apple_stds)):
    print(f"\n  For feature '{feature_names[i]}' = {x_val}:")
    pdf_val = gaussian_pdf(x_val, mean, std)
    p_x_apple *= pdf_val

print(f"📊 **P(x | Apple) = PRODUCT of individual feature probabilities:**")
print(f"                 = {fmt(p_x_apple)}")

print("\n" + "="*40)
print("📌 **Calculating P(x | Banana) using Gaussian PDF:**")
p_x_banana = 1.0
for i, (x_val, mean, std) in enumerate(zip(x_test, banana_means, banana_stds)):
    print(f"\n  For feature '{feature_names[i]}' = {x_val}:")
    pdf_val = gaussian_pdf(x_val, mean, std)
    p_x_banana *= pdf_val

print(f"📊 **P(x | Banana) = PRODUCT of individual feature probabilities:**")
print(f"                  = {fmt(p_x_banana)}")

# ==================== Section 4: Prior Probabilities ====================
print("\n" + "-"*80)
print("🔹 **Section 4: Incorporating Prior Probabilities**\n")

n_apple = len(apple)
n_banana = len(banana)
n_total = n_apple + n_banana

p_apple_prior = n_apple / n_total
p_banana_prior = n_banana / n_total

print(f"Prior probabilities based on training data frequency:")
print(f"  Total fruits: {n_total}")
print(f"  Apples: {n_apple}, Bananas: {n_banana}")
print(f"  P(Apple) = n_apple / n_total = {n_apple} / {n_total} = {fmt(p_apple_prior)}")
print(f"  P(Banana) = n_banana / n_total = {n_banana} / {n_total} = {fmt(p_banana_prior)}")

# ==================== Section 5: Posterior Calculation ====================
print("\n" + "-"*80)
print("🔹 **Section 5: Calculating Posterior Probabilities**\n")

print("Bayes Theorem: P(class | x) = P(x | class) × P(class) / P(x)")
print("Where P(x) = P(x | Apple)×P(Apple) + P(x | Banana)×P(Banana)")

posterior_apple = p_x_apple * p_apple_prior
posterior_banana = p_x_banana * p_banana_prior
p_x_total = posterior_apple + posterior_banana

print(f"\n📊 **Unnormalized Posteriors:**")
print(f"  P(Apple | x) ∝ P(x | Apple) × P(Apple) = {fmt(p_x_apple)} × {fmt(p_apple_prior)} = {fmt(posterior_apple)}")
print(f"  P(Banana | x) ∝ P(x | Banana) × P(Banana) = {fmt(p_x_banana)} × {fmt(p_banana_prior)} = {fmt(posterior_banana)}")
print(f"  P(x) = {fmt(posterior_apple)} + {fmt(posterior_banana)} = {fmt(p_x_total)}")

if p_x_total > 0:
    p_apple_given_x = posterior_apple / p_x_total
    p_banana_given_x = posterior_banana / p_x_total
    
    print(f"\n📊 **Normalized Posteriors:**")
    print(f"  P(Apple | x) = {fmt(posterior_apple)} / {fmt(p_x_total)} = {fmt(p_apple_given_x)}")
    print(f"  P(Banana | x) = {fmt(posterior_banana)} / {fmt(p_x_total)} = {fmt(p_banana_given_x)}")

# ==================== Section 6: Final Decision ====================
print("\n" + "-"*80)
print("🔹 **Section 6: Final Decision**\n")

print("🎯 **Probability Comparison:**")
print(f"  P(Apple | x)  = {fmt(p_apple_given_x)} = {fmt(p_apple_given_x*100)}%")
print(f"  P(Banana | x) = {fmt(p_banana_given_x)} = {fmt(p_banana_given_x*100)}%")

if p_apple_given_x > p_banana_given_x:
    decision = "Apple"
    confidence = (p_apple_given_x - p_banana_given_x) / p_apple_given_x
else:
    decision = "Banana"
    confidence = (p_banana_given_x - p_apple_given_x) / p_banana_given_x

print(f"\n✅ **Final Classification:** {decision}")
print(f"   Confidence: {fmt(confidence*100)}%")
print(f"   Weight: {x_test[0]}g, Sugar: {x_test[1]}%")

print("\n" + "="*80)
print("✅ Gaussian Naive Bayes Classification Completed")
print("="*80)