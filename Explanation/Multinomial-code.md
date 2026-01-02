# الگوریتم چندجمله ای بیز ساده
این کد یک طبقه‌بند چندجمله‌ای بیز ساده برای تشخیص هرزنامه (Spam/Ham) پیاده‌سازی می‌کند. الگوریتم بر اساس فرض استقلال کلمات عمل می‌کند و از توزیع چندجمله‌ای برای مدل‌سازی فراوانی کلمات در اسناد استفاده می‌کند. 
کد به صورت آموزشی طراحی شده و مراحل محاسبه را به طور کامل نمایش می‌دهد.

# بخش ۱: آماده‌سازی و کتابخانه‌ها
```
import math
from collections import Counter

def fmt(x):
    """Format numbers for better display"""
    return f"{x:.4f}"
```

الگوریتم:

برای محاسبات لگاریتمی و نمایی نیاز به کتابخانه math داریم

برای نمایش خوانا اعداد، یک تابع فرمت‌کننده تعریف می‌شود

پیاده‌سازی:

 کتابخانه math: برای توابع ریاضی مانند log() و exp()

 تابع fmt(): اعداد را با ۴ رقم اعشار نمایش می‌دهد.

 # بخش ۲: داده‌های آموزشی

 ```
vocabulary = ["free", "win", "money", "meeting", "project", "urgent"]
V = len(vocabulary)  # Vocabulary size

spam_docs = [
    [3, 2, 4, 0, 0, 1],   # Document 1
    [2, 1, 3, 0, 0, 0],   # Document 2
    [4, 3, 5, 0, 1, 2]    # Document 3
]

ham_docs = [
    [0, 0, 1, 3, 2, 0],   # Document 1
    [1, 0, 0, 2, 3, 1],   # Document 2
    [0, 1, 0, 4, 3, 2]    # Document 3
]

test_doc = [2, 1, 3, 0, 0, 1]  # Similar to spam
```
الگوریتم:

واژگان ثابتی از کلمات مهم تعریف می‌شود.

هر سند به صورت بردار فراوانی کلمات نمایش داده می‌شود.

داده‌ها به دو کلاس اسپم و غیراسپم تقسیم می‌شوند.

یک سند تست برای طبقه‌بندی تعریف می‌شود.

پیاده‌سازی:

 بخش vocabulary: لیست ۶ کلمه کلیدی

بخش spam_docs/ham_docs: ماتریس‌های ۳×۶ از فراوانی کلمات

بخش test_doc: سند آزمایشی با ۲ بار "free" و 1 بار "win" و 3 بار "money" و ۱ بار "urgent"

# بخش ۳: محاسبه پارامترهای چندجمله‌ای

```
def calculate_multinomial_params(docs, class_name, alpha=1):
    total_words = sum(sum(doc) for doc in docs)
    word_counts = [0] * V
    
    for doc in docs:
        for i in range(V):
            word_counts[i] += doc[i]
    
    probabilities = []
    for i in range(V):
        count = word_counts[i]
        prob = (count + alpha) / (total_words + alpha * V)
        probabilities.append(prob)
    
    return probabilities, total_words
```
الگوریتم:

برای هر کلاس، احتمال هر کلمه با هموارسازی لاپلاس( Laplace Smoothing ) محاسبه می‌شود

فرمول: P(word|class) = (count(word,class) + α) / (total_words_in_class + α × V)

هموارسازی لاپلاس (α=1) برای جلوگیری از احتمال صفر

پیاده‌سازی:

بخش total_words: مجموع تمام کلمات در کلاس

بخش word_counts: تعداد هر کلمه در کلاس

محاسبه احتمال با فرمول هموارسازی

# بخش ۴: محاسبه درست‌نمایی (Likelihood)

```
def multinomial_likelihood(doc, word_probs, class_name):
    log_likelihood = 0
    
    for i in range(V):
        count = doc[i]
        prob = word_probs[i]
        if count > 0 and prob > 0:
            log_term = count * math.log(prob)
            log_likelihood += log_term
    
    likelihood = math.exp(log_likelihood)
    return likelihood, log_likelihood
```
لگوریتم:

درست‌نمایی سند با فرض استقلال کلمات: P(doc|class) = ∏ P(word|class)^count

به دلیل اعداد کوچک، از لگاریتم استفاده می‌شود: log(P) = Σ count × log(P(word|class))

پیاده‌سازی:

برای هر کلمه: count × log(probability) جمع می‌شود.

نتیجه نهایی با exp() به احتمال تبدیل می‌شود.

# بخش ۵: احتمالات پیشین (Prior)
```
n_spam = len(spam_docs)
n_ham = len(ham_docs)
n_total = n_spam + n_ham

p_spam_prior = n_spam / n_total
p_ham_prior = n_ham / n_total
```
الگوریتم:

احتمالات پیشین بر اساس نسبت اسناد هر کلاس محاسبه می‌شوند:

P(class) = تعداد_اسناد_کلاس / کل_اسناد

پیاده‌سازی:

p_spam_prior = 3/6 = 0.5

p_ham_prior = 3/6 = 0.5

# بخش ۶: محاسبه احتمالات پسین (Posterior)

```
log_posterior_spam = log_p_x_spam + math.log(p_spam_prior)
log_posterior_ham = log_p_x_ham + math.log(p_ham_prior)

# نرمال‌سازی با log-sum-exp
max_log = max(log_posterior_spam, log_posterior_ham)
log_sum = max_log + math.log(math.exp(log_posterior_spam - max_log) + 
                            math.exp(log_posterior_ham - max_log))

p_spam_given_x = math.exp(log_posterior_spam - log_sum)
p_ham_given_x = math.exp(log_posterior_ham - log_sum)
```

الگوریتم:

قاعده بیز: P(class|doc) ∝ P(doc|class) × P(class)

در فضای لگاریتمی: log(P(class|doc)) = log(P(doc|class)) + log(P(class))

نرمال‌سازی با ترفند log-sum-exp برای جلوگیری از سرریز عددی

پیاده‌سازی:

جمع لگاریتم‌ها برای محاسبه لگاریتم احتمال پسین

استفاده از log-sum-exp برای محاسبه مطمئن جمع نمایی‌ها

# بخش ۷: تصمیم‌گیری نهایی

```
if p_spam_given_x > p_ham_given_x:
    decision = "SPAM"
    confidence = (p_spam_given_x - p_ham_given_x) / p_spam_given_x
else:
    decision = "HAM"
    confidence = (p_ham_given_x - p_spam_given_x) / p_ham_given_x
```

الگوریتم:

کلاس با بیشترین احتمال پسین انتخاب می‌شود

میزان اطمینان بر اساس تفاوت نسبی احتمالات محاسبه می‌شود

پیاده‌سازی:

مقایسه p_spam_given_x و p_ham_given_x

محاسبه اطمینان: |P1 - P2| / max(P1, P2)

## خروجی نمونه الگوریتم

برای سند تست [2, 1, 3, 0, 0, 1]:

احتمال اسپم: ≈ 99.98%

احتمال غیراسپم: ≈ 0.02%

تصمیم: SPAM با اطمینان ≈ 99.96%

## نقاط قوت الگوریتم

سادگی: پیاده‌سازی و درک آسان

کارایی: محاسبات سریع حتی با داده‌های بزرگ

مقاومت در برابر نویز: هموارسازی لاپلاس از اضافه‌برازش جلوگیری می‌کند

مبتنی بر احتمال: خروجی تفسیرپذیر و دارای میزان اطمینان

## محدودیت‌ها

فرض استقلال: کلمات در واقعیت مستقل نیستند

حساسیت به انتخاب ویژگی: وابسته به واژگان از پیش تعریف شده

توزیع چندجمله‌ای: فرض می‌کند طول اسناد مهم نیست
