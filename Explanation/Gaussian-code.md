# طبقه‌بند گاوسی بیز ساده (Gaussian Naive Bayes)
این کد یک طبقه‌بند گاوسی بیز ساده برای داده‌های پیوسته پیاده‌سازی می‌کند. الگوریتم فرض می‌کند که ویژگی‌های هر کلاس از توزیع نرمال (گاوسی) پیروی می‌کنند. 
این روش برای داده‌های عددی پیوسته (مانند وزن، قد، دما) مناسب است و از فرض استقلال ویژگی‌ها استفاده می‌کند.

## بخش ۱: آماده‌سازی و کتابخانه‌ها

```
import math
import numpy as np

def fmt(x):
    """Format numbers for better display"""
    return f"{x:.4f}"
```
الگوریتم:

نیاز به محاسبات ریاضی برای تابع چگالی احتمال گاوسی

کتابخانه numpy برای محاسبات برداری و آماری

تابع فرمت‌کننده برای نمایش خوانا اعداد

پیاده‌سازی:

کتابخانه math: برای sqrt(), pi, exp()

کتابخانه numpy: برای mean ,() std()

تابع fmt(): نمایش ۴ رقم اعشار

## بخش ۲: داده‌های آموزشی

```
apple = [[150, 12], [160, 11], [170, 13], [155, 12.5]]
banana = [[120, 18], [130, 19], [125, 17.5], [135, 18.5]]
x_test = [140, 15]  # Test fruit: weight=140g, sugar=15%

feature_names = ["Weight", "Sugar Content"]
```

الگوریتم:

داده‌های آموزشی شامل دو کلاس: سیب و موز

هر میوه با دو ویژگی پیوسته توصیف می‌شود: وزن (گرم) و محتوای قند (%)

یک نمونه تست برای طبقه‌بندی تعریف می‌شود

پیاده‌سازی:

نمونه های apple: تعداد 4 نمونه سیب با وزن ۱۵۰-۱۷۰ گرم و قند ۱۱-۱۳%

نمونه های banana: تعداد 4 نمونه موز با وزن ۱۲۰-۱۳۵ گرم و قند ۱۷.۵-۱۹%

نمونه تست (x_test) : میوه تست با وزن ۱۴۰ گرم و قند ۱۵%

## بخش ۳: محاسبه پارامترهای توزیع گاوسی

```
def calculate_gaussian_params(data, class_name, feature_names):
    n = len(data)
    k = len(data[0])  # number of features
    
    means = []
    stds = []
    
    for j in range(k):
        feature_values = [row[j] for row in data]
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values) if n > 1 else 0.0001
        
        means.append(mean_val)
        stds.append(std_val)
    
    return means, stds
```
الگوریتم:

برای هر ویژگی در هر کلاس، میانگین (μ) و انحراف معیار (σ) محاسبه می‌شود

فرمول میانگین: μ = Σx_i / n

فرمول انحراف معیار: σ = √[Σ(x_i - μ)²/(n-1)]

اگر n=1 باشد، σ کوچک مثبت در نظر گرفته می‌شود

پیاده‌سازی:

استخراج مقادیر هر ویژگی از تمام نمونه‌ها

محاسبه میانگین و انحراف معیار با توابع numpy

ذخیره پارامترها برای هر ویژگی

## بخش ۴: تابع چگالی احتمال گاوسی (PDF)

```
def gaussian_pdf(x, mean, std):
    if std == 0:
        std = 0.0001  # Prevent division by zero
    
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    coefficient = 1 / (std * math.sqrt(2 * math.pi))
    pdf_value = coefficient * math.exp(exponent)
    
    return pdf_value
```
الگوریتم:

فرمول PDF گاوسی: f(x; μ, σ) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))

این تابع احتمال مشاهده مقدار x را با فرض توزیع نرمال با پارامترهای μ و σ می‌دهد

قسمت نمایی: اندازه‌گیری فاصله x از میانگین

قسمت ضریب: نرمال‌سازی برای تبدیل به احتمال

پیاده‌سازی:

محاسبه قسمت نمایی: -(x-μ)²/(2σ²)

محاسبه ضریب نرمال‌سازی: 1/(σ√(2π))

ضرب دو جزء برای بدست آوردن مقدار PDF

## بخش ۵: محاسبه درست‌نمایی (Likelihood)
```
p_x_apple = 1.0
for i, (x_val, mean, std) in enumerate(zip(x_test, apple_means, apple_stds)):
    pdf_val = gaussian_pdf(x_val, mean, std)
    p_x_apple *= pdf_val

# Similarly for banana
```
الگوریتم:

با فرض استقلال ویژگی‌ها: P(x|class) = Π P(x_i|class)

یعنی احتمال مشاهده همه ویژگی‌ها برابر حاصل‌ضرب احتمالات تک‌تک ویژگی‌هاست

هر P(x_i|class) با PDF گاوسی محاسبه می‌شود

پیاده‌سازی:

مقدار اولیه ۱.۰

برای هر ویژگی: محاسبه PDF و ضرب در مقدار قبلی

برای سیب: P(x|Apple) = P(weight=140|Apple) × P(sugar=15|Apple)

برای موز: p(x|Banana) = P(weight=140|Banana) × P(sugar=15|Banana)

## بخش ۶: احتمالات پیشین (Prior)

```
n_apple = len(apple)
n_banana = len(banana)
n_total = n_apple + n_banana

p_apple_prior = n_apple / n_total
p_banana_prior = n_banana / n_total
```
الگوریتم:

احتمالات پیشین بر اساس فراوانی کلاس‌ها در داده آموزشی

P(class) = تعداد_نمونه_های_کلاس / کل_نمونه‌ها

اگر داده‌ها متعادل باشند، هر کلاس احتمال پیشین برابر دارد

پیاده‌سازی:

هر کلاس ۴ نمونه دارد:

P(Apple) = 4/8 = 0.5

P(Banana) = 4/8 = 0.5

## بخش ۷: محاسبه احتمالات پسین (Posterior)

```
posterior_apple = p_x_apple * p_apple_prior
posterior_banana = p_x_banana * p_banana_prior
p_x_total = posterior_apple + posterior_banana

p_apple_given_x = posterior_apple / p_x_total
p_banana_given_x = posterior_banana / p_x_total
```
الگوریتم:

قاعده بیز:
P(class|x) = [P(x|class) × P(class)] / P(x)

P(x) ثابت نرمال‌سازی: P(x) = Σ P(x|class) × P(class) روی همه کلاس‌ها

ابتدا احتمال‌های غیرنرمال‌شده محاسبه، سپس بر مجموع تقسیم می‌شوند.

پیاده‌سازی:

posterior_apple = P(x|Apple) × P(Apple)

posterior_banana = P(x|Banana) × P(Banana)

p_x_total = posterior_apple + posterior_banana

نرمال‌سازی: تقسیم هر کدام بر مجموع

## بخش ۸: تصمیم‌گیری نهایی
```
if p_apple_given_x > p_banana_given_x:
    decision = "Apple"
    confidence = (p_apple_given_x - p_banana_given_x) / p_apple_given_x
else:
    decision = "Banana"
    confidence = (p_banana_given_x - p_apple_given_x) / p_banana_given_x
```
الگوریتم:

انتخاب کلاس با بیشترین احتمال پسین

محاسبه میزان اطمینان بر اساس تفاوت نسبی احتمالات

فرمول اطمینان: |P1 - P2| / max(P1, P2)

پیاده‌سازی:

مقایسه p_apple_given_x و p_banana_given_x

انتخاب کلاس برنده

محاسبه درصد اطمینان

## خروجی نمونه الگوریتم
برای میوه تست [140g, 15%]:

احتمال سیب: ≈ ۹۹٫۹۸%

احتمال موز: ≈ ۰٫۰۲%

تصمیم: Apple با اطمینان ≈ ۹۹٫۹۶%

## تحلیل ریاضی:

ویژگی وزن (۱۴۰ گرم):

سیب: μ≈۱۵۸٫۷۵، σ≈۸٫۵۴ → احتمال پایین

موز: μ≈۱۲۷٫۵، σ≈۶٫۴۵ → احتمال بسیار پایین

ویژگی قند (۱۵٪):

سیب: μ≈۱۲٫۱۲، σ≈۰٫۸۵ → احتمال بسیار پایین

موز: μ≈۱۸٫۲۵، σ≈۰٫۷۴ → احتمال متوسط

با توجه به ترکیب وزن و قند، میوه تست بیشتر شبیه سیب است.

## نقاط قوت الگوریتم

مناسب برای داده پیوسته: برخلاف Multinomial که برای داده‌های شمارشی است

پارامترهای کم: فقط نیاز به ذخیره μ و σ برای هر ویژگی

سریع در تست: محاسبات ساده و مستقیم

مقاوم در برابر نویز: توزیع گاوسی به داده‌های پرت کمتر حساس است

## محدودیت‌ها

فرض نرمال بودن: ویژگی‌ها باید تقریباً نرمال باشند

فرض استقلال: ویژگی‌ها در واقعیت ممکن است همبسته باشند

حساس به مقیاس: نیاز به نرمال‌سازی اگر ویژگی‌ها مقیاس‌های مختلفی دارند

مشکل با σ=0: اگر یک ویژگی در کلاسی ثابت باشد، احتمال صفر می‌شود
