import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

#%% Weibull Distribution
k = 10
lambda0 = 20
# Calculate the expected wait time
expected_wait_time = lambda0 * gamma(1 + 1/k)
print(expected_wait_time)

# Plot PDF
x = np.linspace(0, 30, 1000)
y0 = weibull_min.rvs(10)
y = weibull_min.pdf(x, k, scale=lambda0)

plt.figure(figsize=(5,5))
plt.plot(x, y, label=f"Shape: {k}, Scale: {lambda0}", color="black")
plt.fill_between(x, x*0, y, facecolor='g', alpha=0.2)
plt.xlabel("x")
plt.ylabel("PDF")
plt.savefig("weibull_pdf.png")
plt.legend()
plt.show()