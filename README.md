# 🧠 Black-Scholes Option Pricer

This project implements a **European option pricing model** using the **Black-Scholes formula**. It's designed to help you understand how option prices behave with respect to volatility, time, and market parameters.

---

## 📌 Features

- ✅ Call and Put pricing using the Black-Scholes formula
- 📈 Visual plots of option price vs. volatility
- 📊 Comparison between call and put behavior
- 🧮 Modular, well-commented Python code
- 🧪 Great base for future extensions (Greeks, Monte Carlo, GUI)

---

## 🧠 Formula

```math
Call Option:
C = S * N(d1) - K * e^(-rT) * N(d2)

Put Option:
P = K * e^(-rT) * N(-d2) - S * N(-d1)

Where:
d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
d2 = d1 - σ * √T
