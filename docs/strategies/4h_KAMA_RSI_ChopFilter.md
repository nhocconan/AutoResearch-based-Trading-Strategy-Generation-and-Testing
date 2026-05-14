# Strategy: 4h_KAMA_RSI_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.076 | +14.4% | -20.6% | 331 | FAIL |
| ETHUSDT | 0.052 | +20.7% | -19.0% | 312 | PASS |
| SOLUSDT | 0.909 | +154.8% | -29.1% | 285 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.025 | +5.5% | -10.1% | 106 | PASS |
| SOLUSDT | 0.097 | +6.7% | -16.5% | 99 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h KAMA + RSI + Chop Regime Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with RSI momentum and Chop filter to avoid
false signals in low volatility regimes. Designed for low trade frequency (<30/year)
to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""
name = "4h_KAMA_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Adaptive Moving Average) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(1, n):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Chop Index (14) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA + RSI > 50 + Chop < 61.8 (trending) + volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and
                chop[i] < 61.8 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 + Chop < 61.8 (trending) + volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and
                  chop[i] < 61.8 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40 OR Chop > 61.8 (ranging)
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60 OR Chop > 61.8 (ranging)
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 06:30
