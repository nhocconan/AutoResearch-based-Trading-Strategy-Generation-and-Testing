# Strategy: 1d_kama_rsi_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.330 | +0.4% | -14.1% | 58 | FAIL |
| ETHUSDT | -0.421 | -12.2% | -39.8% | 69 | FAIL |
| SOLUSDT | 0.764 | +126.9% | -37.0% | 63 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.247 | +10.0% | -15.8% | 19 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
1d KAMA with RSI and chop filter
Long when KAMA trending up and RSI < 60 in choppy market (CHOP > 61.8)
Short when KAMA trending down and RSI > 40 in choppy market
Exit when KAMA changes direction
Designed for mean-reversion in chop and trend-following in trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA (10) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(n):
        if i == 0:
            er[i] = 0
        else:
            er[i] = change[i] / (volatility[i] + 1e-10) if volatility[i] != 0 else 0
    sc = (er * 0.2 + (1 - er) * 0.067) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # === 1w EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down
            if kama[i] < kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up
            if kama[i] > kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Choppy market: CHOP > 61.8 (range)
            if chop[i] > 61.8:
                # Mean reversion in chop
                if kama[i] > kama[i-1] and rsi[i] < 60:
                    position = 1
                    signals[i] = 0.25
                elif kama[i] < kama[i-1] and rsi[i] > 40:
                    position = -1
                    signals[i] = -0.25
            else:
                # Trending market: follow KAMA direction
                if kama[i] > kama[i-1] and ema_50_aligned[i] > ema_200_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif kama[i] < kama[i-1] and ema_50_aligned[i] < ema_200_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 23:24
