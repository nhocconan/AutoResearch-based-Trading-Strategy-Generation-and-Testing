# Strategy: 1d_kama_rsi_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.394 | +8.8% | -10.2% | 60 | FAIL |
| ETHUSDT | -0.109 | +15.1% | -14.9% | 59 | FAIL |
| SOLUSDT | 0.504 | +50.9% | -12.9% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.173 | +8.1% | -11.1% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: On daily timeframe, capture trend continuation using KAMA direction filtered by RSI extremes and Choppiness index regime.
# KAMA adapts to market noise, reducing false signals. RSI >60 or <40 ensures momentum alignment. Chop >61.8 avoids strong trends where mean reversion fails.
# Works in bull/bear by following trend direction (KAMA) with momentum filter. Low trade frequency (~10-20/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA (adaptive moving average)
    # Efficiency Ratio = |change over 10 periods| / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close, n=10, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            direction = abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = direction / volatility if volatility > 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI < 40 (loss of momentum) OR Chop < 38.2 (strong trend - consider trend following instead)
            if (kama[i] < kama[i-1]) or (rsi[i] < 40) or (chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI > 60 (loss of momentum) OR Chop < 38.2
            if (kama[i] > kama[i-1]) or (rsi[i] > 60) or (chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require Chop > 61.8 (ranging market) for mean reversion logic
            if chop[i] > 61.8:
                # Long entry: KAMA turning up AND RSI > 50 (bullish momentum)
                if (kama[i] > kama[i-1]) and (rsi[i] > 50):
                    position = 1
                    signals[i] = 0.25
                # Short entry: KAMA turning down AND RSI < 50 (bearish momentum)
                elif (kama[i] < kama[i-1]) and (rsi[i] < 50):
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 12:43
