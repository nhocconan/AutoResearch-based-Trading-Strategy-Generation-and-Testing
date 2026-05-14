# Strategy: 1h_4h_1d_trend_follow_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.596 | -2.8% | -12.5% | 1869 | FAIL |
| ETHUSDT | -0.258 | +5.5% | -15.1% | 1882 | FAIL |
| SOLUSDT | 0.521 | +68.0% | -23.7% | 1785 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.210 | +8.7% | -6.7% | 553 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1h_4h_1d_trend_follow_v2
# Hypothesis: 1-hour trend following with 4-hour and 1-day filters to reduce false signals.
# Long when: price > 4h EMA20, price > 1d EMA50, and 1h RSI > 50.
# Short when: price < 4h EMA20, price < 1d EMA50, and 1h RSI < 50.
# Exit when any condition fails.
# Uses 4h/1d for trend direction and 1h for entry timing to avoid overtrading.
# Target: 15-30 trades/year to minimize fee drag while capturing trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_follow_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1h RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_20 = np.zeros(len(close_4h))
    ema_4h_20[:] = np.nan
    ema_4h_20[19] = np.mean(close_4h[:20])
    for i in range(20, len(close_4h)):
        ema_4h_20[i] = close_4h[i] * 0.0952 + ema_4h_20[i-1] * 0.9048
    
    ema_4h_20_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.zeros(len(close_1d))
    ema_1d_50[:] = np.nan
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        ema_4h = ema_4h_20_aligned[i]
        ema_1d = ema_1d_50_aligned[i]
        rsi_val = rsi[i]
        
        if np.isnan(ema_4h) or np.isnan(ema_1d) or np.isnan(rsi_val):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price below 4h EMA20 OR below 1d EMA50 OR RSI < 50
            if price < ema_4h or price < ema_1d or rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price above 4h EMA20 OR above 1d EMA50 OR RSI > 50
            if price > ema_4h or price > ema_1d or rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry: price above both EMAs AND RSI > 50 (long)
            # Entry: price below both EMAs AND RSI < 50 (short)
            if price > ema_4h and price > ema_1d and rsi_val > 50:
                position = 1
                signals[i] = 0.20
            elif price < ema_4h and price < ema_1d and rsi_val < 50:
                position = -1
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-08 22:44
