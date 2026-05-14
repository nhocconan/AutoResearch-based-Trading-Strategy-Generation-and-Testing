# Strategy: 4h_Combined_Signal_Trend_Momentum

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.018 | +17.7% | -11.5% | 401 | FAIL |
| ETHUSDT | 0.404 | +48.8% | -17.0% | 391 | PASS |
| SOLUSDT | 1.029 | +189.4% | -24.7% | 371 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.329 | +11.2% | -8.4% | 127 | PASS |
| SOLUSDT | -0.022 | +4.3% | -10.7% | 135 | FAIL |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4h_Combined_Signal_Trend_Momentum
# Hypothesis: Combining EMA trend, RSI momentum, and volume confirmation on 4h timeframe
# with 12h EMA filter reduces false signals and captures momentum in both bull and bear markets.
# Uses tight entry conditions to limit trades and avoid fee drag.

name = "4h_Combined_Signal_Trend_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === EMA Trend (4h) ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === RSI Momentum (4h) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Confirmation (4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(ema_20[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above EMA20, RSI > 50 and rising, above 12h EMA50, volume above average
            if (close[i] > ema_20[i] and 
                rsi[i] > 50 and rsi[i] > rsi[i-1] and 
                close[i] > ema_50_4h[i] and 
                vol_ratio[i] > 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below EMA20, RSI < 50 and falling, below 12h EMA50, volume above average
            elif (close[i] < ema_20[i] and 
                  rsi[i] < 50 and rsi[i] < rsi[i-1] and 
                  close[i] < ema_50_4h[i] and 
                  vol_ratio[i] > 1.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below EMA20 or trend change (below 12h EMA50) or RSI < 40
            if (close[i] < ema_20[i] or 
                close[i] < ema_50_4h[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above EMA20 or trend change (above 12h EMA50) or RSI > 60
            if (close[i] > ema_20[i] or 
                close[i] > ema_50_4h[i] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 07:01
