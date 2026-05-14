# Strategy: 4h_PriceChannelBreakout_Volume_Confirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.071 | +14.9% | -19.2% | 117 | FAIL |
| ETHUSDT | 0.466 | +54.8% | -12.1% | 109 | PASS |
| SOLUSDT | 0.873 | +150.8% | -26.4% | 107 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.129 | +7.3% | -9.9% | 45 | PASS |
| SOLUSDT | 0.097 | +6.7% | -16.3% | 38 | PASS |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_PriceChannelBreakout_Volume_Confirm
Hypothesis: Breakouts of 20-period high/low with volume confirmation and ATR volatility filter.
Works in bull markets via upside breakouts, in bear markets via downside breakouts.
Target: 25-40 trades/year on 4h timeframe with disciplined entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels (highest high, lowest low)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # 14-period ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # 12h EMA34 trend filter (from higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_12h)):
        if i == 34:
            ema34_12h[i] = np.mean(close_12h[0:35])
        else:
            ema34_12h[i] = close_12h[i] * k + ema34_12h[i-1] * (1 - k)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-period high with volume spike and 12h uptrend
            if (close[i] > highest_high[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low with volume spike and 12h downtrend
            elif (close[i] < lowest_low[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below 20-period low or 12h trend turns down
            if (close[i] < lowest_low[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 20-period high or 12h trend turns up
            if (close[i] > highest_high[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannelBreakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 08:29
