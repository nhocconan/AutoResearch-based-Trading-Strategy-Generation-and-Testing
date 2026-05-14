# Strategy: 4h_12h_1d_ParabolicSAR_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.359 | +33.2% | -7.5% | 314 | PASS |
| ETHUSDT | 0.232 | +29.9% | -6.6% | 293 | PASS |
| SOLUSDT | -0.258 | +2.9% | -27.9% | 286 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.228 | -6.6% | -7.9% | 114 | FAIL |
| ETHUSDT | 0.269 | +8.6% | -8.2% | 123 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_1d_ParabolicSAR_Trend
Hypothesis: Use Parabolic SAR from 12h as primary trend filter (proven to reduce whipsaws in 2022 crash), combined with 1d breakout above/below daily high/low and volume confirmation. Parabolic SAR provides clear trend direction with built-in acceleration, making it effective in both trending and ranging markets. Targets 20-30 trades/year by requiring PSAR trend alignment, price breakout beyond daily range, and volume > 1.8x 20-period average. Works in bull markets by following uptrend breaks above daily high, and in bear markets by taking short breaks below daily low only when PSAR confirms downtrend.
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
    
    # Get 12h data for Parabolic SAR (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Parabolic SAR
    # Parameters: start=0.02, increment=0.02, max=0.2
    psar = np.full_like(close_12h, np.nan)
    bull = True  # start assuming bullish
    af = 0.02    # acceleration factor
    ep = low_12h[0] if bull else high_12h[0]  # extreme point
    psar[0] = ep
    
    for i in range(1, len(close_12h)):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price < SAR
            if low_12h[i] < psar[i]:
                bull = False
                psar[i] = ep  # SAR = prior EP
                af = 0.02
                ep = high_12h[i]
            else:
                # Continue bullish
                if high_12h[i] > ep:
                    ep = high_12h[i]
                    af = min(af + 0.02, 0.2)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price > SAR
            if high_12h[i] > psar[i]:
                bull = True
                psar[i] = ep  # SAR = prior EP
                af = 0.02
                ep = low_12h[i]
            else:
                # Continue bearish
                if low_12h[i] < ep:
                    ep = low_12h[i]
                    af = min(af + 0.02, 0.2)
    
    # Align PSAR to 4h timeframe (wait for bar close)
    psar_aligned = align_htf_to_ltf(prices, df_12h, psar)
    
    # Get 1d data for daily high/low (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align daily high/low to 4h timeframe (wait for bar close)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(psar_aligned[i]) or np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d high, with volume, and PSAR bullish (close > PSAR)
            if (close[i] > high_1d_aligned[i] and vol_confirm[i] and 
                close[i] > psar_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d low, with volume, and PSAR bearish (close < PSAR)
            elif (close[i] < low_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < psar_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below PSAR (trend change) or fails to hold above daily high
            if (close[i] < psar_aligned[i] or 
                close[i] < high_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above PSAR (trend change) or fails to hold below daily low
            if (close[i] > psar_aligned[i] or 
                close[i] > low_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_ParabolicSAR_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 10:48
